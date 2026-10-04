import { CourseDetails } from "../CourseText";
import { memo, useMemo, useState } from "react";
import fixture from "../../content/fixtures/sft-data.json";
import course from "../../content/cases/05_sft_data.md?raw";
import type { State } from "../data";
import {
  tokenize,
  align,
  batch,
  referenceLoss,
  parseTokenizerTrace,
  type Row,
} from "./compute";

const fmt = (n: number | null) => (n === null ? "无有效监督" : n.toFixed(6));
function TokenTable({
  rows,
  losses,
  onSelect,
}: {
  rows: Row[];
  losses?: number[];
  onSelect?: (i: number) => void;
}) {
  return (
    <div
      className="sft-table-scroll"
      tabIndex={0}
      aria-label="逐 token 对齐表，可横向滚动"
    >
      <table data-testid="sft-tokens">
        <thead>
          <tr>
            <th>sample / t</th>
            <th>input xₜ</th>
            <th>target xₜ₊₁</th>
            <th>label</th>
            <th>target mask</th>
            <th>CE</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr
              key={i}
              data-mask={r.mask}
              data-valid={r.valid}
              data-boundary={r.position === 0}
            >
              <td>
                {onSelect ? (
                  <button
                    aria-label={"查看 query " + i}
                    onClick={() => onSelect(i)}
                  >
                    {r.sample} / {r.position}
                  </button>
                ) : (
                  r.position
                )}
              </td>
              <td>
                <code>{r.input.text}</code>
                <small>id {r.input.id}</small>
              </td>
              <td>
                <code>{r.target.text}</code>
                <small>
                  {r.target.id >= 0 ? "id " + r.target.id : "无下一个目标"}
                </small>
              </td>
              <td>{r.label}</td>
              <td>{r.mask}</td>
              <td>{losses ? fmt(losses[i]) : "未运行"}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
export const SftDataJourney = memo(function SftDataJourney({
  state,
  patch,
  onSource,
}: {
  state: State;
  patch: (s: Partial<State>) => void;
  onSource: (id?: string) => void;
}) {
  const [fault, setFault] = useState("once");
  const [traceText, setTraceText] = useState("");
  const [imported, setImported] = useState<ReturnType<
    typeof parseTokenizerTrace
  > | null>(null);
  const [error, setError] = useState("");
  const selected = state.sftLayout === "single" ? [state.sample] : [1, 2];
  const sequences = useMemo(
    () =>
      selected.map((i) =>
        align(
          tokenize(fixture, fixture.samples[i], state.mode, state.sftLimit),
          fixture.samples[i].id,
        ),
      ),
    [state.sample, state.mode, state.sftLimit, state.sftLayout],
  );
  const width =
    Math.ceil(Math.max(...sequences.map((s) => s.length)) / state.sftPadding) *
    state.sftPadding;
  const result = useMemo(
    () =>
      batch(
        sequences,
        state.sftPadding,
        false,
        state.sftLayout === "unpacked" ? width : 0,
      ),
    [sequences, state.sftPadding, state.sftLayout, width],
  );
  const loss = useMemo(
    () => referenceLoss(result, fixture.vocabulary.length),
    [result],
  );
  const singles = useMemo(
    () =>
      sequences.map((s) =>
        referenceLoss(batch([s]), fixture.vocabulary.length),
      ),
    [sequences],
  );
  const leaky = useMemo(
    () =>
      referenceLoss(
        batch(sequences, state.sftPadding, true),
        fixture.vocabulary.length,
      ),
    [sequences, state.sftPadding],
  );
  const query = Math.min(state.sftQuery, result.rows.length - 1);
  const selectedRow = result.rows[query];
  const physical = result.cuSeqlensPadded;
  function download() {
    const data = {
      schema_version: 1,
      provenance: fixture.provenance,
      execution: "reference",
      model_execution: "not_run",
      fixture: "content/fixtures/sft-data.json",
      mode: state.mode,
      layout: state.sftLayout,
      limit: state.sftLimit,
      padding: state.sftPadding,
      messages: selected.map((i) => fixture.samples[i].messages),
      ...result,
      loss,
    };
    const url = URL.createObjectURL(
      new Blob([JSON.stringify(data, null, 2)], { type: "application/json" }),
    );
    const a = document.createElement("a");
    a.href = url;
    a.download = "sft-data-reference.json";
    a.click();
    URL.revokeObjectURL(url);
  }
  return (
    <section className="sft-journey" aria-label="SFT 数据演算">
      <div className="section-header">
        <h3>从 messages 到有效目标</h3>
        <span className="badge">authored token fixture · reference</span>
      </div>
      <p>
        上方原文按人工片段拆分。这里的数字只索引教学词表，不是 Qwen token
        IDs；真实 chat template/tokenizer 待 R01 验证。空回复依本 fixture
        约定仍监督 EOS。
      </p>
      <div className="sft-controls">
        <label>
          数据布局
          <select
            value={state.sftLayout}
            onChange={(e) =>
              patch({
                sftLayout: e.target.value as State["sftLayout"],
                sftQuery: 0,
              })
            }
          >
            <option value="single">当前样本</option>
            <option value="unpacked">两个长短样本 · unpacked</option>
            <option value="packed">两个长短样本 · packed</option>
          </select>
        </label>
        <label>
          保留 token 上限
          <input
            type="number"
            min={2}
            max={128}
            value={state.sftLimit}
            onChange={(e) =>
              patch({ sftLimit: Number(e.target.value), sftQuery: 0 })
            }
          />
        </label>
        <label>
          对齐 padding
          <select
            value={state.sftPadding}
            onChange={(e) => patch({ sftPadding: Number(e.target.value) })}
          >
            <option value={1}>不补齐</option>
            <option value={8}>8 的倍数</option>
          </select>
        </label>
      </div>
      {state.sftLayout !== "single" && (
        <details>
          <summary>本次两个样本的原始 messages</summary>
          {selected.map((i) => (
            <div key={i}>
              <h4>{fixture.samples[i].id}</h4>
              {fixture.samples[i].messages.map((m, j) => (
                <p key={j}>
                  {m.role}: {m.content}
                </p>
              ))}
            </div>
          ))}
        </details>
      )}
      <p data-testid="sft-layout">
        {state.sftLayout === "packed"
          ? "THD 教学布局：物理 microbatch=1，包含 2 个独立样本"
          : state.sftLayout === "unpacked"
            ? "unpacked 教学布局：[B,S]=[2," + width + "]，两行分别 attention"
            : "单样本教学布局：[1," + result.rows.length + "]"}
        。position 在每个样本重置；图中 true 表示允许注意。
      </p>
      <div className="sft-boundaries" aria-label="样本物理边界">
        {sequences.map((s, i) => (
          <div key={i}>
            <strong>{s[0].sample}</strong>
            <span>
              物理 [{physical[i]}, {physical[i + 1]})
            </span>
            <span>
              有效长度 {s.length} / padding{" "}
              {physical[i + 1] - physical[i] - s.length}
            </span>
          </div>
        ))}
      </div>
      <p data-testid="sft-metadata">
        cu_seqlens_q/kv（有效）= [{result.cuSeqlens.join(", ")}]<br />
        cu_seqlens_q/kv_padded（物理）= [{physical.join(", ")}]
      </p>
      <p className="muted">
        unpacked 时以上累计长度仅用于并排对照；THD 才向 backend 传 packed
        metadata。这里未调用 fused kernel。
      </p>
      <div className="sft-summary" data-testid="sft-summary">
        <strong>有效目标 N = {loss.count}</strong>
        <span>Σ(mask × CE) = {fmt(loss.sum)}</span>
        <span>全局均值 = {fmt(loss.mean)}</span>
        <span>
          逐样本 sum/count 合并 = {fmt(singles.reduce((a, s) => a + s.sum, 0))}{" "}
          / {singles.reduce((a, s) => a + s.count, 0)}
        </span>
      </div>
      {loss.count === 0 && (
        <p className="notice">
          no_supervision：截断后没有有效 target，均值未定义；不除以零，不伪造为
          loss=0。last_turn 不回退监督之前的回复。
        </p>
      )}
      <p className="muted">
        CE 列先按可见的下一 token 计算教学参考，再由 mask 选择；不代表 backend
        对 ignore_index 返回了这些值。
      </p>
      <TokenTable
        rows={result.rows}
        losses={loss.losses}
        onSelect={(i) => patch({ sftQuery: i })}
      />
      <div className="exercise">
        <h4>query {query} 的可见上下文</h4>
        <p>
          {selectedRow.sample} / position {selectedRow.position}，
          {selectedRow.valid
            ? "允许读取同一样本中不晚于自身的有效 key"
            : "padding query 不参与本参考计算"}
          。
        </p>
        <div className="sft-keys" data-testid="sft-keys">
          {result.rows.map((r, j) => (
            <span
              key={j}
              data-allowed={result.attention[query][j]}
              title={r.sample + " / " + r.position}
            >
              {j}: {r.input.text}
            </span>
          ))}
        </div>
      </div>
      <div className="exercise">
        <h4>错误实现对照</h4>
        <label>
          移位反例
          <select value={fault} onChange={(e) => setFault(e.target.value)}>
            <option value="once">正确：一次 shift</option>
            <option value="twice">反例：重复 shift</option>
            <option value="leak">反例：跨样本 attention</option>
            <option value="mean">反例：局部均值之均值</option>
          </select>
        </label>
        <p role="status">
          {fault === "once"
            ? "目标与输入相差一个位置；mask 对齐 target。dataset/collator 已移位时，后续不能再移位。"
            : fault === "twice"
              ? "错误：x₀ 应预测 " +
                result.rows[0].target.text +
                "，重复 shift 却预测 " +
                (result.rows[1]?.target.text || "无目标") +
                "；单次 shift 合约拒绝。"
              : fault === "leak"
                ? sequences.length === 1
                  ? "请选择两个样本观察泄漏反例。"
                  : "错误：跨样本 key 泄漏后 loss sum = " +
                    fmt(leaky.sum) +
                    "，正确值 = " +
                    fmt(loss.sum)
                : singles.length === 1
                  ? "单样本两种归约相同；请选择两个不同有效长度的样本比较。"
                  : singles.every((s) => s.mean !== null)
                    ? "错误：局部均值之均值 = " +
                      fmt(
                        singles.reduce((a, s) => a + (s.mean || 0), 0) /
                          singles.length,
                      ) +
                      "；正确全局 sum/count = " +
                      fmt(loss.mean)
                    : "无有效监督时不能计算局部均值。"}
        </p>
      </div>
      <div className="sft-controls">
        {[
          ["B-DIRECTSFT", "Dataset → collator"],
          ["B-SFTCOLLATE", "collator → shift → pack"],
          ["B-CONVERSATION", "单次移位实现"],
          ["B-PACK", "THD 边界实现"],
          ["B-STEP", "gpt_step 调用"],
          ["B-LOSS", "loss sum/count"],
        ].map(([id, label]) => (
          <button key={id} onClick={() => onSource(id)}>
            {label} ↗
          </button>
        ))}
        <button onClick={download}>导出数据参考 JSON</button>
      </div>
      <details>
        <summary>读取外部 tokenizer trace（待实测核验）</summary>
        <p>
          只读取 JSON，不执行代码。要求
          schema_version=1、kind=tokenizer_trace、model_id、40 位
          tokenizer_revision、64 位
          template_sha256、loss_mode、rendered_text、messages、未移位
          tokens（id/text）及等长 loss_mask。导入只验证合约，不证明来源真实；R01
          再核验官方结果。
        </p>
        <label>
          Tokenizer trace JSON
          <textarea
            value={traceText}
            maxLength={200000}
            onChange={(e) => {
              setTraceText(e.target.value);
              setImported(null);
              setError("");
            }}
          />
        </label>
        <button
          onClick={() => {
            try {
              setImported(parseTokenizerTrace(traceText));
              setError("");
            } catch (e) {
              setImported(null);
              setError(String(e));
            }
          }}
        >
          读取 tokenizer trace
        </button>
        {error && <p role="alert">{error}</p>}
        {imported && (
          <div data-testid="tokenizer-trace">
            <p>
              {imported.provenance} · {imported.model} · {imported.mode}
            </p>
            <p>
              revision {imported.revision}
              <br />
              template SHA256 {imported.templateHash}
            </p>
            <pre>{imported.rendered}</pre>
            {imported.messages.map((m, i) => (
              <p key={i}>
                {m.role}: {m.content}
              </p>
            ))}
            <TokenTable rows={imported.rows} />
          </div>
        )}
      </details>
      <CourseDetails
        className="sft-course"
        summary="精讲：监督、packing 与独立数学符号表"
        text={course}
      />
    </section>
  );
});
