import { CourseDetails } from "../CourseText";
import { memo, useMemo } from "react";
import model from "../../content/fixtures/decoder-reference.json";
import data from "../../content/fixtures/sft-data.json";
import course from "../../content/cases/08_pipeline_sequence.md?raw";
import type { State } from "../data";
import { align, tokenize } from "../sft/compute";
import { computeDecoder } from "../decoder/compute";
import { pipeline, cpCompute, spCompute } from "./compute";
const shape = (v: number[]) => "[" + v.join(",") + "]",
  fixed = (v: number) => v.toFixed(8);
export const SequenceJourney = memo(function SequenceJourney({
  state,
  patch,
  onSource,
}: {
  state: State;
  patch: (s: Partial<State>) => void;
  onSource: (id?: string) => void;
}) {
  const schedule = useMemo(
    () => pipeline(state.pp, state.microbatches),
    [state.pp, state.microbatches],
  );
  const mb = Math.min(state.pipelineMicrobatch, state.microbatches - 1);
  const ctx = useMemo<{
    good?: ReturnType<typeof cpCompute>;
    actual?: ReturnType<typeof cpCompute>;
    error: string;
  }>(() => {
    try {
      return {
        good: cpCompute(model, data, state.sequenceLayout, state.sftPadding, 1),
        actual: cpCompute(
          model,
          data,
          state.sequenceLayout,
          state.sftPadding,
          state.cp,
          state.sequenceFault,
        ),
        error: "",
      };
    } catch (e) {
      return { error: (e as Error).message };
    }
  }, [state.sequenceLayout, state.sftPadding, state.cp, state.sequenceFault]);
  const g = ctx.good,
    a = ctx.actual;
  const rank = Math.min(state.sequenceRank, state.cp - 1);
  const owned = a?.shards[rank] || [];
  const q = owned.includes(state.sequenceQuery)
    ? state.sequenceQuery
    : [...owned].reverse().find((i) => a?.valid[i]) || 0;
  const head = state.gqaHead,
    merge = a?.reductions[q]?.[head];
  const decoder = useMemo(() => {
    const rows = align(
      tokenize(data, data.samples[0], "assistant"),
      data.samples[0].id,
    );
    return computeDecoder(
      model,
      rows.map((r) => r.input.id),
      rows.map((r) => Math.max(0, r.target.id)),
      rows.map((r) => r.mask),
    );
  }, []);
  const sp = useMemo(
    () =>
      spCompute(
        model,
        decoder.layers[state.tinyLayer].attention,
        state.tinyLayer,
        state.tp,
      ),
    [state.tinyLayer, state.tp],
  );
  const spError = Math.max(
    ...sp.output.flatMap((row, t) =>
      row.map((v, i) =>
        Math.abs(v - decoder.layers[state.tinyLayer].down[t][i]),
      ),
    ),
  );
  const cpError =
    g && a
      ? Math.max(
          ...a.outputs.flatMap((row, t) =>
            row.flatMap((h, n) =>
              h.map((v, i) => Math.abs(v - g.outputs[t][n][i])),
            ),
          ),
        )
      : 0;
  function download() {
    const payload = {
      provenance: "reference_simulation",
      timing: "logical_units_not_wall_time",
      schedule,
      selected_microbatch: mb,
      cp:
        g && a
          ? {
              layout: state.sequenceLayout,
              padding: state.sftPadding,
              cp: state.cp,
              rank,
              query: q,
              head,
              fault: state.sequenceFault,
              cu_seqlens: a.cuSeqlens,
              cu_seqlens_padded: a.cuSeqlensPadded,
              owned_indices: owned,
              visible_keys: a.keys[q],
              output: a.outputs[q],
              full_context_output: g.outputs[q],
              max_error: cpError,
            }
          : null,
      sp: { tp: state.tp, layer: state.tinyLayer, max_error: spError },
    };
    const url = URL.createObjectURL(
      new Blob([JSON.stringify(payload, null, 2)], {
        type: "application/json",
      }),
    );
    const el = document.createElement("a");
    el.href = url;
    el.download = "sequence-reference.json";
    el.click();
    URL.revokeObjectURL(url);
  }
  return (
    <section
      className="panel sequence-reference"
      aria-label="PP SP CP sequence journey"
    >
      <div className="section-header">
        <h2>层的时间调度与序列分片</h2>
        <span className="badge">reference · 逻辑时间</span>
      </div>
      <p>
        PP 绑定同一两层模型；SP 检查其完整上下文 FFN；CP 使用 Layer 0 的同一组
        QKV/RoPE 权重和 G02 样本。以下是分别验证的参考机制，未执行联合 PP/SP/CP
        后端。
      </p>
      <h3>Non-interleaved 1F1B</h3>
      <div className="sft-controls">
        <label>
          PP
          <select
            aria-label="PP"
            value={state.pp}
            onChange={(e) => patch({ pp: Number(e.target.value) })}
          >
            <option value={1}>1</option>
            <option value={2}>2</option>
          </select>
        </label>
        <label>
          Microbatches
          <input
            aria-label="Microbatches"
            type="number"
            min={1}
            max={8}
            value={state.microbatches}
            onChange={(e) => patch({ microbatches: Number(e.target.value) })}
          />
        </label>
        <label>
          选定 microbatch
          <select
            aria-label="选定 microbatch"
            value={mb}
            onChange={(e) =>
              patch({ pipelineMicrobatch: Number(e.target.value) })
            }
          >
            {Array.from({ length: state.microbatches }, (_, m) => (
              <option key={m} value={m}>
                {m}
              </option>
            ))}
          </select>
        </label>
      </div>
      <p>
        每层 F 或 B
        各耗1逻辑单位，embedding/head开销忽略，通信零耗时但保持依赖；一个stage一次一个计算，无VPP、overlap或重计算。PP1每次F/B含两层，所以耗2单位。
      </p>
      <div className="sft-table-scroll" tabIndex={0}>
        <table data-testid="pipeline-timeline">
          <thead>
            <tr>
              <th>Stage / 层与模块</th>
              {Array.from({ length: schedule.duration }, (_, t) => (
                <th key={t}>{t}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {Array.from({ length: state.pp }, (_, s) => (
              <tr key={s}>
                <th>
                  stage{s}:{" "}
                  {state.pp === 1
                    ? "embedding + L0,L1 + final norm/head"
                    : s === 0
                      ? "embedding + L0"
                      : "L1 + final norm/head"}
                </th>
                {Array.from({ length: schedule.duration }, (_, t) => {
                  const e = schedule.events.find(
                    (e) => e.stage === s && e.start <= t && t < e.end,
                  );
                  return (
                    <td
                      key={t}
                      className={e ? "pipeline-" + e.kind : "pipeline-idle"}
                    >
                      {e ? (
                        <button
                          aria-pressed={e.microbatch === mb}
                          title={e.phase}
                          onClick={() =>
                            patch({ pipelineMicrobatch: e.microbatch })
                          }
                        >
                          {e.kind}
                          {e.microbatch}
                        </button>
                      ) : (
                        "·"
                      )}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p data-testid="pipeline-cost">
        makespan={schedule.duration} 逻辑单位；idle stage-slots=
        {schedule.idle_slots}；逻辑 bubble={(100 * schedule.bubble).toFixed(2)}
        %。这不是实测GPU利用率。
      </p>
      <p data-testid="pipeline-lifetime">
        microbatch {mb}：
        {schedule.lifetimes
          .filter((l) => l.microbatch === mb)
          .map(
            (l) =>
              "stage" + l.stage + " activation [" + l.start + "," + l.end + ")",
          )
          .join("；")}
        。分配于F开始、释放于B结束；各stage峰值保留数=
        {shape(schedule.peak_activations)}
        。这里只计待反传microbatch，不估算实际内存字节。
      </p>
      <button onClick={() => onSource("C-SCHEDULE")}>
        查看 1F1B 调度源码 ↗
      </button>
      <h3>SP：TP group 内的 sequence layout</h3>
      <p>
        固定 arithmetic-multiturn / assistant，20个有效输入，当前 tiny layer=
        {state.tinyLayer}、TP={state.tp}（沿用上方TP控件）。来自完整 G03
        attention
        residual，RMSNorm按token独立；head/feature分片与sequence分片是不同维度。
      </p>
      <div className="sft-table-scroll" tabIndex={0}>
        <table data-testid="sp-layout">
          <thead>
            <tr>
              <th>阶段</th>
              <th>每rank shape</th>
              <th>含义</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>local norm</td>
              <td>{shape([20 / state.tp, 8])}</td>
              <td>各rank不同sequence行，H完整</td>
            </tr>
            <tr>
              <td>all-gather sequence</td>
              <td>[20,8]</td>
              <td>复制完整norm行，再做成对column FFN</td>
            </tr>
            <tr>
              <td>local product</td>
              <td>{shape([20, 12 / state.tp])}</td>
              <td>所有sequence，部分FFN通道</td>
            </tr>
            <tr>
              <td>row partial</td>
              <td>[20,8]</td>
              <td>同一位置的各rank贡献</td>
            </tr>
            <tr>
              <td>SUM + reduce-scatter</td>
              <td>{shape([20 / state.tp, 8])}</td>
              <td>恢复完整hidden并重新切sequence</td>
            </tr>
          </tbody>
        </table>
      </div>
      <p data-testid="sp-result">
        重组与 dense FFN down 最大误差={spError.toExponential(3)}；rank0首token=
        {sp.scattered[0][0].map(fixed).join(", ")}
        。all-gather反向在TP后续路径上为reduce-scatter；row
        reduce-scatter反向为all-gather，不能重复平均。
      </p>
      <button onClick={() => onSource("C-TPMAP")}>
        查看 SP gather / reduce-scatter 源码 ↗
      </button>
      <h3>CP：远端 KV 与 packed 边界</h3>
      <div className="sft-controls">
        <label>
          Sequence layout
          <select
            aria-label="Sequence layout"
            value={state.sequenceLayout}
            onChange={(e) =>
              patch({
                sequenceLayout: e.target.value as State["sequenceLayout"],
              })
            }
          >
            <option value="ordinary">普通单样本</option>
            <option value="thd">THD packed 两样本</option>
          </select>
        </label>
        <label>
          CP
          <select
            aria-label="CP"
            value={state.cp}
            onChange={(e) =>
              patch({ cp: Number(e.target.value), sequenceRank: 0 })
            }
          >
            <option value={1}>1</option>
            <option value={2}>2</option>
          </select>
        </label>
        <label>
          序列 padding
          <select
            aria-label="序列 padding"
            value={state.sftPadding}
            onChange={(e) => patch({ sftPadding: Number(e.target.value) })}
          >
            <option value={1}>1（不补齐）</option>
            <option value={8}>8</option>
          </select>
        </label>
        <label>
          CP rank
          <select
            aria-label="CP rank"
            value={rank}
            onChange={(e) => patch({ sequenceRank: Number(e.target.value) })}
          >
            {Array.from({ length: state.cp }, (_, r) => (
              <option key={r} value={r}>
                {r}
              </option>
            ))}
          </select>
        </label>
        <label>
          CP 反例
          <select
            aria-label="CP 反例"
            value={state.sequenceFault}
            onChange={(e) =>
              patch({ sequenceFault: e.target.value as State["sequenceFault"] })
            }
          >
            <option value="none">完整上下文</option>
            <option value="local_kv">丢弃远端 KV</option>
            <option value="leak">跨样本 attention 泄漏</option>
          </select>
        </label>
      </div>
      {ctx.error ? (
        <p role="alert">{ctx.error}</p>
      ) : (
        g &&
        a && (
          <>
            <p data-testid="cp-metadata">
              cu_seqlens={shape(a.cuSeqlens)}；cu_seqlens_padded=
              {shape(a.cuSeqlensPadded)}；
              {state.sequenceLayout === "thd"
                ? "逐文档 zigzag；position 每样本重置"
                : "整条序列 zigzag"}
              。CP rank {rank} 的物理 token 索引={shape(owned)}；Q shape=
              {shape([owned.length, 4, 4])}，本地 K/V=
              {shape([owned.length, 2, 4])}。
            </p>
            <div className="sft-controls">
              <label>
                CP query
                <select
                  aria-label="CP query"
                  value={q}
                  onChange={(e) =>
                    patch({ sequenceQuery: Number(e.target.value) })
                  }
                >
                  {owned.map((t) => (
                    <option key={t} value={t}>
                      {t} · {a.valid[t] ? "有效" : "padding"} · pos
                      {a.positions[t]}
                    </option>
                  ))}
                </select>
              </label>
              <label>
                CP Q head
                <select
                  aria-label="CP Q head"
                  value={head}
                  onChange={(e) => patch({ gqaHead: Number(e.target.value) })}
                >
                  {[0, 1, 2, 3].map((h) => (
                    <option key={h} value={h}>
                      {h}
                    </option>
                  ))}
                </select>
              </label>
            </div>
            <p data-testid="cp-query">
              query={q}、position={a.positions[q]}、sample={a.segments[q]}
              、valid={String(a.valid[q])}。
              {!a.valid[q]
                ? "padding query 不计算 softmax，不伪造全mask结果。"
                : "每个query需同一样本中全部历史KV，而不是仅本rank的历史。"}{" "}
              head {head} → KV group {Math.floor(head / 2)}。
            </p>
            <div className="sft-table-scroll" tabIndex={0}>
              <table data-testid="cp-keys">
                <thead>
                  <tr>
                    <th>KV owner</th>
                    <th>可见的物理key索引</th>
                    <th>局部max / exp sum</th>
                  </tr>
                </thead>
                <tbody>
                  {a.shards.map((_, r) => (
                    <tr key={r}>
                      <td>
                        {r === rank ? "local" : "remote"} rank{r}
                      </td>
                      <td>{shape(a.keys[q][r])}</td>
                      <td>
                        {merge
                          ? String(merge.partials[r].max) +
                            " / " +
                            fixed(merge.partials[r].sum)
                          : "not computed"}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p data-testid="cp-result">
              选定head输出={a.outputs[q][head].map(fixed).join(", ")}
              ；完整KV参考={g.outputs[q][head].map(fixed).join(", ")}
              ；全部query/head最大偏差={cpError.toExponential(3)}。
              {state.sequenceFault === "none"
                ? "正确合并局部softmax统计量；没有平均各块的归一化输出。"
                : "反例改变计算；此结果不代表合法CP。"}{" "}
            </p>
          </>
        )
      )}
      <button onClick={() => onSource("C-COREUTIL")}>查看 CP 分片源码 ↗</button>
      <button onClick={download}>导出 sequence 参考</button>
      <CourseDetails
        className="decoder-course"
        summary="精讲：1F1B、SP/CP 与 THD · 数学符号表"
        text={course}
      />
    </section>
  );
});
