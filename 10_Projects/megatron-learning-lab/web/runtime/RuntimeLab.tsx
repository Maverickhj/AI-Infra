import { useRef, useState } from "react";
import fixtures from "../../content/fixtures/runtime-reference.json";
import lesson from "../../content/cases/11_runtime_contracts.md?raw";
import { CourseDetails } from "../CourseText";
import { compareTraces, validateTrace, type ValidTrace } from "./trace";

type Slot = {
  text: string;
  value: ValidTrace | null;
  error: string;
  busy: boolean;
};
const blank = (): Slot => ({ text: "", value: null, error: "", busy: false });
const format = (x: number) => x.toPrecision(9);
function TraceView({ value }: { value: ValidTrace }) {
  const { trace: t, data, token_count } = value;
  const m = t.manifest,
    measured = t.measurements;
  const [row, setRow] = useState(0);
  const selected = Math.min(row, data.input_ids.length - 1);
  const lp =
    measured?.[t.task === "sft" ? "token_logprobs" : "current_logprobs"];
  const mask = data[t.task === "sft" ? "loss_mask" : "response_mask"];
  const download = () => {
    const url = URL.createObjectURL(
      new Blob([JSON.stringify(t, null, 2)], { type: "application/json" }),
    );
    const a = document.createElement("a");
    a.href = url;
    a.download = "validated-trace.json";
    a.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  };
  return (
    <div className="trace-result" data-testid="trace-result">
      <h3>{t.run_id}</h3>
      <p className="notice" data-testid="trace-provenance">
        <strong>{t.provenance}</strong> · {m.evidence_kind} ·{" "}
        {m.execution.status}
        <br />
        imported_claim：字段与数值合约通过；导入文件的来源声明未被网页独立核实。
      </p>
      <dl className="trace-facts">
        <dt>任务 / shape</dt>
        <dd>
          {t.task} · [{data.input_ids.length}, {data.input_ids[0].length}] ·{" "}
          {m.layout}
        </dd>
        <dt>有效 token</dt>
        <dd>{token_count ?? "derived 无测量计数"}</dd>
        <dt>模型 / 权重</dt>
        <dd>
          {m.model.id} · {m.model.revision}
          <br />
          {m.model.weights_origin} · {m.model.architecture_origin}
        </dd>
        <dt>Tokenizer</dt>
        <dd>
          {m.tokenizer.id} · {m.tokenizer.revision}
        </dd>
        <dt>Backend / dtype</dt>
        <dd>
          {m.backend} · {m.dtype}
        </dd>
        <dt>并行组</dt>
        <dd>
          TP{m.parallel.tp} / PP{m.parallel.pp} / DP{m.parallel.dp} / CP
          {m.parallel.cp} / EP{m.parallel.ep} · {m.parallel.groups_origin}
        </dd>
        <dt>采集边界</dt>
        <dd>
          {m.capture.scope} · timing={m.capture.timing}
        </dd>
        <dt>Input SHA256</dt>
        <dd>
          <code>{t.input_sha256}</code>
        </dd>
        <dt>Config SHA256</dt>
        <dd>
          <code>{t.config_sha256}</code>
        </dd>
      </dl>
      {measured && (
        <p data-testid="trace-loss">
          {t.task === "sft" ? (
            <>
              loss sum={format(measured.loss_sum)} / count=
              {measured.token_count} → mean={format(measured.loss_mean)}
            </>
          ) : (
            <>
              actor={format(measured.actor_loss)} · KL=
              {format(measured.kl_loss)} · policy={format(measured.policy_loss)}
              {measured.value_loss !== null && (
                <> · value={format(measured.value_loss)}</>
              )}
            </>
          )}
        </p>
      )}
      {t.task === "rl" && (
        <p data-testid="trace-versions">
          generation={data.policy_versions.generation} / previous=
          {data.policy_versions.previous} / current=
          {data.policy_versions.current} / after={data.policy_versions.after}
          <br />
          refit={measured?.refit?.status ?? "not_run"}
          {measured?.refit?.status === "acknowledged" && (
            <span data-testid="trace-refit-boundary">
              <br />
              文件记录 refit 调用完成；权重 hash 未核验。
            </span>
          )}
        </p>
      )}
      {lp && (
        <>
          <label>
            查看序列
            <select
              aria-label="查看序列"
              value={selected}
              onChange={(e) => setRow(Number(e.target.value))}
            >
              {data.input_ids.map((_: unknown, i: number) => (
                <option key={i} value={i}>
                  {i} ·{" "}
                  {data.trajectory_ids?.[i] ?? data.sample_ids?.[i] ?? "sample"}
                </option>
              ))}
            </select>
          </label>
          <div
            className="table-scroll"
            tabIndex={0}
            role="region"
            aria-label="导入 token 数值表"
          >
            <table data-testid="trace-tokens">
              <thead>
                <tr>
                  <th>位置</th>
                  <th>Input ID</th>
                  <th>{t.task === "sft" ? "Target ID" : "Action ID"}</th>
                  <th>mask</th>
                  <th>{t.task === "sft" ? "target" : "current"} logprob</th>
                </tr>
              </thead>
              <tbody>
                {data.input_ids[selected].map((id: number, j: number) => (
                  <tr key={j}>
                    <td>{j}</td>
                    <td>{id}</td>
                    <td>{t.task === "sft" ? data.labels[selected][j] : id}</td>
                    <td>{mask[selected][j]}</td>
                    <td>{format(lp[selected][j])}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
      <details>
        <summary>查看实际来源声明与未采集项</summary>
        <p>
          以下路径、版本和命令均是文件中的文本，不会在浏览器执行或自动访问。
        </p>
        <pre>
          {JSON.stringify(
            {
              runtime_sources: m.runtime_sources,
              software: m.software,
              execution: m.execution,
              capture: m.capture,
              limitations: m.limitations,
            },
            null,
            2,
          )}
        </pre>
      </details>
      <details>
        <summary>查看完整显式配置与输入</summary>
        <pre>{t.config_json}</pre>
        <pre>{t.input_json}</pre>
      </details>
      <button onClick={download}>导出已校验 trace</button>
    </div>
  );
}

export function RuntimeLab({
  onSource,
}: {
  onSource: (id: string, excerpt?: string) => void;
}) {
  const [slots, setSlots] = useState<Slot[]>([blank(), blank()]);
  const versions = useRef([0, 0]);
  const replace = (index: number, patch: Partial<Slot>) =>
    setSlots((old) =>
      old.map((s, i) => (i === index ? { ...s, ...patch } : s)),
    );
  const read = async (
    index: number,
    text: string,
    ticket = ++versions.current[index],
  ) => {
    replace(index, { text, value: null, error: "", busy: true });
    try {
      const value = await validateTrace(text);
      if (ticket === versions.current[index])
        replace(index, { value, busy: false });
    } catch (error) {
      if (ticket === versions.current[index])
        replace(index, {
          value: null,
          busy: false,
          error: error instanceof Error ? error.message : "无效 trace",
        });
    }
  };
  const file = async (index: number, selected?: File) => {
    if (!selected) return;
    const ticket = ++versions.current[index];
    replace(index, { text: "", value: null, error: "", busy: true });
    try {
      if (selected.size > 1048576) throw Error("trace exceeds 1 MiB");
      const text = await selected.text();
      if (ticket === versions.current[index]) await read(index, text, ticket);
    } catch (error) {
      if (ticket === versions.current[index])
        replace(index, { error: String(error), busy: false });
    }
  };
  let comparison: ReturnType<typeof compareTraces> | null = null,
    compareError = "";
  if (slots[0].value && slots[1].value) {
    try {
      comparison = compareTraces(slots[0].value, slots[1].value);
    } catch (error) {
      compareError = error instanceof Error ? error.message : "不可比较";
    }
  }
  return (
    <section className="panel runtime-lab" aria-label="只读运行对照">
      <div className="section-header">
        <div>
          <span className="eyebrow">IMPORT · INSPECT · COMPARE</span>
          <h2>把同一批输入的证据放在一起</h2>
        </div>
        <span className="badge">本地文件 · 只读</span>
      </div>
      <p>
        从 SFT 的 target / mask 或 RL 的 action / policy
        版本出发。每份文件单独保留来源；这里只检查数值与字段，不启动训练，也不把
        reference 升格为 observed。
      </p>
      <p className="muted">
        内置样例来自实际执行的 authored CPU 模型。真实 HF、Bridge SFT 和 NeMo RL
        仍待资源授权及验证。
      </p>
      <div className="trace-slots">
        {slots.map((slot, i) => (
          <article
            className="trace-slot"
            aria-label={`Trace ${i === 0 ? "A" : "B"}`}
            key={i}
          >
            <h3>Trace {i === 0 ? "A" : "B"}</h3>
            <div className="trace-actions">
              <button
                onClick={() =>
                  void read(i, JSON.stringify(fixtures.traces[0], null, 2))
                }
              >
                载入 SFT CPU 样例
              </button>
              <button
                onClick={() =>
                  void read(i, JSON.stringify(fixtures.traces[1], null, 2))
                }
              >
                载入 PPO CPU 样例
              </button>
            </div>
            <label>
              选择 trace JSON（最多 1 MiB）
              <input
                type="file"
                accept=".json,application/json"
                onChange={(e) => {
                  void file(i, e.target.files?.[0]);
                  e.target.value = "";
                }}
              />
            </label>
            <label>
              粘贴 trace JSON
              <textarea
                rows={5}
                spellCheck={false}
                value={slot.text}
                onChange={(e) => {
                  versions.current[i]++;
                  replace(i, {
                    text: e.target.value,
                    value: null,
                    error: "",
                    busy: false,
                  });
                }}
              />
            </label>
            <div className="trace-actions">
              <button
                disabled={!slot.text || slot.busy}
                onClick={() => void read(i, slot.text)}
              >
                校验并载入
              </button>
              <button
                onClick={() => {
                  versions.current[i]++;
                  replace(i, blank());
                }}
              >
                清空
              </button>
            </div>
            {slot.busy && <p role="status">正在校验文件内容…</p>}
            {slot.error && (
              <p role="alert" className="notice">
                拒绝导入：{slot.error}
              </p>
            )}
            {slot.value && (
              <TraceView
                key={slot.value.trace.run_id + slot.value.trace.input_sha256}
                value={slot.value}
              />
            )}
          </article>
        ))}
      </div>
      <section className="trace-compare" aria-label="Trace 比较结果">
        <h3>同一数据定义的数值差</h3>
        {!comparison && !compareError && (
          <p>
            先载入两份 trace。任务、输入/mask/版本 hash、模型与 tokenizer
            身份一致后才比较。
          </p>
        )}
        {compareError && <p role="status">不能逐 token 比较：{compareError}</p>}
        {comparison && (
          <>
            <p data-testid="trace-difference">
              {comparison.field} · 最大绝对差=
              {comparison.max_abs_difference.toExponential(6)}
            </p>
            <p>
              {comparison.evidence.join(" ↔ ")} · {comparison.claim}
            </p>
            <p>
              配置 hash{" "}
              {slots[0].value!.trace.config_sha256 ===
              slots[1].value!.trace.config_sha256
                ? "一致"
                : "不同；请展开两侧显式配置"}
              。差值不是兼容性通过结论；梯度、更新、恢复与生产引擎仍需各自的运行证据。
            </p>
          </>
        )}
      </section>
      <div className="trace-actions">
        <button onClick={() => onSource("B-STEP")}>
          查看 Bridge forward 参考源码
        </button>
        <button onClick={() => onSource("R-LOSS", "rl-ratio-clip")}>
          查看 RL loss 参考源码
        </button>
      </div>
      <CourseDetails
        className="runtime-course"
        summary="精讲：接入接口、单次 shift、采集边界与来源合约"
        text={lesson}
      />
    </section>
  );
}
