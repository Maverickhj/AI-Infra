import { memo, useMemo, useState } from "react";
import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import fixture from "../../content/fixtures/gqa-reference.json";
import lesson from "../../content/cases/04_gqa_source_walkthrough.md?raw";
import { compute, validateSelection, type Fault } from "./compute";
import { steps, stepFor } from "./steps";
import type { Model, State } from "../data";

const number = (value: number) =>
  value === -Infinity ? "−∞" : value.toFixed(6);
function Prose({ text }: { text: string }) {
  return (
    <div className="prose">
      <Markdown
        remarkPlugins={[remarkGfm, remarkMath]}
        rehypePlugins={[rehypeKatex]}
      >
        {text.replace(/^---\n[\s\S]*?\n---\n/, "")}
      </Markdown>
    </div>
  );
}
export const GqaWalkthrough = memo(function GqaWalkthrough({
  model,
  state,
  patch,
  onSource,
}: {
  model: Model;
  state: State;
  patch: (s: Partial<State>) => void;
  onSource: (id?: string) => void;
}) {
  const variant = model.qk_norm ? "qwen3" : "qwen25";
  const result = useMemo(() => compute(fixture, variant), [variant]);
  const [fault, setFault] = useState<Fault>("none");
  const altered = useMemo(
    () => compute(fixture, variant, fault),
    [variant, fault],
  );
  const [error, setError] = useState("");
  const step = stepFor(state.operator);
  const t = state.query,
    h = state.gqaHead,
    g = validateSelection(h, t);
  const index = steps.findIndex((s) => s.id === step.id);
  const selected: Record<string, number[][]> = {
    input: [result.input[t]],
    rms: [result.norm[t]],
    qkv: [result.q[t][h], result.k[t][g], result.v[t][g]],
    qknorm: [result.qnorm[t][h], result.knorm[t][g]],
    rope: [result.qrope[t][h], result.krope[t][g]],
    scores: [result.scores[t][h]],
    scale: [result.scaled[t][h]],
    mask: [result.masked[t][h]],
    softmax: [result.probabilities[t][h]],
    value: [result.heads[t][h]],
    merge: [result.merged[t]],
    projection: [result.projected[t]],
    residual: [result.residual[t]],
  };
  const sourceId =
    step.id === "qknorm" && !model.qk_norm ? "B-Q2" : step.sourceId;
  const kind =
    step.id === "qknorm" && !model.qk_norm ? "调用入口 · 配置参照" : step.kind;
  const delta = Math.max(
    ...result.residual.flatMap((row, j) =>
      row.map((v, i) => Math.abs(v - altered.residual[j][i])),
    ),
  );
  const worked: Record<string, string> = {
    rms: `输入均方=${number(fixture.x[t].reduce((sum, x) => sum + x * x, 0) / 8)}，加 ε 后开方=${number(Math.sqrt(fixture.x[t].reduce((sum, x) => sum + x * x, 0) / 8 + fixture.epsilon))}；分量0得到 ${number(result.norm[t][0])}。`,
    qkv: `所选Q head在mixed中的起始列=${g * 16 + (h % 2) * 4}；先做8项输入×权重乘加，${variant === "qwen25" ? "再加相应QKV bias" : "bias为0"}。本group的K起始列=${g * 16 + 8}，V起始列=${g * 16 + 12}。`,
    qknorm:
      variant === "qwen3"
        ? `Q head均方=${number(result.q[t][h].reduce((sum, x) => sum + x * x, 0) / 4)}；使用Q gain=${fixture.q_gain.join(",")}；K使用另一份gain。`
        : "本分支不调用Q/K norm，输出数组逐项等于拆分后的Q/K。",
    rope: `position=${fixture.positions[t]}，配对(0,2)角度=${fixture.positions[t]}，配对(1,3)角度=${fixture.positions[t] / 100}；所选Q的第0维由原第0和第2维共同决定。`,
    value: `输出feature0：${result.probabilities[t][h].map((p, j) => `(${number(p)} × ${number(result.v[j][g][0])})`).join(" + ")} = ${number(result.heads[t][h][0])}。`,
    projection: `输出维0：${result.merged[t].map((c, i) => `(${number(c)} × ${number(fixture.wo[i][0])})`).join(" + ")} = ${number(result.projected[t][0])}（bias=0）。`,
    residual: `分量0：原始X=${number(fixture.x[t][0])} + 投影Y=${number(result.projected[t][0])} = ${number(result.residual[t][0])}。`,
  };
  function select(field: "query" | "gqaHead", value: number) {
    try {
      validateSelection(
        field === "gqaHead" ? value : h,
        field === "query" ? value : t,
      );
      patch({ [field]: value });
      setError("");
    } catch {
      setError("教学 token/head 必须是 0–3 的整数；保留上一个合法选择。");
    }
  }
  function download() {
    const payload = {
      evidence: "reference",
      architecture_origin: "authored_scaled_gqa",
      weights_origin: "authored_fixture",
      training_execution: "not_run",
      fixture,
      variant,
      context: {
        model: state.model,
        layer: state.layer,
        query: t,
        head: h,
        substep: step.id,
      },
      result,
      serialization:
        "-Infinity is serialized as a string; all finite values retain JS number precision",
    };
    const blob = new Blob(
      [
        JSON.stringify(
          payload,
          (_, v) => (v === -Infinity ? "-Infinity" : v),
          2,
        ),
      ],
      { type: "application/json" },
    );
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "gqa-reference.json";
    a.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
  return (
    <section className="panel gqa-unit" aria-label="GQA 精讲">
      <div className="gqa-context">
        <strong>
          {model.hf_id} · Layer {state.layer} / {model.layers - 1}
        </strong>
        <p>
          tokens → embedding → {model.layers} × decoder（当前层 attention →
          FFN）→ final norm → LM head → loss
        </p>
        <label>
          当前 GQA 层{" "}
          <select
            value={state.layer}
            onChange={(e) => patch({ layer: Number(e.target.value) })}
          >
            {Array.from({ length: model.layers }, (_, i) => (
              <option key={i} value={i}>
                Layer {i}
              </option>
            ))}
          </select>
        </label>
      </div>
      <h2>GQA：从一个 query 走到 residual</h2>
      <p data-testid="gqa-real-config">
        <span className="badge">derived · 真实配置</span> H={model.hidden_size}
        ，n_q={model.heads}，n_kv={model.kv_heads}，d={model.head_dim}；Q 宽=
        {model.heads * model.head_dim!}；QKV bias{" "}
        {model.qkv_bias ? "开启" : "关闭"}，QK norm{" "}
        {model.qk_norm ? "开启" : "关闭"}。
      </p>
      <p>
        <span className="badge">reference · 教学数值</span>{" "}
        B=1，S=4，H=8，n_q=4，n_kv=2，d=4；输出投影16→8。当前 {variant}
        -style；换模型改变 bias / QK norm 分支。不同层复用同一
        fixture，不是该层实测。
      </p>
      <p className="notice" data-testid="gqa-provenance">
        architecture_origin=authored_scaled_gqa ·
        weights_origin=authored_fixture · HF / Bridge /
        RL：not_run。下列矩阵是实际计算的教学数组，不是 checkpoint 激活或 TE
        kernel trace。显示舍入至6位，导出保留完整浮点数。
      </p>
      <nav className="gqa-steps" aria-label="GQA 子步骤">
        {steps.map((s, i) => (
          <button
            key={s.id}
            aria-pressed={step.id === s.id}
            onClick={() => patch({ operator: s.id })}
          >
            {i + 1}. {s.title}
          </button>
        ))}
      </nav>
      <div className="gqa-controls">
        <label>
          教学 query token{" "}
          <input
            type="number"
            min="0"
            max="3"
            step="1"
            aria-label="教学 query token"
            value={t}
            onChange={(e) => select("query", e.target.valueAsNumber)}
          />
        </label>
        <label>
          教学 Q head{" "}
          <input
            type="number"
            min="0"
            max="3"
            step="1"
            aria-label="教学 Q head"
            value={h}
            onChange={(e) => select("gqaHead", e.target.valueAsNumber)}
          />
        </label>
        <strong data-testid="gqa-selection">
          query {t} · Q head {h} → KV group {g}
        </strong>
      </div>
      {error && <p role="alert">{error}</p>}
      <article aria-label="GQA 当前子步骤" data-substep={step.id}>
        <h3>{step.title}</h3>
        <code>{step.shape}</code>
        <Prose text={`$$\n${step.formula}\n$$`} />
        <p>{step.explanation}</p>
        <p>
          <b>配置与边界：</b>
          {step.condition}
        </p>
        <p>
          <b>验证：</b>
          {step.verify}
        </p>
        <p>
          所选 token 的输出；QKV 行分别为 Q[h]、K[group]、V[group]，norm/RoPE
          行分别为 Q、K。
        </p>
        <div className="gqa-table-wrap">
          <table data-testid="gqa-values">
            <tbody>
              {selected[step.id].map((row, i) => (
                <tr key={i}>
                  <th>row {i}</th>
                  {row.map((n, j) => (
                    <td key={j}>{number(n)}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        {worked[step.id] && <p data-testid="gqa-worked">{worked[step.id]}</p>}
        <p data-testid="gqa-dot">
          当前 query 与 key 0 的点积：
          {result.qrope[t][h]
            .map((q, i) => `(${number(q)} × ${number(result.krope[0][g][i])})`)
            .join(" + ")}{" "}
          = {number(result.scores[t][h][0])}；缩放后{" "}
          {number(result.scaled[t][h][0])}。
        </p>
        <p>
          <span className="badge" data-testid="gqa-source-kind">
            {kind}
          </span>{" "}
          · {sourceId} · 静态固定源码，未运行后端。
        </p>
        <button
          onClick={() => {
            patch({ operator: step.id });
            onSource(sourceId);
          }}
        >
          打开 GQA 子步骤源码 ↗
        </button>
        {step.id === "qknorm" && !model.qk_norm && (
          <>
            <p>
              QK norm 关闭来自当前模型档案；B-Q2 此片段直接证明 QKV bias
              配置，不包含显式
              qk_layernorm=false。下方条件实现说明模块不存在时旁路，不能把它当作已执行
              norm。
            </p>
            <button
              onClick={() => {
                patch({ operator: "qknorm" });
                onSource("C-ATTN");
              }}
            >
              查看 QK norm 条件实现 ↗
            </button>
          </>
        )}
        {step.id === "rope" && (
          <button
            onClick={() => {
              patch({ operator: "rope" });
              onSource("C-ATTN");
            }}
          >
            查看 RoPE 调用边界 ↗
          </button>
        )}
      </article>
      <div className="gqa-controls">
        <button
          disabled={index === 0}
          onClick={() => patch({ operator: steps[index - 1].id })}
        >
          ← GQA 上一步
        </button>
        <button
          disabled={index === steps.length - 1}
          onClick={() => patch({ operator: steps[index + 1].id })}
        >
          GQA 下一步 →
        </button>
      </div>
      <h3>选中 head 的完整概率矩阵</h3>
      <p>行=query，列=key；当前行为 query {t}，未来位置严格为0。</p>
      <div className="gqa-table-wrap">
        <table data-testid="gqa-probabilities">
          <thead>
            <tr>
              <th>query / key</th>
              {[0, 1, 2, 3].map((j) => (
                <th key={j}>{j}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {result.probabilities.map((row, j) => (
              <tr key={j} aria-selected={j === t}>
                <th>{j}</th>
                {row[h].map((p, k) => (
                  <td key={k} data-masked={k > j}>
                    {number(p)}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <h3>可运行的反例</h3>
      <label>
        反例{" "}
        <select
          aria-label="GQA 反例"
          value={fault}
          onChange={(e) => setFault(e.target.value as Fault)}
        >
          <option value="none">正确路径</option>
          <option value="omit_scale">故意漏缩放</option>
          <option value="wrong_group">故意选错 KV group</option>
        </select>
      </label>
      <p data-testid="gqa-counterexample">
        与正确 residual 的最大绝对差：{delta.toPrecision(12)}
        。反例使用独立标注的错误路径，不替换上方正确演算。
      </p>
      <div className="gqa-table-wrap">
        <table data-testid="gqa-counterexample-values">
          <tbody>
            <tr>
              <th>反例概率</th>
              {altered.probabilities[t][h].map((v, i) => (
                <td key={i}>{number(v)}</td>
              ))}
            </tr>
            <tr>
              <th>反例 residual</th>
              {altered.residual[t].map((v, i) => (
                <td key={i}>{number(v)}</td>
              ))}
            </tr>
          </tbody>
        </table>
      </div>
      <details>
        <summary>完整 fixture、权重与中间值（reference）</summary>
        <pre className="gqa-json">
          {JSON.stringify(
            { fixture, result },
            (_, v) => (v === -Infinity ? "-Infinity" : v),
            2,
          )}
        </pre>
      </details>
      <button onClick={download}>导出 reference 演算</button>
      <details className="gqa-course">
        <summary>完整 GQA 精讲与独立数学符号表</summary>
        <Prose text={lesson} />
      </details>
    </section>
  );
});
