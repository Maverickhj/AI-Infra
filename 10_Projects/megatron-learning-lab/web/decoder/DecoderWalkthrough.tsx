import { CourseDetails } from "../CourseText";
import { memo, useMemo } from "react";
import model from "../../content/fixtures/decoder-reference.json";
import evidence from "../../content/generated/decoder-cpu.json";
import data from "../../content/fixtures/sft-data.json";
import course from "../../content/cases/06_decoder_update.md?raw";
import type { State } from "../data";
import { tokenize, align } from "../sft/compute";
import { computeDecoder } from "./compute";
const ops = [
  ["embedding", "Embedding", "embedding"],
  ["input_norm", "输入 RMSNorm", "decoder"],
  ["attention", "GQA + residual", "decoder"],
  ["ffn_norm", "FFN RMSNorm", "decoder"],
  ["gate", "Gate", "decoder"],
  ["up", "Up", "decoder"],
  ["silu", "SiLU", "decoder"],
  ["product", "Gate × Up", "decoder"],
  ["down", "Down", "decoder"],
  ["residual", "FFN residual", "decoder"],
  ["final_norm", "Final RMSNorm", "final_norm"],
  ["lm_head", "词表 head", "lm_head"],
  ["loss", "Masked CE", "loss"],
  ["backward", "CPU backward", "backward"],
  ["update", "Update / resume", "update"],
];
const fixed = (v: number) => v.toFixed(8);
export const DecoderWalkthrough = memo(function DecoderWalkthrough({
  state,
  patch,
  onSource,
}: {
  state: State;
  patch: (s: Partial<State>) => void;
  onSource: (id?: string) => void;
}) {
  const tokens = useMemo(
    () => tokenize(data, data.samples[state.sample], state.mode),
    [state.sample, state.mode],
  );
  const rows = useMemo(
    () => align(tokens, data.samples[state.sample].id),
    [tokens, state.sample],
  );
  const ids = rows.map((r) => r.input.id),
    labels = rows.map((r) => Math.max(0, r.target.id)),
    mask = rows.map((r) => r.mask);
  const tied = state.decoderTied === "tied";
  const trace = useMemo(
    () => computeDecoder(model, ids, labels, mask, tied),
    [state.sample, state.mode, tied],
  );
  const t = Math.min(state.decoderToken, rows.length - 1),
    l = state.tinyLayer,
    layer = trace.layers[l];
  const op = state.step === "decoder" ? state.decoderOp : state.step;
  const before = l === 0 ? trace.embedding[t] : trace.layers[l - 1].residual[t];
  const views: Record<
    string,
    {
      input: number[];
      output: number[];
      shape: string;
      explain: string;
      source: string;
    }
  > = {
    embedding: {
      input: [ids[t]],
      output: trace.embedding[t],
      shape: "[S] → [S,8]",
      explain: "从 authored embedding [V,H] 的选定行读取。没有加载预训练权重。",
      source: "C-GPT",
    },
    input_norm: {
      input: before,
      output: layer.input_norm[t],
      shape: "[S,8] → [S,8]",
      explain: "输入 RMSNorm 接 GQA；l0_input_gain 在 CPU 参考中冻结。",
      source: "C-LAYER",
    },
    attention: {
      input: before,
      output: layer.attention[t],
      shape: "[S,8] → GQA [4 heads × 4] → [S,8]",
      explain:
        "调用 G01 同一 GQA 运算图：nq=4,nkv=2,d=4，再作 attention residual。",
      source: "C-ATTN",
    },
    ffn_norm: {
      input: layer.attention[t],
      output: layer.ffn_norm[t],
      shape: "[S,8] → [S,8]",
      explain:
        "FFN 前独立的 RMSNorm；真实 TE spec 可将它融合进 fc1。这里显式展示语义输入输出。",
      source: "C-LAYER",
    },
    gate: {
      input: layer.ffn_norm[t],
      output: layer.gate[t],
      shape: "[S,8] → [S,12]",
      explain:
        "fc1 的前半为 gate，后半为 up；存储 [24,8]，逻辑右乘其转置 [8,24]。",
      source: "C-MLP",
    },
    up: {
      input: layer.ffn_norm[t],
      output: layer.up[t],
      shape: "[S,8] → [S,12]",
      explain: "up 不经过 SiLU；它与激活后的 gate 逐元素相乘。",
      source: "C-MLP",
    },
    silu: {
      input: layer.gate[t],
      output: layer.silu[t],
      shape: "[S,12] → [S,12]",
      explain: "SiLU(g)=g/(1+exp(-g))，不是对 up 激活，也不是只做 sigmoid。",
      source: "C-MLP",
    },
    product: {
      input: layer.silu[t],
      output: layer.product[t],
      shape: "[S,12] × [S,12] → [S,12]",
      explain:
        "相同位置逐元素相乘；另一输入 up[0]=" +
        fixed(layer.up[t][0]) +
        "，首项积=" +
        fixed(layer.silu[t][0] * layer.up[t][0]) +
        "。",
      source: "C-MLP",
    },
    down: {
      input: layer.product[t],
      output: layer.down[t],
      shape: "[S,12] → [S,8]",
      explain: "Down 投影存储 [8,12]，输出重新进入 H=8 residual stream。",
      source: "C-MLP",
    },
    residual: {
      input: layer.attention[t],
      output: layer.residual[t],
      shape: "[S,8] + [S,8] → [S,8]",
      explain:
        "FFN residual 首项=" +
        fixed(layer.attention[t][0]) +
        " + " +
        fixed(layer.down[t][0]) +
        "；反向在此分流。",
      source: "C-LAYER",
    },
    final_norm: {
      input: trace.layers[1].residual[t],
      output: trace.final_norm[t],
      shape: "[S,8] → [S,8]",
      explain:
        "两层结束后的独立 final norm，不是最后一层的 pre-FFN norm。Core 的 stage/PP/MTP 条件决定实际模块归属。",
      source: "C-BLOCK",
    },
    lm_head: {
      input: trace.final_norm[t],
      output: trace.logits[t],
      shape: "[S,8] × [8,27] → [S,27]",
      explain: tied
        ? "head 与 embedding 使用同一参数；反向梯度必须累加两条路径。"
        : "head 是独立参数，初始复制 embedding 数值，所以 forward 相同，梯度归属不同。",
      source: "C-GPT",
    },
    loss: {
      input: [trace.logprobs[t][labels[t]]],
      output: [trace.token_loss[t]],
      shape: "[S,27] → [S] → scalar",
      explain: "CE=-目标 logprob；mask=0 仅删除直接监督。总分母是有效目标数。",
      source: "B-LOSS",
    },
  };
  const view = views[op] || views.ffn_norm;
  const sum = trace.token_loss.reduce((a, v, i) => a + v * mask[i], 0),
    count = mask.reduce((a, b) => a + b, 0);
  const normDen = Math.sqrt(
    view.input.reduce((a, v) => a + v * v, 0) / view.input.length +
      model.epsilon,
  );
  function exportSlice() {
    const payload = {
      provenance: "reference",
      model_origin: model.provenance,
      weights_origin: model.weights_origin,
      sample: data.samples[state.sample].id,
      mode: state.mode,
      tied,
      layer: l,
      token: t,
      operator: op,
      slice: views[op] ? { input: view.input, output: view.output } : null,
      loss: trace.loss,
      cpu_evidence: evidence,
    };
    const url = URL.createObjectURL(
      new Blob([JSON.stringify(payload, null, 2)], {
        type: "application/json",
      }),
    );
    const a = document.createElement("a");
    a.href = url;
    a.download = "decoder-reference-slice.json";
    a.click();
    URL.revokeObjectURL(url);
  }
  return (
    <section className="panel decoder-reference" aria-label="完整微型 decoder">
      <div className="section-header">
        <div>
          <span className="eyebrow">MESSAGES → TWO LAYERS → LOSS → UPDATE</span>
          <h2>完整微型 decoder 演算</h2>
        </div>
        <span className="badge">architecture-scaled · authored weights</span>
      </div>
      <p>
        接续 {data.samples[state.sample].id} 的完整 {ids.length} 个人工
        token，监督模式 {state.mode}。模型 V=27、H=8、两层、FFN=12；Qwen3-style
        GQA/QK norm。上方真实模型配置是 derived，此处尺寸与权重是
        reference，不能当作该 checkpoint 的执行结果。
      </p>
      <div className="sft-controls">
        <label>
          微型层
          <select
            value={l}
            onChange={(e) => patch({ tinyLayer: Number(e.target.value) })}
          >
            <option value={0}>Layer 0</option>
            <option value={1}>Layer 1</option>
          </select>
        </label>
        <label>
          Decoder token
          <input
            type="number"
            min={0}
            max={rows.length - 1}
            value={t}
            onChange={(e) => patch({ decoderToken: Number(e.target.value) })}
          />
        </label>
        <label>
          词表权重
          <select
            value={state.decoderTied}
            onChange={(e) =>
              patch({ decoderTied: e.target.value as State["decoderTied"] })
            }
          >
            <option value="tied">tied embedding/head</option>
            <option value="untied">untied 同初始数值</option>
          </select>
        </label>
      </div>
      <nav className="decoder-ops" aria-label="微型模型子步骤">
        {ops.map(([id, title, step]) => (
          <button
            key={id}
            aria-pressed={op === id}
            onClick={() =>
              patch({
                step,
                decoderOp: step === "decoder" ? id : state.decoderOp,
                operator: "overview",
              })
            }
          >
            {title}
          </button>
        ))}
      </nav>
      {op !== "backward" && op !== "update" ? (
        <>
          <h3 data-testid="decoder-op">
            {ops.find(([id]) => id === op)?.[1] || "FFN RMSNorm"} · tiny layer{" "}
            {l} · token {t}
          </h3>
          <p data-testid="decoder-shape">{view.shape}</p>
          <p>{view.explain}</p>
          {(op === "input_norm" ||
            op === "ffn_norm" ||
            op === "final_norm") && (
            <p data-testid="decoder-norm">
              sqrt(mean(x²)+ε) = {fixed(normDen)}；ε = {model.epsilon}
              ；输出第0维 = {fixed(view.output[0])}。
            </p>
          )}
          <div className="sft-table-scroll" tabIndex={0}>
            <table data-testid="decoder-values">
              <thead>
                <tr>
                  <th>切片</th>
                  <th>选定 token 的前8维（完整维数见 shape）</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td>input</td>
                  <td>{view.input.slice(0, 8).map(fixed).join(", ")}</td>
                </tr>
                <tr>
                  <td>output</td>
                  <td>{view.output.slice(0, 8).map(fixed).join(", ")}</td>
                </tr>
              </tbody>
            </table>
          </div>
          <p data-testid="decoder-loss">
            input={rows[t].input.text} → target={rows[t].target.text}；mask=
            {mask[t]}；CE={fixed(trace.token_loss[t])}；全局 sum/count=
            {fixed(sum)}/{count}，loss={fixed(trace.loss)}。
          </p>
          <button onClick={() => onSource(view.source)}>
            查看此子步骤源码 {view.source} ↗
          </button>
        </>
      ) : (
        <div className="decoder-cpu" data-testid="decoder-cpu">
          <h3>
            {op === "backward"
              ? "真实 CPU autograd · 教学权重"
              : "一次 CPU 更新与保存恢复"}
          </h3>
          <p>
            以下是本次生成的固定参考：{evidence.sample} / {evidence.mode} /
            tied，float64 CPU。它不随上方选项改写为另一场景；采集时间{" "}
            {evidence.created_at}；对应源码 SHA256 可在导出中查看。不是
            observed_bridge。
          </p>
          <p>
            初始 loss {fixed(evidence.loss_before)} → SGD 更新后{" "}
            {fixed(evidence.loss_after)}；冻结参数不变：
            {String(evidence.frozen_unchanged)}。
          </p>
          <div className="sft-table-scroll" tabIndex={0}>
            <table>
              <thead>
                <tr>
                  <th>参数 / 下标</th>
                  <th>autograd</th>
                  <th>有限差分 h=1e-5</th>
                  <th>更新前 → 后</th>
                </tr>
              </thead>
              <tbody>
                {evidence.selected_parameters.map((p) => (
                  <tr key={p.name}>
                    <td>
                      {p.name}[{p.index.join(",")}]
                    </td>
                    <td>{p.gradient.toExponential(6)}</td>
                    <td>{p.finite_difference.toExponential(6)}</td>
                    <td>
                      {fixed(p.value)} → {fixed(p.after)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p data-testid="prompt-gradient">
            prompt token 0 直接 loss mask={evidence.prompt_direct_loss_mask}；其
            activation 梯度前4维：
            {evidence.prompt_activation_gradient
              .slice(0, 4)
              .map((v) => v.toExponential(5))
              .join(", ")}
            。后续目标仍依赖此上下文。
          </p>
          <p data-testid="decoder-resume">
            连续下一步 loss={fixed(evidence.resume.continuous_next_loss)}
            ；恢复下一步 loss={fixed(evidence.resume.resumed_next_loss)}
            ；最大参数误差={evidence.resume.max_parameter_error}。保存
            model、optimizer momentum、CPU RNG；resume 检查启用 dropout=
            {evidence.resume.dropout}。
          </p>
          <p>
            手算：首个参数 θ′=θ−0.03g ={" "}
            {fixed(evidence.selected_parameters[0].value)} − 0.03 ×{" "}
            {evidence.selected_parameters[0].gradient.toExponential(6)} ={" "}
            {fixed(evidence.selected_parameters[0].after)}
            。只证明当前样本的一次更新，不证明能力提升。
          </p>
        </div>
      )}
      <button onClick={exportSlice}>导出 decoder 切片与 CPU 证据</button>
      <CourseDetails
        className="decoder-course"
        summary="精讲：norm、SwiGLU、共享权重与更新 · 符号表"
        text={course}
      />
    </section>
  );
});
