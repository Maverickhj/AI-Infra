import { memo, useMemo } from "react";
import { CourseDetails } from "../CourseText";
import type { State } from "../data";
import { models } from "../data";
import base from "../../content/fixtures/decoder-reference.json";
import fixture from "../../content/fixtures/moe-mla-reference.json";
import samples from "../../content/fixtures/sft-data.json";
import course from "../../content/cases/09_moe_mla.md?raw";
import { tokenize, align } from "../sft/compute";
import {
  computeFamily,
  expertGroups,
  familyData,
  type Family,
  type Fault,
} from "./compute";

const families: Record<string, Family> = {
  "qwen3-30ba3b": "qwen3-moe",
  "deepseek-v2-lite": "deepseek-v2-lite",
  "deepseek-v3": "deepseek-v3",
};
const ops = [
  ["router", "Router / top-k"],
  ["dispatch", "Dispatch / experts"],
  ["combine", "Routed + shared"],
  ["aux", "Auxiliary loss"],
  ["bias", "Bias 更新"],
  ["mla-norm", "MLA Q/KV norm"],
  ["mla-expanded", "MLA 展开"],
  ["mla-cache", "MLA latent/cache"],
] as const;
const fmt = (n: number) => n.toFixed(8);
const vector = (a: number[]) => a.map(fmt).join(", ");
const sourceMap: Record<string, [string, string]> = {
  router: ["C-MOEUTIL", "moe-route"],
  dispatch: ["C-DISPATCH", "moe-permute"],
  combine: ["C-DISPATCH", "moe-combine"],
  aux: ["C-ROUTER", "moe-seq-aux"],
  bias: ["C-FINALGRAD", "moe-bias-global-batch"],
  "mla-norm": ["C-MLA", "mla-latent-norm"],
  "mla-expanded": ["C-MLA", "c-mla-l638"],
  "mla-cache": ["C-MLA", "mla-cached-latent"],
};
export const FamilyWalkthrough = memo(function FamilyWalkthrough({
  state,
  patch,
  onSource,
}: {
  state: State;
  patch: (p: Partial<State>) => void;
  onSource: (id: string, excerpt?: string) => void;
}) {
  const family = families[state.model],
    model = models.find((m) => m.id === state.model)!;
  const config = family ? fixture.families[family].config : null;
  const data = useMemo(() => {
    const sample =
      samples.samples[Math.min(state.sample, samples.samples.length - 1)];
    const rows = align(tokenize(samples, sample, "assistant"), sample.id);
    return {
      sample: sample.id,
      ...familyData(
        rows.map((r) => r.input.id),
        rows.map((r) => Math.max(0, r.target.id)),
        rows.map((r) => r.mask),
        state.familyPadding,
      ),
    };
  }, [state.sample, state.familyPadding]);
  const result = useMemo(() => {
    if (!family) return null;
    const args = [
      base,
      fixture,
      family,
      data.ids,
      data.labels,
      data.mask,
      data.padding,
    ] as const;
    const normal = computeFamily(...args, state.ep, state.etp);
    try {
      return {
        normal,
        trace:
          state.familyFault === "none"
            ? normal
            : computeFamily(...args, state.ep, state.etp, state.familyFault),
        error: "",
      };
    } catch (e) {
      return { normal, trace: normal, error: String(e) };
    }
  }, [family, data, state.ep, state.etp, state.familyFault]);
  if (!family || !config || !result) return null;
  const trace = result.trace,
    l = state.tinyLayer,
    t = Math.min(state.familyToken, data.ids.length - 1),
    h = state.familyHead;
  const layer = trace.layers[l],
    r = layer.moe,
    a = trace.mla[l],
    groups = expertGroups(state.ep, state.etp);
  const op = state.familyOp;
  const source = sourceMap[op];
  const scopedSource =
    op === "aux" && family === "qwen3-moe"
      ? ["C-MOEUTIL", "moe-aux-formula"]
      : source;
  const tiny = config.moe_layers.includes(l) ? "MoE" : "dense SwiGLU";
  const actualDense =
    "first_dense_layers" in model ? model.first_dense_layers : 0;
  const steps = ops.filter(
    ([id]) => config.attention === "mla" || !id.startsWith("mla"),
  );
  function download() {
    const output = {
      schema_version: 1,
      provenance: "reference",
      backend: "browser_float64_math",
      runtime_training: "not_run",
      family,
      sample: data.sample,
      supervision: "assistant",
      dimensions: fixture.dimensions,
      config,
      selection: {
        layer: l,
        token: t,
        head: h,
        expert: state.familyExpert,
        op,
        mode: state.mla,
        ep: state.ep,
        etp: state.etp,
        edp: 1,
        fault: state.familyFault,
      },
      complete_model: {
        ce: trace.loss,
        aux: trace.aux_loss,
        total: trace.total_loss,
        final_norm: trace.final_norm[t],
        logits: trace.logits[t],
      },
      moe: r
        ? {
            indices: r.indices[t],
            weights: r.weights[t],
            counts: r.counts,
            aux_counts: r.aux_counts,
            routed: r.routed[t],
            shared: r.shared[t],
            combined: r.combined[t],
            bias: r.bias,
            next_bias: r.updated_bias,
          }
        : null,
      training_activation:
        a && state.mla === "train"
          ? {
              q_nope: a.q_nope[t][h],
              q_rope: a.q_rope[t][h],
              kv_latent: a.kv_latent[t],
              expanded_k: a.k_expanded[t][h],
              expanded_v: a.v_expanded[t][h],
              probabilities_prefix: a.probabilities[h][t].slice(0, t + 1),
              context: a.context[t][h],
            }
          : null,
      decode_cache:
        a && state.mla === "decode"
          ? {
              row: a.cache[t],
              prefix_length: t + 1,
              context: a.cached_context[t][h],
              payload_bytes: (t + 1) * 5 * 8,
            }
          : null,
      limitations: fixture.assumptions,
    };
    const url = URL.createObjectURL(
      new Blob([JSON.stringify(output, null, 2)], { type: "application/json" }),
    );
    const link = document.createElement("a");
    link.href = url;
    link.download = "family-reference.json";
    link.click();
    URL.revokeObjectURL(url);
  }
  return (
    <section
      className="panel family-reference"
      aria-label="MoE MLA 完整微型模型"
    >
      <div className="section-header">
        <div>
          <span className="eyebrow">G06 · COMPLETE FAMILY REFERENCE</span>
          <h2>从完整 decoder 看 MoE / MLA</h2>
        </div>
        <span className="badge">authored · CPU/TS reference</span>
      </div>
      <p data-testid="family-architecture">
        {model.family}：全尺寸 {model.layers} 层，前 {actualDense} 层
        dense；本例 2 层，
        {config.moe_layers.length === 2
          ? "L0/L1 均为 MoE"
          : "L0 dense → L1 MoE"}
        。当前 L{l}：{tiny}。
        {family === "deepseek-v2-lite"
          ? "直接 Q，无 Q latent norm"
          : family === "deepseek-v3"
            ? "Q down 8→4 → norm → up"
            : "GQA + QK norm"}
        ；shared={config.shared_experts}。
      </p>
      <p className="notice">
        数据：{data.sample} / assistant（固定本演算）；V27、H8、untied
        head。保留 embedding → 2×attention/FFN/residual → final norm → head →
        CE。无真实模型权重、GPU 或分布式实测。
      </p>
      <div className="sft-controls">
        <label>
          微型 family 层
          <select
            aria-label="Family layer"
            value={l}
            onChange={(e) => patch({ tinyLayer: Number(e.target.value) })}
          >
            <option value={0}>L0</option>
            <option value={1}>L1</option>
          </select>
        </label>
        <label>
          Token
          <input
            aria-label="Family token"
            type="number"
            min={0}
            max={data.ids.length - 1}
            value={t}
            onChange={(e) => patch({ familyToken: Number(e.target.value) })}
          />
        </label>
        <label>
          专家
          <select
            aria-label="Family expert"
            value={state.familyExpert}
            onChange={(e) => patch({ familyExpert: Number(e.target.value) })}
          >
            {[0, 1, 2, 3].map((i) => (
              <option key={i} value={i}>
                E{i}
              </option>
            ))}
          </select>
        </label>
        <label>
          Suffix padding
          <select
            aria-label="Family padding"
            value={state.familyPadding}
            onChange={(e) => patch({ familyPadding: Number(e.target.value) })}
          >
            <option value={0}>0</option>
            <option value={2}>2</option>
          </select>
        </label>
        <label>
          EP
          <select
            aria-label="Family EP"
            value={state.ep}
            onChange={(e) => patch({ ep: Number(e.target.value) })}
          >
            <option value={1}>1</option>
            <option value={2}>2</option>
          </select>
        </label>
        <label>
          ETP
          <select
            aria-label="Family ETP"
            value={state.etp}
            onChange={(e) => patch({ etp: Number(e.target.value) })}
          >
            <option value={1}>1</option>
            <option value={2}>2</option>
          </select>
        </label>
      </div>
      <p data-testid="family-loss">
        CE={fmt(trace.loss)}；aux={fmt(trace.aux_loss)}；total=
        {fmt(trace.total_loss)}。相对正确参考 ΔCE=
        {fmt(trace.loss - result.normal.loss)}
      </p>
      <div className="sft-table-scroll">
        <table data-testid="family-backbone">
          <thead>
            <tr>
              <th>完整主链 · token {t}</th>
              <th>值 [H8]</th>
            </tr>
          </thead>
          <tbody>
            {[
              ["Embedding", trace.embedding[t]],
              ["Attention residual", layer.attention[t]],
              ["FFN norm", layer.ffn_norm[t]],
              ["FFN down", layer.down[t]],
              ["Layer residual", layer.residual[t]],
              ["Final norm", trace.final_norm[t]],
            ].map(([name, values]) => (
              <tr key={name as string}>
                <th>{name as string}</th>
                <td>{vector(values as number[])}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <nav className="family-ops" aria-label="Family 子步骤">
        {steps.map(([id, label]) => (
          <button
            key={id}
            aria-pressed={op === id}
            onClick={() =>
              patch({
                familyOp: id,
                ...(id.startsWith("mla")
                  ? { mla: id === "mla-cache" ? "decode" : "train" }
                  : {}),
              })
            }
          >
            {label}
          </button>
        ))}
      </nav>
      <div className="family-source-actions">
        <button onClick={() => onSource(scopedSource[0], scopedSource[1])}>
          查看 family 子步骤源码
        </button>
        <button
          onClick={() =>
            onSource(
              family === "qwen3-moe"
                ? "B-Q3M"
                : family === "deepseek-v2-lite"
                  ? "B-DS2"
                  : "B-DS3",
              family === "deepseek-v2-lite" ? "ds2-provider" : undefined,
            )
          }
        >
          查看 family 配置分支
        </button>
      </div>
      {!op.startsWith("mla") && !r && (
        <p className="notice" data-testid="family-dense">
          当前 L0 是实际计算的 dense SwiGLU，未运行 router。
          <button onClick={() => patch({ tinyLayer: 1 })}>进入 L1 MoE</button>
        </p>
      )}
      {r && !op.startsWith("mla") && (
        <>
          <p data-testid="family-routing-scope">
            {config.score} /{" "}
            {config.pre_softmax
              ? "pre-softmax 或 sigmoid 原分数"
              : "selected logits softmax"}
            ；top-{config.topk}，scaling={config.scaling}，groups=
            {config.groups}。valid={r.valid_tokens}，physical={data.ids.length}
            ；当前 token{" "}
            {data.padding[t] ? "是 padding，已排除" : "有效，仍参与路由"}，loss
            mask={data.mask[t]}。
          </p>
          {op === "router" && (
            <div className="sft-table-scroll">
              <table data-testid="family-router">
                <thead>
                  <tr>
                    <th>expert</th>
                    <th>logit</th>
                    <th>原始 score</th>
                    <th>bias（仅选择）</th>
                    <th>选中</th>
                    <th>组合 weight</th>
                  </tr>
                </thead>
                <tbody>
                  {[0, 1, 2, 3].map((e) => (
                    <tr key={e}>
                      <th>E{e}</th>
                      <td>{fmt(r.logits[t][e])}</td>
                      <td>{fmt(r.scores[t][e])}</td>
                      <td>{fmt(r.bias[e])}</td>
                      <td>
                        {!data.padding[t] && r.indices[t].includes(e)
                          ? "yes"
                          : "no"}
                      </td>
                      <td>{fmt(r.routing[t][e])}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
          {op === "dispatch" && (
            <>
              <p>
                按专家排列后恢复 token；展示 E{state.familyExpert}{" "}
                的派发。总派发={r.dispatch.length}。同 token
                的不同专家各计算一次。
              </p>
              <div className="sft-table-scroll">
                <table data-testid="family-dispatch">
                  <thead>
                    <tr>
                      <th>原 token</th>
                      <th>expert / EP rank</th>
                      <th>weight</th>
                      <th>expert output [前4]</th>
                      <th>weighted output [前4]</th>
                    </tr>
                  </thead>
                  <tbody>
                    {r.dispatch
                      .filter((d) => d.expert === state.familyExpert)
                      .map((d) => (
                        <tr key={d.token}>
                          <td>{d.token}</td>
                          <td>
                            E{d.expert} / {d.rank}
                          </td>
                          <td>{fmt(d.weight)}</td>
                          <td>{vector(d.output.slice(0, 4))}</td>
                          <td>{vector(d.weighted.slice(0, 4))}</td>
                        </tr>
                      ))}
                  </tbody>
                </table>
              </div>
              {!r.dispatch.some((d) => d.expert === state.familyExpert) && (
                <p>该专家本批没有 token。</p>
              )}
              <button onClick={() => onSource("C-EXPERT", "moe-experts")}>
                查看专家计算入口
              </button>
            </>
          )}
          {op === "combine" && (
            <div className="sft-table-scroll">
              <table data-testid="family-combine">
                <thead>
                  <tr>
                    <th>token {t}</th>
                    <th>值 [H8]</th>
                  </tr>
                </thead>
                <tbody>
                  {[
                    ["routed", r.routed[t]],
                    ["shared", r.shared[t]],
                    ["combined", r.combined[t]],
                  ].map(([name, values]) => (
                    <tr key={name as string}>
                      <th>{name as string}</th>
                      <td>{vector(values as number[])}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
          {op === "aux" && (
            <>
              <p data-testid="family-aux">
                {config.aux_type}，B=1，α={config.aux_coeff}，本层 aux=
                {fmt(r.aux_loss)}。辅助 counts 不使用 bias 或 group 限制；prompt
                不按 loss mask 排除。
              </p>
              <div className="sft-table-scroll">
                <table>
                  <thead>
                    <tr>
                      <th>expert</th>
                      <th>dispatch count</th>
                      <th>aux count</th>
                      <th>token {t} 的 aux score</th>
                    </tr>
                  </thead>
                  <tbody>
                    {r.counts.map((n, e) => (
                      <tr key={e}>
                        <th>E{e}</th>
                        <td>{n}</td>
                        <td>{r.aux_counts[e]}</td>
                        <td>{fmt(r.aux_scores[t][e])}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <button onClick={() => onSource("C-MOEUTIL", "moe-aux-scores")}>
                查看 padding 与辅助索引
              </button>
            </>
          )}
          {op === "bias" && (
            <div data-testid="family-bias">
              {config.bias ? (
                <>
                  <p>
                    global batch 计数后的无梯度符号更新；η={config.bias_rate}
                    。本页预览下一次值，不随点击重复更新。
                  </p>
                  <p>counts=[{r.counts.join(", ")}]</p>
                  <p>before=[{vector(r.bias)}]</p>
                  <p>next=[{vector(r.updated_bias)}]</p>
                  <button
                    onClick={() => onSource("C-MOEUTIL", "moe-bias-update")}
                  >
                    查看符号更新公式
                  </button>
                </>
              ) : (
                <p>
                  当前 family 无 expert bias；辅助 loss 与 bias 更新是不同机制。
                </p>
              )}
            </div>
          )}
        </>
      )}
      {op.startsWith("mla") && a && (
        <>
          <div className="sft-controls">
            <label>
              MLA head
              <select
                aria-label="Family MLA head"
                value={h}
                onChange={(e) => patch({ familyHead: Number(e.target.value) })}
              >
                <option value={0}>0</option>
                <option value={1}>1</option>
              </select>
            </label>
            <label>
              MLA 视图
              <select
                aria-label="Family MLA mode"
                value={state.mla}
                onChange={(e) => patch({ mla: e.target.value as State["mla"] })}
              >
                <option value="train">训练展开激活</option>
                <option value="decode">Decode latent cache</option>
              </select>
            </label>
          </div>
          <p data-testid="family-mla-error">
            展开 / 吸收误差={a.max_absorption_error.toExponential(3)}；逐 prefix
            cache 误差={a.max_cache_error.toExponential(3)}。原
            scale=1/√4，mscale=1；无 YaRN 扩展。
          </p>
          {state.mla === "train" ? (
            <div className="sft-table-scroll">
              <table data-testid="family-mla-train">
                <thead>
                  <tr>
                    <th>
                      训练张量 · t={t}, h={h}
                    </th>
                    <th>值</th>
                  </tr>
                </thead>
                <tbody>
                  {[
                    ["Q raw / direct input", a.q_raw[t]],
                    ["Q latent / direct input", a.q_latent[t]],
                    ["KV raw [rkv+rope]", a.kv_raw[t]],
                    ["KV normalized [rkv]", a.kv_latent[t]],
                    ["Q content", a.q_nope[t][h]],
                    ["Q RoPE", a.q_rope[t][h]],
                    ["K positional RoPE", a.k_rope[t]],
                    ["Expanded K content", a.k_expanded[t][h]],
                    ["Expanded V", a.v_expanded[t][h]],
                    [
                      "Attention probabilities (causal prefix)",
                      a.probabilities[h][t].slice(0, t + 1),
                    ],
                    ["Expanded context", a.context[t][h]],
                    ["Absorbed context", a.absorbed[t][h]],
                  ].map(([name, values]) => (
                    <tr key={name as string}>
                      <th>{name as string}</th>
                      <td>{vector(values as number[])}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <div data-testid="family-mla-cache">
              <p>
                Decode cache：每层每 token [normalized latent(3), rotated
                Kpos(2)]。只读取 prefix {t + 1}；payload={(t + 1) * 5 * 8}{" "}
                bytes（float64），不计 allocator/训练激活，不是 GPU 显存。
              </p>
              <div className="sft-table-scroll">
                <table>
                  <thead>
                    <tr>
                      <th>字段</th>
                      <th>值</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr>
                      <th>cache row {t}</th>
                      <td>{vector(a.cache[t])}</td>
                    </tr>
                    <tr>
                      <th>absorbed Q</th>
                      <td>{vector(a.absorbed_q[t][h])}</td>
                    </tr>
                    <tr>
                      <th>weighted latent</th>
                      <td>{vector(a.latent_context[t][h])}</td>
                    </tr>
                    <tr>
                      <th>prefix context</th>
                      <td>{vector(a.cached_context[t][h])}</td>
                    </tr>
                  </tbody>
                </table>
              </div>
              <button onClick={() => onSource("C-MLA", "mla-value-absorption")}>
                查看 decode V up
              </button>
            </div>
          )}
          {family === "deepseek-v2-lite" && (
            <button onClick={() => onSource("C-MLA", "mla-direct-q")}>
              查看 V2 直接 Q
            </button>
          )}
        </>
      )}
      <details>
        <summary>已验证的 EP / ETP / EDP 参考组</summary>
        <p>
          EDP=1；rank = ep_rank × ETP + etp_rank。仅 CPU
          内存内参考编号，不映射任意 Core world，不与 dense TP world 机械相乘。
        </p>
        <div className="sft-table-scroll">
          <table data-testid="family-groups">
            <thead>
              <tr>
                <th>rank</th>
                <th>experts</th>
                <th>EP</th>
                <th>ETP</th>
                <th>EDP</th>
                <th>FFN columns</th>
              </tr>
            </thead>
            <tbody>
              {groups.map((g) => (
                <tr key={g.rank}>
                  <td>{g.rank}</td>
                  <td>{g.experts.join(",")}</td>
                  <td>{g.ep.join(",")}</td>
                  <td>{g.etp.join(",")}</td>
                  <td>{g.edp.join(",")}</td>
                  <td>[{g.ffn_columns.join(",")})</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </details>
      <label>
        反例
        <select
          aria-label="Family fault"
          value={state.familyFault}
          onChange={(e) => patch({ familyFault: e.target.value as Fault })}
        >
          <option value="none">正确参考</option>
          <option value="wrong_weights">错误路由归一化 / bias 权重</option>
          <option value="duplicate_dispatch">重复派发同一 token/expert</option>
          {config.shared_experts > 0 && (
            <option value="missing_shared">遗漏 shared</option>
          )}
          {a && (
            <>
              <option value="wrong_norm">吸收路径遗漏 latent norm</option>
              <option value="wrong_scale">吸收路径误用 √5</option>
              <option value="missing_rope">吸收路径遗漏 RoPE</option>
            </>
          )}
        </select>
      </label>
      {result.error && (
        <p role="alert">
          {result.error}；已拒绝该派发，表中保留正确参考供对照，导出停用。
        </p>
      )}
      <p className="muted">
        独立检查：test:moe-mla（实时 Torch 与
        TS、完整模型梯度及有限差分）。图中数组为当前浏览器演算，不是
        observed_bridge。
      </p>
      <button disabled={!!result.error} onClick={download}>
        导出 family reference 切片
      </button>
      <CourseDetails
        className="family-course"
        summary="精讲：MoE / MLA 数学、源码条件与独立符号表"
        text={course}
      />
    </section>
  );
});
