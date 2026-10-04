import { CourseDetails } from "../CourseText";
import { memo, useState } from "react";
import evidenceRaw from "../../content/generated/tp-dp-cpu.json";
import samples from "../../content/fixtures/sft-data.json";
import course from "../../content/cases/07_tp_dp.md?raw";
import type { State } from "../data";
import { groups } from "./layout";
type Rank = {
  rank: number;
  weight_shape: number[];
  input_shape: number[];
  output_shape: number[];
  input_token7: number[];
  token7: (number | string)[];
  weight_preview: number[][];
  gradient_preview: number[][];
  weight_rows?: number[];
  input_columns?: number[];
  gate_rows?: number[];
  up_rows?: number[];
  vocab_range?: number[];
  valid_range?: number[];
};
type Event = {
  name: string;
  layer: number;
  collective: string;
  global_shape: number[];
  global_token7: (number | string)[];
  parameter: string;
  global_weight_shape: number[];
  global_weight_preview: number[][];
  ranks: Rank[];
  ce_reductions?: {
    global_max: number;
    global_exp_sum: number;
    target: number;
    target_logit: number;
    local_max: number[];
    local_exp_sum: number[];
  };
};
type Case = {
  tp: number;
  sample: number;
  tied: boolean;
  loss: number;
  max_gradient_error: number;
  max_output_error: number;
  rank_events: Event[];
  communication: { payload_bytes: number; ring_allreduce_sent_bytes: number };
  vocabulary: { logical: number; physical: number };
};
const evidence = evidenceRaw as unknown as {
  cases: Case[];
  dp: typeof evidenceRaw.dp;
  wrong_dp: typeof evidenceRaw.wrong_dp;
  source_hashes: Record<string, string>;
};
const ops = [
  ["qkv", "QKV 列分片"],
  ["attention_output", "Attention 输出部分和"],
  ["ffn_pair", "Gate/Up 成对分片"],
  ["ffn_output", "FFN 输出部分和"],
  ["vocab", "词表分片 CE"],
];
const shape = (v: number[]) => "[" + v.join(",") + "]";
const fmt = (v: number | string) => (typeof v === "number" ? v.toFixed(8) : v);
const vector = (v: (number | string)[]) => v.map(fmt).join(", ");
const matrix = (v: number[][]) =>
  v.map((row) => "[" + vector(row) + "]").join(" / ");
const backward: Record<string, string> = {
  qkv: "列分片的 dW 留在本 rank；每个分片给复制输入贡献 dX_r=dY_r W_r，完整 dX=Σ_r dX_r。每个 rank 的 Q/K/V 头可独立 attention，无须先 gather 头。",
  attention_output:
    "行分片先 SUM 部分输出；反向每个 rank 使用同一个 dY，得到本地 dX_r=dY W_r。拼接本地输入梯度恢复逻辑 dX，不再把 dY 额外平均。",
  ffn_pair:
    "gate/up 必须按同一 FFN 通道配对，SiLU(gate_r)×up_r 留在本地。列分片反向的 dX 贡献求和；全局 [gate;up] 不能直接把 rank-local 成对存储按 rank 串起来。",
  ffn_output:
    "down 的输入列和前一步 product_r 对齐。前向 SUM 部分和；反向 dY 复制、本地 dX_r 与 dW_r。这里没有 SP，因此不是 reduce-scatter。",
  vocab:
    "CE 反向每个 rank 得到自己的 softmax−one_hot_owner，再乘 mask/global_count；head 输入梯度跨词表片求和。tied 时所示 embedding 参数梯度已汇集查表与 head 两条路径。",
};
export const ParallelInspector = memo(function ParallelInspector({
  state,
  patch,
  onSource,
}: {
  state: State;
  patch: (s: Partial<State>) => void;
  onSource: (id?: string) => void;
}) {
  const g = groups(state.tp, state.dp, state.parallelRank);
  const c = evidence.cases.find(
    (x) =>
      x.tp === state.tp &&
      x.sample === state.sample &&
      x.tied === (state.decoderTied === "tied"),
  )!;
  const event = c.rank_events.find(
    (x) =>
      x.name === state.parallelOp &&
      (x.name === "vocab" || x.layer === state.tinyLayer),
  )!;
  const rank = event.ranks[g.tpRank];
  const [trial, setTrial] = useState("3"),
    [error, setError] = useState("");
  const [wrongDP, setWrongDP] = useState(false);
  const d = evidence.dp,
    bad = evidence.wrong_dp;
  const source = state.parallelOp === "vocab" ? "C-VOCABCE" : "C-TPLINEAR";
  function exportSlice() {
    const payload = {
      provenance: "reference_simulation",
      device: "cpu",
      distributed_backend: "not_run",
      sample: samples.samples[state.sample].id,
      mode: "assistant",
      token: 7,
      tied: c.tied,
      groups: g,
      event,
      selected_rank: rank,
      loss: c.loss,
      max_gradient_error: c.max_gradient_error,
      dp: d,
      wrong_dp: bad,
      source_hashes: evidence.source_hashes,
    };
    const url = URL.createObjectURL(
      new Blob([JSON.stringify(payload, null, 2)], {
        type: "application/json",
      }),
    );
    const a = document.createElement("a");
    a.href = url;
    a.download = "tp-dp-reference-slice.json";
    a.click();
    URL.revokeObjectURL(url);
  }
  return (
    <section
      className="panel parallel-reference"
      aria-label="TP DP rank inspector"
    >
      <div className="section-header">
        <h2>同一完整模型的 TP / DP</h2>
        <span className="badge">reference simulation · CPU</span>
      </div>
      <p>
        接续 G03 两层模型 V27/H8/F12，当前 {samples.samples[state.sample].id} /{" "}
        {state.decoderTied}。此 CPU 证据固定 assistant、token 7；上方监督模式为{" "}
        {state.mode}，不改变这份参考。没有进程组、NCCL 或通信耗时实测。TP 与 DP
        分别数值验证，group 图展示其逻辑组合。
      </p>
      <div className="sft-controls">
        <label>
          TP
          <select
            aria-label="TP"
            value={state.tp}
            onChange={(e) =>
              patch({ tp: Number(e.target.value), parallelRank: 0 })
            }
          >
            <option value={1}>1</option>
            <option value={2}>2</option>
          </select>
        </label>
        <label>
          DP
          <select
            aria-label="DP"
            value={state.dp}
            onChange={(e) =>
              patch({ dp: Number(e.target.value), parallelRank: 0 })
            }
          >
            <option value={1}>1</option>
            <option value={2}>2</option>
          </select>
        </label>
        <label>
          Global rank
          <select
            aria-label="Global rank"
            value={state.parallelRank}
            onChange={(e) => patch({ parallelRank: Number(e.target.value) })}
          >
            {Array.from({ length: g.world }, (_, r) => (
              <option key={r} value={r}>
                {r}
              </option>
            ))}
          </select>
        </label>
        <label>
          TP 层
          <select
            aria-label="TP 层"
            value={state.tinyLayer}
            onChange={(e) => patch({ tinyLayer: Number(e.target.value) })}
          >
            <option value={0}>Layer 0</option>
            <option value={1}>Layer 1</option>
          </select>
        </label>
        <label>
          TP 算子
          <select
            aria-label="TP 算子"
            value={state.parallelOp}
            onChange={(e) => patch({ parallelOp: e.target.value })}
          >
            {ops.map(([id, title]) => (
              <option key={id} value={id}>
                {title}
              </option>
            ))}
          </select>
        </label>
      </div>
      <p data-testid="parallel-groups">
        world={g.world}；global rank={state.parallelRank} → TP rank={g.tpRank}
        、DP rank={g.dpRank}；TP group={shape(g.tpMembers)}；DP group=
        {shape(g.dpMembers)}。本图 PP=CP=1；没有机械乘 EP。
      </p>
      <div className="parallel-check">
        <label>
          试算 TP
          <input
            type="number"
            value={trial}
            onChange={(e) => setTrial(e.target.value)}
          />
        </label>
        <button
          onClick={() => {
            try {
              groups(Number(trial), state.dp, 0);
              setError("");
              patch({ tp: Number(trial), parallelRank: 0 });
            } catch (e) {
              setError(String((e as Error).message));
            }
          }}
        >
          验证配置
        </button>
        {error && <p role="alert">{error}</p>}
      </div>
      <h3>{ops.find(([id]) => id === state.parallelOp)?.[1]} · token 7</h3>
      <p data-testid="parallel-shape">
        参数 {event.parameter}：全局存储 {shape(event.global_weight_shape)} →
        rank-local {shape(rank.weight_shape)}；局部输入{" "}
        {shape(rank.input_shape)} → 局部输出 {shape(rank.output_shape)}
        ；重组逻辑输出 {shape(event.global_shape)}。
      </p>
      <p data-testid="parallel-ranges">
        {rank.weight_rows && "权重行 " + shape(rank.weight_rows) + "；"}
        {rank.input_columns &&
          "输入/权重列 " + shape(rank.input_columns) + "；"}
        {rank.gate_rows &&
          "gate 行 " +
            shape(rank.gate_rows) +
            "；up 行 " +
            shape(rank.up_rows!) +
            "；"}
        {rank.vocab_range &&
          "物理词表 " +
            shape(rank.vocab_range) +
            "；有效词表 " +
            shape(rank.valid_range!) +
            "；"}
        范围均为左闭右开。
      </p>
      <div className="sft-table-scroll" tabIndex={0}>
        <table data-testid="parallel-values">
          <thead>
            <tr>
              <th>切片</th>
              <th>数值（仅一个 token / 参数前2行4列）</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <th>global weight</th>
              <td>{matrix(event.global_weight_preview)}</td>
            </tr>
            <tr>
              <th>rank weight</th>
              <td>{matrix(rank.weight_preview)}</td>
            </tr>
            <tr>
              <th>rank input</th>
              <td>{vector(rank.input_token7)}</td>
            </tr>
            <tr>
              <th>rank output / partial</th>
              <td>{vector(rank.token7)}</td>
            </tr>
            <tr>
              <th>global output</th>
              <td>{vector(event.global_token7)}</td>
            </tr>
            <tr>
              <th>rank parameter gradient</th>
              <td>{matrix(rank.gradient_preview)}</td>
            </tr>
          </tbody>
        </table>
      </div>
      <p data-testid="parallel-collective">前向：{event.collective}。</p>
      <p>{backward[state.parallelOp]}</p>
      {event.ce_reductions && (
        <div data-testid="vocab-reductions">
          <p>
            本例逻辑 V={c.vocabulary.logical}，物理 V={c.vocabulary.physical}
            。额外槽位 logit=−∞，明确排除 CE；这不是任意 Core 默认行为。无 label
            smoothing。
          </p>
          <p>
            local MAX={vector(event.ce_reductions.local_max)} → global MAX=
            {fmt(event.ce_reductions.global_max)}；local exp sum=
            {vector(event.ce_reductions.local_exp_sum)} → SUM=
            {fmt(event.ce_reductions.global_exp_sum)}；target=
            {event.ce_reductions.target}，owner logit SUM=
            {fmt(event.ce_reductions.target_logit)}。
          </p>
        </div>
      )}
      <p data-testid="parallel-error">
        完整模型 loss={fmt(c.loss)}；恢复 logits 最大误差=
        {c.max_output_error.toExponential(3)}；所有可训练参数梯度最大误差=
        {c.max_gradient_error.toExponential(3)}（CPU
        float64，atol=rtol=1e-10）。
      </p>
      <p data-testid="parallel-bytes">
        单次 [S,8] ring SUM：payload={c.communication.payload_bytes} bytes；每
        rank 发送量 2(TP−1)/TP × payload=
        {c.communication.ring_allreduce_sent_bytes} bytes。FP64=8
        bytes；未计接收、其他 collective、梯度或整个模型通信，无耗时估计。
      </p>
      <button onClick={() => onSource(source)}>
        查看 TP 算子源码 {source} ↗
      </button>
      <h3>DP：不同有效长度与两次梯度累积</h3>
      <p>
        固定 assistant/tied，同一批4个 microbatch：DP2 分配 [sample1,sample1] 与
        [sample2,sample0]；DP1 对照把这4个 microbatch 全部放在同一模型。当前 DP=
        {state.dp}，以下完整表保留 DP2 的两 rank 归约证据。global rank 的 DP
        坐标对应表中 rank。
      </p>
      <div className="sft-table-scroll" tabIndex={0}>
        <table data-testid="dp-counts">
          <thead>
            <tr>
              <th>DP rank</th>
              <th>microbatches</th>
              <th>有效 token</th>
              <th>loss sum</th>
              <th>local mean</th>
            </tr>
          </thead>
          <tbody>
            {d.ranks.map((r) => (
              <tr key={r.rank}>
                <td>{r.rank}</td>
                <td>{r.microbatches}</td>
                <td>{r.valid_tokens}</td>
                <td>{fmt(r.loss_sum)}</td>
                <td>{fmt(r.local_mean)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <label>
        DP 归约
        <select
          aria-label="DP 归约"
          value={wrongDP ? "wrong" : "correct"}
          onChange={(e) => setWrongDP(e.target.value === "wrong")}
        >
          <option value="correct">全局 sum / count</option>
          <option value="wrong">反例：均值之均值</option>
        </select>
      </label>
      <p data-testid="dp-loss">
        global_count={d.global_count}；正确全局 loss={fmt(d.global_loss)}；所选
        loss={fmt(wrongDP ? d.mean_of_means : d.global_loss)}；梯度最大误差=
        {(wrongDP ? bad : d).max_gradient_error.toExponential(3)}。
        {wrongDP
          ? "错误：每 rank 权重相同，把4个目标与18个目标等权平均。"
          : "每个 rank 累积 token loss sum / global_count，随后 SUM 梯度；不再除 DP。"}
      </p>
      <p>
        选定 {d.selected_gradient.parameter}：单模型=
        {fmt(d.selected_gradient.reference)}；当前归约=
        {fmt((wrongDP ? bad : d).selected_gradient.reduced)}。
      </p>
      <button onClick={() => onSource("C-FINALGRAD")}>
        查看 token 归一化源码 ↗
      </button>
      <button onClick={exportSlice}>导出 TP DP 切片</button>
      <CourseDetails
        className="decoder-course"
        summary="精讲：分片、collective 与全局归约 · 符号表"
        text={course}
      />
    </section>
  );
});
