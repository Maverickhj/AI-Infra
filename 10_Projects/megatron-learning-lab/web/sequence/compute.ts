import { compute as gqa } from "../gqa/compute.ts";
import {
  tokenize,
  align,
  batch as makeBatch,
  type Fixture,
} from "../sft/compute.ts";
import type { DecoderFixture } from "../decoder/compute.ts";
export type Task = {
  stage: number;
  kind: "F" | "B";
  microbatch: number;
  phase: string;
  start: number;
  end: number;
  layers: number[];
};
export function pipeline(pp = 2, microbatches = 4) {
  if (
    ![1, 2].includes(pp) ||
    !Number.isInteger(pp) ||
    !Number.isInteger(microbatches) ||
    microbatches < 1 ||
    microbatches > 8
  )
    throw Error("两层模型仅支持 PP1/2、microbatch 1–8");
  const pending = Array.from({ length: pp }, (_, s) => {
    const warm = Math.min(pp - s - 1, microbatches);
    const list: { kind: "F" | "B"; microbatch: number; phase: string }[] = [];
    for (let m = 0; m < warm; m++)
      list.push({ kind: "F", microbatch: m, phase: "warmup" });
    for (let i = 0; i < microbatches - warm; i++) {
      list.push({ kind: "F", microbatch: warm + i, phase: "steady" });
      list.push({ kind: "B", microbatch: i, phase: "steady" });
    }
    for (let m = microbatches - warm; m < microbatches; m++)
      list.push({ kind: "B", microbatch: m, phase: "cooldown" });
    return list;
  });
  const events: Task[] = [],
    available = Array(pp).fill(0);
  const ends = new Map<string, number>();
  const key = (s: number, k: string, m: number) => [s, k, m].join(":");
  let time = 0;
  while (pending.some((p) => p.length)) {
    let started = false;
    for (let s = 0; s < pp; s++) {
      const task = pending[s][0];
      if (!task || available[s] > time) continue;
      const deps: string[] = [];
      if (task.kind === "F" && s > 0)
        deps.push(key(s - 1, "F", task.microbatch));
      if (task.kind === "B") {
        deps.push(key(s, "F", task.microbatch));
        if (s < pp - 1) deps.push(key(s + 1, "B", task.microbatch));
      }
      if (deps.some((d) => !ends.has(d) || ends.get(d)! > time)) continue;
      const event = {
        ...task,
        stage: s,
        start: time,
        end: time + 2 / pp,
        layers: Array.from({ length: 2 / pp }, (_, i) => (s * 2) / pp + i),
      };
      pending[s].shift();
      events.push(event);
      ends.set(key(s, task.kind, task.microbatch), event.end);
      available[s] = event.end;
      started = true;
    }
    time++;
    if (time > 200) throw Error("调度依赖死锁");
  }
  const duration = Math.max(...events.map((e) => e.end));
  const lifetimes = Array.from({ length: pp }, (_, s) =>
    Array.from({ length: microbatches }, (_, m) => ({
      stage: s,
      microbatch: m,
      start: events.find(
        (e) => e.stage === s && e.kind === "F" && e.microbatch === m,
      )!.start,
      end: ends.get(key(s, "B", m))!,
    })),
  ).flat();
  const peak_activations = Array.from({ length: pp }, (_, s) =>
    Math.max(
      ...Array.from(
        { length: duration },
        (_, t) =>
          lifetimes.filter((l) => l.stage === s && l.start <= t && t < l.end)
            .length,
      ),
    ),
  );
  const idle_slots =
    pp * duration - events.reduce((a, e) => a + e.end - e.start, 0);
  return {
    events,
    duration,
    lifetimes,
    peak_activations,
    idle_slots,
    bubble: idle_slots / (pp * duration),
  };
}

export function partition(boundaries: number[], cp: number) {
  if (!Number.isInteger(cp) || ![1, 2].includes(cp))
    throw Error("CP 仅支持1/2");
  if (
    boundaries.length < 2 ||
    boundaries[0] !== 0 ||
    boundaries.some(
      (n, i) => !Number.isInteger(n) || (i > 0 && n <= boundaries[i - 1]),
    )
  )
    throw Error("非法物理 cu_seqlens");
  if (cp === 1)
    return [Array.from({ length: boundaries.at(-1)! }, (_, i) => i)];
  const ranks: number[][] = Array.from({ length: cp }, () => []);
  for (let doc = 0; doc < boundaries.length - 1; doc++) {
    const lo = boundaries[doc],
      width = (boundaries[doc + 1] - lo) / (2 * cp);
    if (!Number.isInteger(width))
      throw Error("每条物理序列长度须整除 2×CP；请选择 padding=8");
    for (let r = 0; r < cp; r++)
      for (const chunk of [r, 2 * cp - r - 1])
        for (let i = 0; i < width; i++) ranks[r].push(lo + chunk * width + i);
  }
  return ranks;
}
export function mergeAttention(scores: number[][], values: number[][][]) {
  if (scores.length !== values.length || !scores.some((s) => s.length))
    throw Error("all-masked query");
  const partials = scores.map((row, r) => {
    if (row.length !== values[r].length) throw Error("KV长度不匹配");
    if (!row.length)
      return { max: null as number | null, sum: 0, numerator: [0, 0, 0, 0] };
    const max = Math.max(...row),
      exp = row.map((x) => Math.exp(x - max)),
      sum = exp.reduce((a, b) => a + b, 0);
    return {
      max,
      sum,
      numerator: [0, 1, 2, 3].map((d) =>
        exp.reduce((a, e, j) => a + e * values[r][j][d], 0),
      ),
    };
  });
  const max = Math.max(
    ...partials.map((p) => (p.max === null ? -Infinity : p.max)),
  );
  const factors = partials.map((p) =>
    p.max === null ? 0 : Math.exp(p.max - max),
  );
  const sum = partials.reduce((a, p, r) => a + factors[r] * p.sum, 0);
  return {
    output: [0, 1, 2, 3].map(
      (d) =>
        partials.reduce((a, p, r) => a + factors[r] * p.numerator[d], 0) / sum,
    ),
    partials,
    max,
    sum,
  };
}
export function cpCompute(
  model: DecoderFixture,
  data: Fixture,
  layout: "ordinary" | "thd" = "ordinary",
  padding = 1,
  cp = 2,
  fault: "none" | "local_kv" | "leak" = "none",
) {
  const indices = layout === "ordinary" ? [0] : [1, 2];
  const b = makeBatch(
    indices.map((i) =>
      align(tokenize(data, data.samples[i], "assistant"), data.samples[i].id),
    ),
    padding,
  );
  const shards = partition(
    layout === "thd" ? b.cuSeqlensPadded : [0, b.rows.length],
    cp,
  );
  const matrix = (k: string) => model.parameters[k] as number[][];
  const gain = (k: string) => model.parameters[k] as number[];
  const trans = (w: number[][]) => w[0].map((_, j) => w.map((row) => row[j]));
  const qkv = gqa({
    dimensions: { B: 1, S: b.rows.length, H: 8, nq: 4, nkv: 2, d: 4 },
    epsilon: model.epsilon,
    theta: model.theta,
    x: b.rows.map((r) => matrix("embedding")[r.input.id]),
    positions: b.rows.map((r) => r.position),
    rotary_layout: "split_half",
    wqkv: trans(matrix("l0_qkv")),
    wo: trans(matrix("l0_out")),
    input_gain: gain("l0_input_gain"),
    q_gain: gain("l0_q_gain"),
    k_gain: gain("l0_k_gain"),
    qkv_bias: Array(32).fill(0),
    output_bias: Array(8).fill(0),
  });
  const keys: number[][][] = [],
    reductions: ReturnType<typeof mergeAttention>[][] = [];
  const outputs = b.rows.map((row, t) => {
    const owner = shards.findIndex((ids) => ids.includes(t));
    const visible = shards.map((ids, r) =>
      ids.filter(
        (j) =>
          row.valid &&
          b.rows[j].valid &&
          j <= t &&
          (fault === "leak" || b.rows[j].sample === row.sample) &&
          (fault !== "local_kv" || r === owner),
      ),
    );
    keys.push(visible);
    if (!row.valid) {
      reductions.push([]);
      return Array.from({ length: 4 }, () => [0, 0, 0, 0]);
    }
    const heads = [0, 1, 2, 3].map((h) => {
      const scores = visible.map((ids) =>
        ids.map(
          (j) =>
            qkv.qrope[t][h].reduce(
              (a, q, d) => a + q * qkv.krope[j][Math.floor(h / 2)][d],
              0,
            ) / 2,
        ),
      );
      const values = visible.map((ids) =>
        ids.map((j) => qkv.v[j][Math.floor(h / 2)]),
      );
      return mergeAttention(scores, values);
    });
    reductions.push(heads);
    return heads.map((h) => h.output);
  });
  return {
    outputs,
    shards,
    positions: b.rows.map((r) => r.position),
    valid: b.rows.map((r) => r.valid),
    segments: b.rows.map((r) => r.sample),
    cuSeqlens: b.cuSeqlens,
    cuSeqlensPadded: b.cuSeqlensPadded,
    batch: b,
    keys,
    reductions,
  };
}

export function spCompute(
  model: DecoderFixture,
  input: number[][],
  layer = 0,
  tp = 2,
  omitReduce = false,
) {
  if (
    !Number.isInteger(tp) ||
    ![1, 2].includes(tp) ||
    input.length % tp !== 0 ||
    ![0, 1].includes(layer)
  )
    throw Error("SP 要求 TP1/2、S 可分及真实层0/1");
  const weights = model.parameters,
    chunk = input.length / tp,
    channels = 12 / tp;
  const gain = weights["l" + layer + "_ffn_gain"] as number[],
    w = weights["l" + layer + "_gate_up"] as number[][],
    down = weights["l" + layer + "_down"] as number[][];
  const localInputs = Array.from({ length: tp }, (_, r) =>
    input.slice(r * chunk, (r + 1) * chunk),
  );
  const localNorms = localInputs.map((rows) =>
    rows.map((x) => {
      const rms = Math.sqrt(
        x.reduce((a, v) => a + v * v, 0) / 8 + model.epsilon,
      );
      return x.map((v, i) => (v / rms) * gain[i]);
    }),
  );
  const gathered = localNorms.flat();
  const products = Array.from({ length: tp }, (_, r) =>
    gathered.map((x) =>
      Array.from({ length: channels }, (_, i) => {
        const j = r * channels + i,
          g = x.reduce((a, v, k) => a + v * w[j][k], 0),
          u = x.reduce((a, v, k) => a + v * w[12 + j][k], 0);
        return (g / (1 + Math.exp(-g))) * u;
      }),
    ),
  );
  const partials = products.map((rows, r) =>
    rows.map((x) =>
      down.map((wrow) =>
        x.reduce((a, v, i) => a + v * wrow[r * channels + i], 0),
      ),
    ),
  );
  const scattered = Array.from({ length: tp }, (_, r) =>
    Array.from({ length: chunk }, (_, i) =>
      Array.from({ length: 8 }, (_, d) =>
        omitReduce
          ? partials[r][r * chunk + i][d]
          : partials.reduce((a, p) => a + p[r * chunk + i][d], 0),
      ),
    ),
  );
  return {
    localInputs,
    localNorms,
    gathered,
    products,
    partials,
    scattered,
    output: scattered.flat(),
  };
}
