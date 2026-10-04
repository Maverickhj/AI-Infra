import {
  denseAttention,
  denseFeedForward,
  runDecoder,
  type DecoderFixture,
} from "../decoder/compute.ts";
import { rms } from "../gqa/compute.ts";
type Matrix = number[][];
type Params = Record<string, number[] | Matrix>;
export type Family = "qwen3-moe" | "deepseek-v2-lite" | "deepseek-v3";
export type Config = {
  attention: string;
  q_rank: number | null;
  moe_layers: number[];
  experts: number;
  topk: number;
  expert_width: number;
  shared_experts: number;
  score: string;
  pre_softmax: boolean;
  groups: number;
  group_topk: number;
  scaling: number;
  aux_type: string;
  aux_coeff: number;
  bias_rate: number;
  bias: number[] | null;
};
export type FamilyFixture = {
  families: Record<string, { config: Config; parameters: Params }>;
};
export type Fault =
  | "none"
  | "wrong_weights"
  | "duplicate_dispatch"
  | "missing_shared"
  | "wrong_norm"
  | "wrong_scale"
  | "missing_rope";
const sum = (a: number[]) => a.reduce((s, v) => s + v, 0);
const dot = (a: number[], b: number[]) => sum(a.map((v, i) => v * b[i]));
const add = (a: number[], b: number[]) => a.map((v, i) => v + b[i]);
const zeros = (n: number, d: number) =>
  Array.from({ length: n }, () => Array(d).fill(0) as number[]);
const project = (x: Matrix, w: Matrix) =>
  x.map((row) => w.map((col) => dot(row, col)));
const softmax = (a: number[]) => {
  const m = Math.max(...a),
    e = a.map((v) => Math.exp(v - m)),
    z = sum(e);
  return e.map((v) => v / z);
};
const top = (a: number[], k: number) =>
  a
    .map((v, i) => ({ v, i }))
    .sort((a, b) => b.v - a.v || a.i - b.i)
    .slice(0, k)
    .map((v) => v.i);
const maxDiff = (a: number[], b: number[]) =>
  Math.max(...a.map((v, i) => Math.abs(v - b[i])));
export function expertGroups(ep = 1, etp = 1, edp = 1) {
  if (![1, 2].includes(ep) || ![1, 2].includes(etp) || edp !== 1)
    throw Error("verified expert groups require EP1/2, ETP1/2, EDP1");
  const range = (a: number, b: number) =>
    Array.from({ length: b - a }, (_, i) => a + i);
  return range(0, ep * etp).map((rank) => {
    const e = Math.floor(rank / etp),
      t = rank % etp;
    return {
      rank,
      ep: range(0, ep).map((i) => i * etp + t),
      etp: range(e * etp, (e + 1) * etp),
      edp: [rank],
      experts: range((e * 4) / ep, ((e + 1) * 4) / ep),
      ffn_columns: [(t * 4) / etp, ((t + 1) * 4) / etp],
    };
  });
}
export function route(
  logits: Matrix,
  c: Config,
  padding: boolean[],
  fault: Fault = "none",
) {
  if (
    !logits.length ||
    logits.some((r) => r.length !== 4 || r.some((v) => !Number.isFinite(v))) ||
    padding.length !== logits.length ||
    padding.every(Boolean)
  )
    throw Error("invalid router logits/padding or no valid tokens");
  const scores = logits.map((row) =>
    c.score === "sigmoid"
      ? row.map((v) => 1 / (1 + Math.exp(-v)))
      : softmax(row),
  );
  const bias = c.bias ?? [0, 0, 0, 0];
  const indices = logits.map((row, t) => {
    let selection = (
      c.score === "sigmoid" || c.pre_softmax ? scores[t] : row
    ).map((v, e) => v + (c.bias ? bias[e] : 0));
    if (c.groups > 1) {
      const groupScores = [0, 1].map((g) => {
        const group = selection.slice(2 * g, 2 * g + 2);
        return sum(top(group, c.topk / c.group_topk).map((i) => group[i]));
      });
      const groups = top(groupScores, c.group_topk);
      selection = selection.map((v, e) =>
        groups.includes(Math.floor(e / 2)) ? v : -Infinity,
      );
    }
    return top(selection, c.topk);
  });
  const weights = indices.map((es, t) => {
    let w = es.map((e) => scores[t][e]);
    if (c.score === "sigmoid") {
      const z = sum(w) + 1e-20;
      w = w.map((v) => v / z);
    } else if (!c.pre_softmax) w = softmax(es.map((e) => logits[t][e]));
    if (fault === "wrong_weights") {
      if (c.score === "sigmoid") w = es.map((e) => scores[t][e] + bias[e]);
      else
        w = c.pre_softmax
          ? softmax(es.map((e) => logits[t][e]))
          : es.map((e) => scores[t][e]);
    }
    return w.map((v) => (padding[t] ? 0 : v * c.scaling));
  });
  const routing = zeros(logits.length, 4),
    counts = [0, 0, 0, 0],
    aux_counts = [0, 0, 0, 0];
  indices.forEach((es, t) =>
    es.forEach((e, i) => {
      routing[t][e] = weights[t][i];
      if (!padding[t]) counts[e]++;
    }),
  );
  const aux_scores = scores.map((row, t) => {
    const z = c.score === "sigmoid" ? sum(row) + 1e-20 : 1;
    if (!padding[t]) top(row, c.topk).forEach((e) => aux_counts[e]++);
    return row.map((v) => (padding[t] ? 0 : v / z));
  });
  const valid_tokens = padding.filter((v) => !v).length;
  const aux_loss =
    (c.aux_coeff *
      4 *
      sum(aux_counts.map((n, e) => n * sum(aux_scores.map((row) => row[e]))))) /
    (c.topk * valid_tokens ** 2);
  const average = sum(counts) / 4;
  const updated_bias = bias.map((v, e) =>
    c.bias ? v + Math.sign(average - counts[e]) * c.bias_rate : v,
  );
  return {
    logits,
    scores,
    indices,
    weights,
    routing,
    counts,
    aux_counts,
    aux_scores,
    aux_loss,
    bias,
    updated_bias,
    valid_tokens,
  };
}
export type Dispatch = {
  token: number;
  expert: number;
  rank: number;
  weight: number;
  output: number[];
  weighted: number[];
};
export function validateDispatch(
  dispatch: Dispatch[],
  indices: Matrix,
  padding: boolean[],
) {
  const expected = indices
    .flatMap((row, t) => (padding[t] ? [] : row.map((e) => t + ":" + e)))
    .sort();
  const actual = dispatch.map((d) => d.token + ":" + d.expert).sort();
  if (
    new Set(actual).size !== actual.length ||
    JSON.stringify(actual) !== JSON.stringify(expected)
  )
    throw Error("duplicate or missing token/expert dispatch");
}
function swiglu(x: Matrix, gateUp: Matrix, down: Matrix, etp = 1) {
  const width = down[0].length,
    output = zeros(x.length, 8);
  for (let rank = 0; rank < etp; rank++) {
    const a = (rank * width) / etp,
      b = ((rank + 1) * width) / etp;
    x.forEach((row, t) => {
      const product = gateUp.slice(a, b).map((w, j) => {
        const g = dot(row, w),
          u = dot(row, gateUp[width + a + j]);
        return (g / (1 + Math.exp(-g))) * u;
      });
      down.forEach((w, i) => {
        output[t][i] += dot(product, w.slice(a, b));
      });
    });
  }
  return output;
}
export function computeMoE(
  x: Matrix,
  p: Params,
  c: Config,
  padding: boolean[],
  ep = 1,
  etp = 1,
  fault: Fault = "none",
) {
  expertGroups(ep, etp);
  const r = route(project(x, p.router as Matrix), c, padding, fault),
    routed = zeros(x.length, 8),
    dispatch: Dispatch[] = [];
  for (let rank = 0; rank < ep; rank++)
    for (let e = (rank * 4) / ep; e < ((rank + 1) * 4) / ep; e++) {
      const tokens = x
        .map((_, i) => i)
        .filter((t) => !padding[t] && r.indices[t].includes(e));
      const out = swiglu(
        tokens.map((t) => x[t]),
        p["e" + e + "_gate_up"] as Matrix,
        p["e" + e + "_down"] as Matrix,
        etp,
      );
      tokens.forEach((token, i) => {
        const weight = r.routing[token][e],
          weighted = out[i].map((v) => v * weight);
        routed[token] = add(routed[token], weighted);
        dispatch.push({
          token,
          expert: e,
          rank,
          weight,
          output: out[i],
          weighted,
        });
      });
    }
  if (fault === "duplicate_dispatch" && dispatch.length)
    dispatch.push(dispatch[0]);
  validateDispatch(dispatch, r.indices, padding);
  const shared = c.shared_experts
    ? swiglu(x, p.shared_gate_up as Matrix, p.shared_down as Matrix).map(
        (row, t) => (padding[t] ? row.map(() => 0) : row),
      )
    : zeros(x.length, 8);
  const down = routed.map((row, t) =>
    fault === "missing_shared" ? row.slice() : add(row, shared[t]),
  );
  return { down, moe: { ...r, dispatch, routed, shared, combined: down } };
}
const rotate = (v: number[], t: number) => [
  v[0] * Math.cos(t) - v[1] * Math.sin(t),
  v[0] * Math.sin(t) + v[1] * Math.cos(t),
];
export function computeMLA(
  x: Matrix,
  p: Params,
  qrank: number | null,
  epsilon = 1e-6,
  fault: Fault = "none",
) {
  const n = x.length;
  const q_raw = qrank
    ? project(x, p.q_down as Matrix)
    : x.map((r) => r.slice());
  const q_latent = qrank
    ? q_raw.map((row) => rms(row, p.q_latent_gain as number[], epsilon))
    : q_raw;
  const q = project(q_latent, p[qrank ? "q_up" : "q_direct"] as Matrix).map(
    (row) => [row.slice(0, 4), row.slice(4)],
  );
  const kv_raw = project(x, p.kv_down as Matrix);
  const kv_latent = kv_raw.map((row) =>
    rms(row.slice(0, 3), p.kv_gain as number[], epsilon),
  );
  const q_nope = q.map((row) => row.map((h) => h.slice(0, 2)));
  const q_rope = q.map((row, t) => row.map((h) => rotate(h.slice(2), t)));
  const k_rope = kv_raw.map((row, t) => rotate(row.slice(3), t));
  const up = p.kv_up as Matrix,
    wk = [up.slice(0, 2), up.slice(4, 6)],
    wv = [up.slice(2, 4), up.slice(6, 8)];
  const k_expanded = kv_latent.map((c) =>
    wk.map((w) => w.map((d) => dot(c, d))),
  );
  const v_expanded = kv_latent.map((c) =>
    wv.map((w) => w.map((d) => dot(c, d))),
  );
  const probabilities = [0, 1].map((h) =>
    x.map((_, t) => {
      const logits = Array.from(
        { length: t + 1 },
        (_, s) =>
          (dot(q_nope[t][h], k_expanded[s][h]) + dot(q_rope[t][h], k_rope[s])) /
          2,
      );
      return [...softmax(logits), ...Array(n - t - 1).fill(0)] as number[];
    }),
  );
  const context = x.map((_, t) =>
    [0, 1].map((h) =>
      [0, 1].map((d) =>
        sum(x.map((_, s) => probabilities[h][t][s] * v_expanded[s][h][d])),
      ),
    ),
  );
  const absorbed_q = q_nope.map((row) =>
    row.map((q, h) =>
      [0, 1, 2].map((r) => sum(q.map((v, d) => v * wk[h][d][r]))),
    ),
  );
  const used =
    fault === "wrong_norm" ? kv_raw.map((row) => row.slice(0, 3)) : kv_latent;
  const lp = [0, 1].map((h) =>
    x.map((_, t) => {
      const scores = Array.from({ length: t + 1 }, (_, s) => {
        const pos =
          fault === "missing_rope"
            ? dot(q[t][h].slice(2), kv_raw[s].slice(3))
            : dot(q_rope[t][h], k_rope[s]);
        return (
          (dot(absorbed_q[t][h], used[s]) + pos) /
          (fault === "wrong_scale" ? Math.sqrt(5) : 2)
        );
      });
      return [...softmax(scores), ...Array(n - t - 1).fill(0)] as number[];
    }),
  );
  const latent_context = x.map((_, t) =>
    [0, 1].map((h) =>
      [0, 1, 2].map((r) => sum(x.map((_, s) => lp[h][t][s] * used[s][r]))),
    ),
  );
  const absorbed = latent_context.map((row) =>
    row.map((c, h) => wv[h].map((w) => dot(c, w))),
  );
  const projected = project(
    context.map((row) => row.flat()),
    p.mla_out as Matrix,
  );
  const cache: Matrix = [],
    cached_context: number[][][] = [];
  for (let t = 0; t < n; t++) {
    cache.push([...used[t], ...k_rope[t]]);
    cached_context.push(
      [0, 1].map((h) => {
        const logits = cache.map(
          (c) =>
            (dot(absorbed_q[t][h], c.slice(0, 3)) +
              dot(q_rope[t][h], c.slice(3))) /
            (fault === "wrong_scale" ? Math.sqrt(5) : 2),
        );
        const probs = softmax(logits),
          c = [0, 1, 2].map((r) => sum(cache.map((v, s) => probs[s] * v[r])));
        return wv[h].map((w) => dot(c, w));
      }),
    );
  }
  return {
    q_raw,
    q_latent,
    kv_raw,
    kv_latent,
    q_nope,
    q_rope,
    k_rope,
    k_expanded,
    v_expanded,
    probabilities,
    context,
    absorbed_q,
    latent_context,
    absorbed,
    cache,
    cached_context,
    projected,
    max_absorption_error: maxDiff(context.flat(2), absorbed.flat(2)),
    max_cache_error: maxDiff(context.flat(2), cached_context.flat(2)),
  };
}
export function familyData(
  ids: number[],
  labels: number[],
  mask: number[],
  padding = 0,
) {
  if (!Number.isInteger(padding) || padding < 0 || padding > 4)
    throw Error("padding must be 0..4");
  return {
    ids: [...ids, ...Array(padding).fill(0)],
    labels: [...labels, ...Array(padding).fill(0)],
    mask: [...mask, ...Array(padding).fill(0)],
    padding: ids.map(() => false).concat(Array(padding).fill(true)),
  };
}
type FamilyFFN = {
  down: Matrix;
  moe?: ReturnType<typeof computeMoE>["moe"];
  gate?: Matrix;
  up?: Matrix;
  silu?: Matrix;
  product?: Matrix;
};
export function computeFamily(
  base: DecoderFixture,
  fixture: FamilyFixture,
  family: Family,
  ids: number[],
  labels: number[],
  mask: number[],
  padding: boolean[],
  ep = 1,
  etp = 1,
  fault: Fault = "none",
) {
  const source = fixture.families[family];
  if (!source) throw Error("unsupported family");
  if (
    padding.length !== ids.length ||
    padding.some((v, i) => v && mask[i] !== 0)
  )
    throw Error("invalid padding mask or padded supervision");
  const first = padding.indexOf(true);
  if (first >= 0 && padding.slice(first).some((v) => !v))
    throw Error("only suffix padding supported");
  expertGroups(ep, etp);
  const f = {
    ...base,
    parameters: { ...base.parameters, ...source.parameters },
  };
  const c = source.config,
    mla: Array<ReturnType<typeof computeMLA> | null> = [null, null];
  const params = (l: number): Params =>
    Object.fromEntries(
      Object.entries(f.parameters)
        .filter(([k]) => k.startsWith("l" + l + "_"))
        .map(([k, v]) => [k.slice(3), v]),
    );
  const trace = runDecoder(f, ids, labels, mask, false, {
    attention: (x, l) => {
      if (c.attention === "gqa") return denseAttention(f, x, l);
      const p = params(l),
        norm = x.map((row) => rms(row, p.input_gain as number[], f.epsilon));
      const result = computeMLA(norm, p, c.q_rank, f.epsilon, fault);
      mla[l] = result;
      return {
        norm,
        residual: x.map((row, t) => add(row, result.projected[t])),
      };
    },
    ffn: (x, l): FamilyFFN =>
      c.moe_layers.includes(l)
        ? computeMoE(x, params(l), c, padding, ep, etp, fault)
        : denseFeedForward(f, x, l),
  });
  const aux_loss = sum(trace.layers.map((layer) => layer.moe?.aux_loss ?? 0));
  return { ...trace, aux_loss, total_loss: trace.loss + aux_loss, mla };
}
