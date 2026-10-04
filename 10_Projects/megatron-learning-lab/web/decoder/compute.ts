import { compute as gqa, rms, linear } from "../gqa/compute.ts";
export type DecoderFixture = {
  epsilon: number;
  theta: number;
  parameters: Record<string, number[] | number[][]>;
  dimensions: { V: number; H: number; F: number; layers: number };
};
const transpose = (m: number[][]) => m[0].map((_, j) => m.map((row) => row[j]));
export function computeDecoder(
  f: DecoderFixture,
  ids: number[],
  labels: number[],
  mask: number[],
  tied = true,
) {
  if (
    ids.length < 1 ||
    ids.length > 64 ||
    labels.length !== ids.length ||
    mask.length !== ids.length ||
    ids.some((v) => !Number.isInteger(v) || v < 0 || v >= f.dimensions.V) ||
    labels.some((v) => !Number.isInteger(v) || v < 0 || v >= f.dimensions.V) ||
    mask.some((v) => v !== 0 && v !== 1) ||
    mask.reduce((a, b) => a + b, 0) <= 0
  )
    throw Error("invalid input/target/mask or no_supervision");
  const matrix = (key: string) => f.parameters[key] as number[][];
  const gain = (key: string) => f.parameters[key] as number[];
  const embedding = ids.map((id) => [...matrix("embedding")[id]]);
  let x = embedding;
  const layers = [];
  for (let l = 0; l < 2; l++) {
    const k = (name: string) => "l" + l + "_" + name;
    const a = gqa({
      dimensions: { B: 1, S: ids.length, H: 8, nq: 4, nkv: 2, d: 4 },
      epsilon: f.epsilon,
      theta: f.theta,
      positions: ids.map((_, i) => i),
      rotary_layout: "split_half",
      x,
      wqkv: transpose(matrix(k("qkv"))),
      wo: transpose(matrix(k("out"))),
      input_gain: gain(k("input_gain")),
      q_gain: gain(k("q_gain")),
      k_gain: gain(k("k_gain")),
      qkv_bias: Array(32).fill(0),
      output_bias: Array(8).fill(0),
    });
    const ffn_norm = a.residual.map((row) =>
      rms(row, gain(k("ffn_gain")), f.epsilon),
    );
    const mixed = linear(
      ffn_norm,
      transpose(matrix(k("gate_up"))),
      Array(24).fill(0),
    );
    const gate = mixed.map((row) => row.slice(0, 12)),
      up = mixed.map((row) => row.slice(12));
    const silu = gate.map((row) => row.map((v) => v / (1 + Math.exp(-v))));
    const product = silu.map((row, t) => row.map((v, i) => v * up[t][i]));
    const down = linear(
      product,
      transpose(matrix(k("down"))),
      Array(8).fill(0),
    );
    x = down.map((row, t) => row.map((v, i) => v + a.residual[t][i]));
    layers.push({
      input_norm: a.norm,
      attention: a.residual,
      ffn_norm,
      gate,
      up,
      silu,
      product,
      down,
      residual: x,
    });
  }
  const final_norm = x.map((row) => rms(row, gain("final_gain"), f.epsilon));
  const head = tied
    ? matrix("embedding")
    : matrix("head") || matrix("embedding");
  const logits = linear(
    final_norm,
    transpose(head),
    Array(f.dimensions.V).fill(0),
  );
  const logprobs = logits.map((row) => {
    const max = Math.max(...row),
      logZ = max + Math.log(row.reduce((s, z) => s + Math.exp(z - max), 0));
    return row.map((z) => z - logZ);
  });
  const token_loss = logprobs.map((row, i) => -row[labels[i]]);
  const loss =
    token_loss.reduce((s, v, i) => s + v * mask[i], 0) /
    mask.reduce((a, b) => a + b, 0);
  return {
    embedding,
    layers,
    final_norm,
    logits,
    logprobs,
    token_loss,
    loss,
    mask,
  };
}
