/** Float64 teaching reference only. B=1 batch axis is omitted from returned arrays. */
export type Variant = "qwen3" | "qwen25";
export type Fault = "none" | "omit_scale" | "wrong_group";
export type Fixture = {
  dimensions: {
    B: number;
    S: number;
    H: number;
    nq: number;
    nkv: number;
    d: number;
  };
  epsilon: number;
  theta: number;
  positions: number[];
  rotary_layout: string;
  x: number[][];
  wqkv: number[][];
  wo: number[][];
  input_gain: number[];
  q_gain: number[];
  k_gain: number[];
  qkv_bias: number[];
  output_bias: number[];
};
const seq = (length: number) => Array.from({ length }, (_, i) => i);
const dot = (a: number[], b: number[]) =>
  a.reduce((sum, v, i) => sum + v * b[i], 0);
const linear = (x: number[][], w: number[][], bias: number[]) =>
  x.map((row) =>
    bias.map((b, j) => row.reduce((sum, v, i) => sum + v * w[i][j], 0) + b),
  );
function rms(row: number[], gain: number[], epsilon: number) {
  const inv = Math.sqrt(dot(row, row) / row.length + epsilon);
  return row.map((v, i) => (v * gain[i]) / inv);
}
function rope(row: number[], position: number, theta: number) {
  const half = row.length / 2;
  return row.map((value, i) => {
    const angle = position / theta ** ((2 * (i % half)) / row.length);
    const rotated = i < half ? -row[i + half] : row[i - half];
    return value * Math.cos(angle) + rotated * Math.sin(angle);
  });
}
export function validateSelection(head: number, token: number) {
  if (
    !Number.isInteger(head) ||
    head < 0 ||
    head >= 4 ||
    !Number.isInteger(token) ||
    token < 0 ||
    token >= 4
  )
    throw new Error("head/token 必须是 0–3 的整数");
  return Math.floor(head / 2);
}
export function softmax(row: number[]) {
  if (!row.length || row.every((v) => v === -Infinity))
    throw new Error("All-masked row：教学参考没有可读取的 key");
  if (
    row.some((v) => Number.isNaN(v) || v === Infinity || typeof v !== "number")
  )
    throw new Error("Invalid softmax input");
  const maximum = Math.max(...row);
  const values = row.map((v) => Math.exp(v - maximum));
  const total = values.reduce((a, b) => a + b, 0);
  return values.map((v) => v / total);
}
function validate(f: Fixture) {
  const expected = { B: 1, S: 4, H: 8, nq: 4, nkv: 2, d: 4 };
  if (
    Object.entries(expected).some(
      ([key, value]) => f.dimensions[key as keyof typeof expected] !== value,
    )
  )
    throw new Error(
      "Only the declared B1/S4/H8/nq4/nkv2/d4 fixture is supported",
    );
  const tensor = (value: unknown, shape: number[]): void => {
    if (shape.length) {
      if (!Array.isArray(value) || value.length !== shape[0])
        throw new Error("Invalid tensor shape or empty input");
      value.forEach((child) => tensor(child, shape.slice(1)));
    } else if (typeof value !== "number" || !Number.isFinite(value))
      throw new Error("Tensor entries must be finite numbers");
  };
  const shapes: [keyof Fixture, number[]][] = [
    ["x", [4, 8]],
    ["wqkv", [8, 32]],
    ["wo", [16, 8]],
    ["input_gain", [8]],
    ["q_gain", [4]],
    ["k_gain", [4]],
    ["qkv_bias", [32]],
    ["output_bias", [8]],
    ["positions", [4]],
  ];
  shapes.forEach(([key, shape]) => tensor(f[key], shape));
  if (
    f.rotary_layout !== "split_half" ||
    [f.epsilon, f.theta].some((v) => !Number.isFinite(v) || v <= 0) ||
    f.positions.some((p) => !Number.isInteger(p) || p < 0)
  )
    throw new Error("Invalid epsilon, theta, position IDs or RoPE layout");
}
export function compute(
  f: Fixture,
  variant: Variant = "qwen3",
  fault: Fault = "none",
) {
  validate(f);
  if (
    !["qwen3", "qwen25"].includes(variant) ||
    !["none", "omit_scale", "wrong_group"].includes(fault)
  )
    throw new Error("Unknown variant/fault");
  const norm = f.x.map((row) => rms(row, f.input_gain, f.epsilon));
  const mixed = linear(
    norm,
    f.wqkv,
    variant === "qwen25" ? f.qkv_bias : Array(32).fill(0),
  );
  const q = mixed.map((row) =>
    seq(4).map((h) =>
      row.slice(
        Math.floor(h / 2) * 16 + (h % 2) * 4,
        Math.floor(h / 2) * 16 + (h % 2) * 4 + 4,
      ),
    ),
  );
  const k = mixed.map((row) =>
    seq(2).map((g) => row.slice(g * 16 + 8, g * 16 + 12)),
  );
  const v = mixed.map((row) =>
    seq(2).map((g) => row.slice(g * 16 + 12, g * 16 + 16)),
  );
  const qnorm = q.map((token) =>
    token.map((head) =>
      variant === "qwen3" ? rms(head, f.q_gain, f.epsilon) : [...head],
    ),
  );
  const knorm = k.map((token) =>
    token.map((head) =>
      variant === "qwen3" ? rms(head, f.k_gain, f.epsilon) : [...head],
    ),
  );
  const qrope = qnorm.map((token, t) =>
    token.map((head) => rope(head, f.positions[t], f.theta)),
  );
  const krope = knorm.map((token, t) =>
    token.map((head) => rope(head, f.positions[t], f.theta)),
  );
  const group = (h: number) =>
    (Math.floor(h / 2) + (fault === "wrong_group" ? 1 : 0)) % 2;
  const scores = qrope.map((token) =>
    token.map((head, h) => krope.map((keys) => dot(head, keys[group(h)]))),
  );
  const scaled = scores.map((token) =>
    token.map((head) =>
      head.map((value) => value / (fault === "omit_scale" ? 1 : Math.sqrt(4))),
    ),
  );
  const masked = scaled.map((token, t) =>
    token.map((head) => head.map((value, j) => (j <= t ? value : -Infinity))),
  );
  const probabilities = masked.map((token) => token.map(softmax));
  const heads = probabilities.map((token) =>
    token.map((p, h) =>
      seq(4).map((i) =>
        p.reduce((sum, weight, j) => sum + weight * v[j][group(h)][i], 0),
      ),
    ),
  );
  const merged = heads.map((token) => token.flat());
  const projected = linear(merged, f.wo, f.output_bias);
  const residual = projected.map((row, t) =>
    row.map((value, i) => value + f.x[t][i]),
  );
  return {
    input: f.x,
    norm,
    mixed,
    q,
    k,
    v,
    qnorm,
    knorm,
    qrope,
    krope,
    scores,
    scaled,
    masked,
    probabilities,
    heads,
    merged,
    projected,
    residual,
  };
}
export type Computation = ReturnType<typeof compute>;
