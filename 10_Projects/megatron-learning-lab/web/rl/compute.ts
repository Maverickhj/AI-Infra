import { computeDecoder, type DecoderFixture } from "../decoder/compute.ts";
type Matrix = number[][];
export type Algorithm = "grpo" | "ppo";
export type Reduction = "token" | "sequence";
export type RLConfig = {
  clip: number;
  kl_beta: number;
  kl_input_clamp: number;
  kl_output_clamp: number;
  gamma: number;
  gae_lambda: number;
  value_clip: number;
  actor_lr: number;
  critic_lr: number;
  grpo_epsilon: number;
};
export type RLFixture = {
  trajectories: Array<{
    id: string;
    group: string;
    prompt: number[];
    response: number[];
    response_mask: number[];
    reward: number;
  }>;
  heads: Record<string, Matrix>;
  critic_old: number[];
  critic_current: number[];
  config: RLConfig;
};
export type RLFault =
  "none" | "constant_ratio" | "detach_kl_weight" | "wrong_clip";
const sum = (a: number[]) => a.reduce((s, v) => s + v, 0);
const dot = (a: number[], b: number[]) => sum(a.map((v, i) => v * b[i]));
const zeros = (rows: number, cols: number) =>
  Array.from({ length: rows }, () => Array(cols).fill(0) as number[]);
const clip = (v: number, lo: number, hi: number) =>
  Math.max(lo, Math.min(hi, v));
const softmax = (row: number[]) => {
  const max = Math.max(...row),
    v = row.map((z) => Math.exp(z - max)),
    n = sum(v);
  return v.map((z) => z / n);
};
export function prepareBatch(base: DecoderFixture, f: RLFixture) {
  const length = Math.max(
    ...f.trajectories.map((t) => t.prompt.length + t.response.length),
  );
  const ids: Matrix = [],
    mask: Matrix = [],
    features: number[][][] = [],
    groups: string[] = [],
    rewards: number[] = [];
  for (const item of f.trajectories) {
    if (
      item.response.length !== item.response_mask.length ||
      !item.response_mask.some(Boolean)
    )
      throw Error("missing response mask");
    const tokens = [...item.prompt, ...item.response],
      m = [...item.prompt.map(() => 0), ...item.response_mask];
    const labels = [...tokens.slice(1), 0],
      nextMask = [...m.slice(1), 0];
    const trace = computeDecoder(base, tokens, labels, nextMask, false),
      pad = length - tokens.length;
    features.push([
      Array(8).fill(0),
      ...trace.final_norm.slice(0, -1),
      ...zeros(pad, 8),
    ]);
    ids.push([...tokens, ...Array(pad).fill(0)]);
    mask.push([...m, ...Array(pad).fill(0)]);
    groups.push(item.group);
    rewards.push(item.reward);
  }
  return { ids, mask, features, groups, rewards };
}
export function groupAdvantages(
  groups: string[],
  rewards: number[],
  epsilon = 1e-6,
) {
  const baseline = rewards.map(() => 0),
    std = rewards.map(() => 0);
  for (const g of new Set(groups)) {
    const indices = groups
        .map((v, i) => (v === g ? i : -1))
        .filter((i) => i >= 0),
      mean = sum(indices.map((i) => rewards[i])) / indices.length;
    const sd =
      indices.length > 1
        ? Math.sqrt(
            sum(indices.map((i) => (rewards[i] - mean) ** 2)) /
              (indices.length - 1),
          )
        : 0;
    indices.forEach((i) => {
      baseline[i] = mean;
      std[i] = sd;
    });
  }
  return {
    baseline,
    std,
    sequence_advantages: rewards.map((v, i) =>
      std[i] > 0 ? (v - baseline[i]) / (std[i] + epsilon) : v - baseline[i],
    ),
  };
}
export function computeGAE(
  rewards: number[],
  values: Matrix,
  mask: Matrix,
  gamma = 0.9,
  lambda = 0.95,
) {
  const token_rewards = zeros(mask.length, mask[0].length),
    delta = zeros(mask.length, mask[0].length);
  const advantages = zeros(mask.length, mask[0].length),
    returns = zeros(mask.length, mask[0].length);
  mask.forEach((m, i) => {
    const valid = m.map((v, t) => (v ? t : -1)).filter((t) => t >= 0);
    if (!valid.length) throw Error("no response tokens");
    token_rewards[i][valid.at(-1)!] = rewards[i];
    valid.forEach((t, j) => {
      delta[i][t] =
        token_rewards[i][t] +
        gamma * (j + 1 < valid.length ? values[i][valid[j + 1]] : 0) -
        values[i][t];
    });
    // Direct finite sum over valid positions, independent of Python's backward
    // carry recurrence. Masked gaps do not insert another discount step.
    m.forEach((keep, t) => {
      const future = valid.filter((pos) => pos >= t);
      const raw = sum(
        future.map((pos, k) => (gamma * lambda) ** k * delta[i][pos]),
      );
      advantages[i][t] = keep ? raw : 0;
      returns[i][t] = raw + values[i][t];
    });
  });
  return { token_rewards, delta, advantages, returns };
}
export function reduceRows(values: Matrix, mask: Matrix, reduction: Reduction) {
  const counts = mask.map(sum),
    total = sum(counts);
  if (
    !["token", "sequence"].includes(reduction) ||
    total <= 0 ||
    counts.some((v) => v <= 0)
  )
    throw Error("invalid reduction or response mask");
  const rows = values.map((row, i) => sum(row.map((v, t) => v * mask[i][t])));
  return reduction === "token"
    ? sum(rows) / total
    : sum(rows.map((v, i) => v / counts[i])) / rows.length;
}
export function policyLogprobs(
  features: number[][][],
  head: Matrix,
  ids: Matrix,
) {
  const probabilities = features.map((row) =>
    row.map((x) => softmax(head.map((w) => dot(x, w)))),
  );
  const logprobs = features.map((row, b) =>
    row.map((x, t) => {
      if (t === 0) return 0;
      const logits = head.map((w) => dot(x, w)),
        max = Math.max(...logits);
      return (
        logits[ids[b][t]] -
        max -
        Math.log(sum(logits.map((v) => Math.exp(v - max))))
      );
    }),
  );
  return { logprobs, probabilities };
}
export function computePolicyTerms(
  curr: Matrix,
  prev: Matrix,
  ref: Matrix,
  adv: Matrix,
  mask: Matrix,
  c: RLConfig,
  reduction: Reduction,
  force: boolean,
  fault: RLFault = "none",
) {
  const ratio = curr.map((row, b) =>
    row.map((v, t) => (force ? 1 : Math.exp(v - prev[b][t]))),
  );
  const clipped_ratio = ratio.map((row) =>
    row.map((v) => (force ? v : clip(v, 1 - c.clip, 1 + c.clip))),
  );
  const pg_token = ratio.map((row, b) =>
    row.map((v, t) => {
      const a = -adv[b][t] * v,
        k = -adv[b][t] * clipped_ratio[b][t];
      return fault === "wrong_clip" ? Math.min(a, k) : Math.max(a, k);
    }),
  );
  const kl_token = curr.map((row, b) =>
    row.map((v, t) => {
      const d = clip(ref[b][t] - v, -c.kl_input_clamp, c.kl_input_clamp);
      return clip(Math.exp(d) - 1 - d, -c.kl_output_clamp, c.kl_output_clamp);
    }),
  );
  const counts = mask.map(sum),
    total = sum(counts);
  const dlogprob = curr.map((row, b) =>
    row.map((v, t) => {
      const advantage = adv[b][t],
        r = ratio[b][t];
      const normallyClipped =
        advantage > 0 ? r > 1 + c.clip : advantage < 0 ? r < 1 - c.clip : false;
      const active =
        fault === "wrong_clip"
          ? normallyClipped || (r >= 1 - c.clip && r <= 1 + c.clip)
          : !normallyClipped;
      let dpg = force
        ? fault === "constant_ratio"
          ? 0
          : -advantage
        : active
          ? -advantage * r
          : 0;
      const d = ref[b][t] - v,
        clamped = clip(d, -c.kl_input_clamp, c.kl_input_clamp),
        k = Math.exp(clamped) - 1 - clamped;
      let dkl =
        Math.abs(d) > c.kl_input_clamp || Math.abs(k) > c.kl_output_clamp
          ? 0
          : fault === "detach_kl_weight"
            ? 1 - Math.exp(d)
            : -d;
      const weight =
        mask[b][t] *
        (reduction === "token" ? 1 / total : 1 / (mask.length * counts[b]));
      return weight * (dpg + c.kl_beta * dkl);
    }),
  );
  const actor_loss = reduceRows(pg_token, mask, reduction),
    kl_loss = c.kl_beta * reduceRows(kl_token, mask, reduction);
  return {
    terms: {
      ratio,
      clipped_ratio,
      pg_token,
      kl_token,
      actor_loss,
      kl_loss,
      policy_loss: actor_loss + kl_loss,
    },
    dlogprob,
  };
}
export function computeValueTerms(
  values: Matrix,
  old: Matrix,
  returns: Matrix,
  mask: Matrix,
  c: RLConfig,
) {
  const clipped_values = values.map((row, b) =>
    row.map((v, t) =>
      clip(v, old[b][t] - c.value_clip, old[b][t] + c.value_clip),
    ),
  );
  const value_token = values.map((row, b) =>
    row.map(
      (v, t) =>
        0.5 *
        Math.max(
          (v - returns[b][t]) ** 2,
          (clipped_values[b][t] - returns[b][t]) ** 2,
        ),
    ),
  );
  const count = sum(mask.map(sum));
  const gradient = values.map((row, b) =>
    row.map((v, t) => {
      const raw = v - returns[b][t],
        limited = clipped_values[b][t] - returns[b][t];
      const g =
        raw ** 2 >= limited ** 2
          ? raw
          : Math.abs(v - old[b][t]) <= c.value_clip
            ? limited
            : 0;
      return (g * mask[b][t]) / count;
    }),
  );
  return {
    value: {
      values,
      old_values: old,
      returns,
      clipped_values,
      value_token,
      value_loss: reduceRows(value_token, mask, "token"),
    },
    gradient,
  };
}
export function computeRL(
  base: DecoderFixture,
  f: RLFixture,
  algorithm: Algorithm = "grpo",
  reduction: Reduction = "token",
  kl = true,
  force = false,
  equal_rewards = false,
  fault: RLFault = "none",
) {
  if (!["grpo", "ppo"].includes(algorithm))
    throw Error("unsupported RL algorithm");
  const data = prepareBatch(base, f),
    { ids, mask, features } = data,
    c = { ...f.config, kl_beta: kl ? f.config.kl_beta : 0 };
  const rewards = equal_rewards ? data.rewards.map(() => 0.5) : data.rewards;
  const head = (force ? f.heads.previous : f.heads.current).map((row) =>
      row.slice(),
    ),
    critic = f.critic_current.slice();
  const current = policyLogprobs(features, head, ids),
    previous_logprobs = policyLogprobs(
      features,
      f.heads.previous,
      ids,
    ).logprobs;
  const generation_logprobs = policyLogprobs(
      features,
      f.heads.generation,
      ids,
    ).logprobs,
    reference_logprobs = policyLogprobs(
      features,
      f.heads.reference,
      ids,
    ).logprobs;
  const group = groupAdvantages(data.groups, rewards, c.grpo_epsilon),
    old = features.map((row) => row.map((x) => dot(x, f.critic_old)));
  const gae = computeGAE(rewards, old, mask, c.gamma, c.gae_lambda);
  const advantages =
    algorithm === "grpo"
      ? mask.map((row, b) => row.map((m) => m * group.sequence_advantages[b]))
      : gae.advantages;
  const { terms, dlogprob } = computePolicyTerms(
    current.logprobs,
    previous_logprobs,
    reference_logprobs,
    advantages,
    mask,
    c,
    reduction,
    force,
    fault,
  );
  const vt = computeValueTerms(
    features.map((row) => row.map((x) => dot(x, critic))),
    old,
    gae.returns,
    mask,
    c,
  );
  const head_gradient = zeros(27, 8),
    critic_gradient = Array(8).fill(0) as number[];
  features.forEach((row, b) =>
    row.forEach((x, t) => {
      for (let v = 0; v < 27; v++)
        for (let d = 0; d < 8; d++)
          head_gradient[v][d] +=
            dlogprob[b][t] *
            ((ids[b][t] === v ? 1 : 0) - current.probabilities[b][t][v]) *
            x[d];
      if (algorithm === "ppo")
        x.forEach((value, d) => {
          critic_gradient[d] += vt.gradient[b][t] * value;
        });
    }),
  );
  const head_after = head.map((row, v) =>
    row.map((w, d) => w - c.actor_lr * head_gradient[v][d]),
  );
  const critic_after = critic.map(
    (w, d) => w - c.critic_lr * critic_gradient[d],
  );
  return {
    algorithm,
    critic_used: algorithm === "ppo",
    reduction,
    kl_enabled: kl,
    force_on_policy: force,
    equal_rewards,
    ids,
    mask,
    features,
    rewards,
    group,
    gae,
    advantages,
    probabilities: current.probabilities,
    generation_logprobs,
    previous_logprobs,
    current_logprobs: current.logprobs,
    reference_logprobs,
    terms,
    value: vt.value,
    head_gradient,
    logprob_gradient: dlogprob,
    critic_gradient,
    head_after,
    critic_after,
    logprobs_after: policyLogprobs(features, head_after, ids).logprobs,
    current_version: force ? 0 : 1,
    next_version: force ? 1 : 2,
    scope:
      "authored reference; frozen complete decoder; trainable policy/critic heads; no rollout",
  };
}
export async function stateHash(weights: Matrix) {
  if (
    weights.length !== 27 ||
    weights.some(
      (row) => row.length !== 8 || row.some((v) => !Number.isFinite(v)),
    )
  )
    throw Error("policy head must have finite shape [27,8]");
  const header = new TextEncoder().encode("f64le:27x8\n"),
    buffer = new ArrayBuffer(header.length + 27 * 8 * 8);
  new Uint8Array(buffer).set(header);
  const view = new DataView(buffer);
  weights
    .flat()
    .forEach((v, i) => view.setFloat64(header.length + i * 8, v, true));
  const digest = await crypto.subtle.digest("SHA-256", buffer);
  return [...new Uint8Array(digest)]
    .map((v) => v.toString(16).padStart(2, "0"))
    .join("");
}
export async function refitContract(
  policy_version: number,
  weights: Matrix,
  exported_version: number,
  exported_hash: string,
  generation_version: number,
  ack_hash: string,
  completed: boolean,
) {
  const digest = await stateHash(weights);
  if (
    !Number.isInteger(policy_version) ||
    policy_version < 0 ||
    exported_version !== policy_version
  )
    throw Error("policy/export version mismatch");
  if (exported_hash !== digest) throw Error("export weight hash mismatch");
  if (!completed) throw Error("refit not completed");
  if (generation_version !== policy_version || ack_hash !== digest)
    throw Error("generation version/refit acknowledgment mismatch");
  return {
    status: "reference_contract_synchronized",
    version: policy_version,
    weights_sha256: digest,
    actual_rollout: "not_run",
  };
}
