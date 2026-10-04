import assert from "node:assert/strict";
import fs from "node:fs";
import { spawnSync } from "node:child_process";
import {
  computeRL,
  computePolicyTerms,
  computeGAE,
  groupAdvantages,
  stateHash,
  refitContract,
} from "../web/rl/compute.ts";
const load = (p) =>
  JSON.parse(fs.readFileSync(new URL(p, import.meta.url), "utf8"));
const base = load("../content/fixtures/decoder-reference.json"),
  fixture = load("../content/fixtures/rl-reference.json");
const env = { ...process.env, CUDA_VISIBLE_DEVICES: "" };
delete env.PYTHONPATH;
const run = spawnSync(
  "python",
  ["-m", "experiments.rl_reference", "--forward"],
  { env, encoding: "utf8", maxBuffer: 32 * 1024 * 1024 },
);
assert.equal(run.status, 0, run.stderr);
let count = 0,
  maxError = 0;
function compare(a, b, path = "root") {
  if (typeof b === "string" || typeof b === "boolean" || b === null) {
    assert.equal(a, b, path);
    return;
  }
  if (Array.isArray(b)) {
    assert.equal(a.length, b.length, path);
    b.forEach((v, i) => compare(a[i], v, path + "[" + i + "]"));
    return;
  }
  if (typeof b === "object") {
    assert.deepEqual(Object.keys(a).sort(), Object.keys(b).sort(), path);
    for (const k of Object.keys(b)) compare(a[k], b[k], path + "." + k);
    return;
  }
  assert.ok(Number.isFinite(a) && Number.isFinite(b), path);
  const error = Math.abs(a - b);
  assert.ok(error <= 1e-10 + 1e-10 * Math.abs(b), path + ": " + a + " vs " + b);
  count++;
  maxError = Math.max(maxError, error);
}
const rows = JSON.parse(run.stdout);
for (const r of rows) {
  const actual = computeRL(
    base,
    fixture,
    r.algorithm,
    r.reduction,
    r.kl_enabled,
    r.force_on_policy,
  );
  compare(actual, r, r.algorithm + "/" + r.reduction);
  if (r.force_on_policy) {
    assert.ok(actual.terms.ratio.flat().every((v) => v === 1));
    assert.ok(actual.head_gradient.flat().some((v) => Math.abs(v) > 1e-5));
  }
  assert.ok(
    actual.logprob_gradient
      .flat()
      .every((v, i) => actual.mask.flat()[i] || v === 0),
  );
  if (r.algorithm === "grpo")
    assert.ok(actual.critic_gradient.every((v) => v === 0));
}
const c = { ...fixture.config, kl_beta: 0 },
  lp = [[Math.log(1.5), Math.log(0.5), Math.log(0.5), Math.log(1.5)]],
  zero = [[0, 0, 0, 0]],
  adv = [[2, 2, -2, -2]],
  mask = [[1, 1, 1, 1]];
const hand = computePolicyTerms(lp, zero, zero, adv, mask, c, "token", false);
compare(hand.terms.pg_token, [[-2.4, -1, 1.6, 3]], "hand clipping");
compare(hand.dlogprob, [[0, -0.25, 0, 0.75]], "hand gradient");
const gap = computeGAE([1], [[0.2, 999, 0.4]], [[1, 0, 1]], 0.9, 0.8);
compare(gap.advantages, [[0.592, 0, 0.6]], "hand GAE");
compare(
  groupAdvantages(["a", "a"], [0.5, 0.5]).sequence_advantages,
  [0, 0],
  "equal group",
);
const forced = computeRL(base, fixture, "grpo", "token", false, true),
  constant = computeRL(
    base,
    fixture,
    "grpo",
    "token",
    false,
    true,
    false,
    "constant_ratio",
  );
assert.ok(forced.head_gradient.flat().some((v) => Math.abs(v) > 1e-5));
assert.ok(constant.head_gradient.flat().every((v) => v === 0));
const correct = computeRL(base, fixture, "ppo", "token", true),
  wrongKL = computeRL(
    base,
    fixture,
    "ppo",
    "token",
    true,
    false,
    false,
    "detach_kl_weight",
  );
assert.ok(
  correct.head_gradient
    .flat()
    .some((v, i) => Math.abs(v - wrongKL.head_gradient.flat()[i]) > 1e-5),
);
const hash = await stateHash(fixture.heads.current);
const pyhash = spawnSync(
  "python",
  [
    "-c",
    "import json;from experiments.rl_reference import FIXTURE,state_hash;print(state_hash(json.loads(FIXTURE.read_text())['heads']['current']))",
  ],
  { env, encoding: "utf8" },
);
assert.equal(pyhash.status, 0, pyhash.stderr);
assert.equal(hash, pyhash.stdout.trim());
assert.equal(
  (await refitContract(1, fixture.heads.current, 1, hash, 1, hash, true))
    .actual_rollout,
  "not_run",
);
await assert.rejects(
  () => refitContract(1, fixture.heads.current, 1, hash, 0, hash, true),
  /generation version/,
);
await assert.rejects(
  () => refitContract(1, fixture.heads.current, 1, hash, 1, hash, false),
  /not completed/,
);
const checks = spawnSync("python", ["-m", "tests.rl_cpu_checks"], {
  env,
  encoding: "utf8",
});
assert.equal(checks.status, 0, checks.stderr);
process.stdout.write(checks.stderr);
console.log(
  JSON.stringify(
    {
      status: "passed",
      scenarios: rows.length,
      values_compared: count,
      max_abs_error: maxError,
      atol: 1e-10,
      rtol: 1e-10,
      scope:
        "authored fixed trajectories; complete frozen decoder with trainable actor/critic heads; no rollout",
      parameter_hash: hash,
    },
    null,
    2,
  ),
);
