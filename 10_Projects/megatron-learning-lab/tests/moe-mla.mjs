import assert from "node:assert/strict";
import fs from "node:fs";
import { spawnSync } from "node:child_process";
import {
  computeFamily,
  computeMLA,
  expertGroups,
  route,
  familyData,
} from "../web/family/compute.ts";
import { tokenize, align } from "../web/sft/compute.ts";
const read = (p) =>
  JSON.parse(fs.readFileSync(new URL(p, import.meta.url), "utf8"));
const base = read("../content/fixtures/decoder-reference.json"),
  fixture = read("../content/fixtures/moe-mla-reference.json"),
  data = read("../content/fixtures/sft-data.json");
const env = { ...process.env, CUDA_VISIBLE_DEVICES: "" };
delete env.PYTHONPATH;
const r = spawnSync(
  "python",
  ["-m", "experiments.moe_mla_reference", "--forward"],
  { env, encoding: "utf8", maxBuffer: 24 * 1024 * 1024 },
);
assert.equal(r.status, 0, r.stderr);
let count = 0,
  maxError = 0;
function compare(a, b, path = "root") {
  if (b === null) {
    assert.equal(a, null, path);
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
const rows = JSON.parse(r.stdout);
for (const row of rows) {
  const aligned = align(
    tokenize(data, data.samples[row.sample], "assistant"),
    data.samples[row.sample].id,
  );
  const d = familyData(
    aligned.map((v) => v.input.id),
    aligned.map((v) => Math.max(0, v.target.id)),
    aligned.map((v) => v.mask),
    row.padding,
  );
  const args = [base, fixture, row.family, d.ids, d.labels, d.mask, d.padding];
  const result = computeFamily(...args);
  compare(result, row.trace, row.family);
  const config = fixture.families[row.family].config;
  for (const [ep, etp] of [
    [1, 1],
    [1, 2],
    [2, 1],
    [2, 2],
  ]) {
    const split = computeFamily(...args, ep, etp);
    compare(split.logits, result.logits, "expert shards");
    compare(split.total_loss, result.total_loss, "expert shard CE+aux");
    const groups = expertGroups(ep, etp);
    assert.equal(groups.length, ep * etp);
    groups.forEach((g) => {
      assert.ok(g.ep.includes(g.rank));
      assert.ok(g.etp.includes(g.rank));
      assert.deepEqual(g.edp, [g.rank]);
    });
  }
  assert.throws(
    () => computeFamily(...args, 1, 1, "duplicate_dispatch"),
    /duplicate/,
  );
  const wrong = computeFamily(...args, 1, 1, "wrong_weights");
  assert.ok(Math.abs(wrong.loss - result.loss) > 1e-6);
  if (config.shared_experts) {
    const missing = computeFamily(...args, 1, 1, "missing_shared");
    assert.ok(Math.abs(missing.loss - result.loss) > 1e-6);
  }
  if (config.attention === "mla") {
    for (const fault of ["wrong_norm", "wrong_scale", "missing_rope"]) {
      const bad = computeFamily(...args, 1, 1, fault);
      assert.ok(bad.mla[1].max_absorption_error > 1e-5, fault);
    }
  }
}
const c = fixture.families["qwen3-moe"].config;
const hand = route([[0, Math.log(2), Math.log(3), Math.log(4)]], c, [false]);
compare(hand.weights, [[4 / 7, 3 / 7]], "hand routing");
assert.throws(() => expertGroups(3, 1), /verified/);
assert.throws(() => expertGroups(2, 2, 2), /EDP1/);
const checks = spawnSync("python", ["-m", "tests.moe_mla_cpu_checks"], {
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
      expert_layouts: 4,
      values_compared: count,
      max_abs_error: maxError,
      atol: 1e-10,
      rtol: 1e-10,
      scope:
        "authored complete two-layer CPU reference; no observed Bridge or distributed runtime",
    },
    null,
    2,
  ),
);
