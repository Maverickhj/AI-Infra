import assert from "node:assert/strict";
import fs from "node:fs";
import { spawnSync } from "node:child_process";
import { computeDecoder } from "../web/decoder/compute.ts";
import { tokenize, align } from "../web/sft/compute.ts";
const load = (p) =>
  JSON.parse(fs.readFileSync(new URL(p, import.meta.url), "utf8"));
const fixture = load("../content/fixtures/decoder-reference.json"),
  data = load("../content/fixtures/sft-data.json");
const env = { ...process.env, CUDA_VISIBLE_DEVICES: "" };
delete env.PYTHONPATH;
const r = spawnSync(
  "python",
  ["-m", "experiments.decoder_reference", "--forward"],
  { encoding: "utf8", env, maxBuffer: 16 * 1024 * 1024 },
);
assert.equal(r.status, 0, r.stderr);
let count = 0,
  maxError = 0;
function compare(a, b, path = "") {
  if (Array.isArray(b)) {
    assert.equal(a.length, b.length, path);
    b.forEach((v, i) => compare(a[i], v, path + "[" + i + "]"));
    return;
  }
  if (b && typeof b === "object") {
    assert.deepEqual(Object.keys(a).sort(), Object.keys(b).sort());
    for (const k of Object.keys(b)) compare(a[k], b[k], path + "." + k);
    return;
  }
  assert.ok(Number.isFinite(a) && Number.isFinite(b), path);
  const err = Math.abs(a - b);
  assert.ok(err <= 1e-10 + 1e-10 * Math.abs(b), path + ": " + a + " vs " + b);
  count++;
  maxError = Math.max(maxError, err);
}
const references = JSON.parse(r.stdout);
for (const ref of references) {
  const rows = align(
    tokenize(data, data.samples[ref.sample], ref.mode),
    data.samples[ref.sample].id,
  );
  const ids = rows.map((r) => r.input.id),
    labels = rows.map((r) => Math.max(0, r.target.id)),
    mask = rows.map((r) => r.mask);
  const result = computeDecoder(fixture, ids, labels, mask, ref.tied);
  compare(result, ref.trace, ref.mode);
  const missingFinal = structuredClone(fixture);
  missingFinal.parameters.final_gain = Array(8).fill(0);
  assert.ok(
    Math.abs(
      computeDecoder(missingFinal, ids, labels, mask, ref.tied).loss -
        result.loss,
    ) > 1e-5,
  );
  const doubleLabels = [...labels.slice(1), 0];
  assert.ok(
    Math.abs(
      computeDecoder(fixture, ids, doubleLabels, mask, ref.tied).loss -
        result.loss,
    ) > 1e-5,
  );
  assert.throws(
    () =>
      computeDecoder(
        fixture,
        ids,
        labels,
        mask.map(() => 0),
      ),
    /no_supervision/,
  );
}
const checks = spawnSync("python", ["-m", "tests.decoder_cpu_checks"], {
  encoding: "utf8",
  env,
});
assert.equal(checks.status, 0, checks.stderr);
process.stdout.write(checks.stderr);
console.log(
  JSON.stringify(
    {
      status: "passed",
      scenarios: references.length,
      values_compared: count,
      max_abs_error: maxError,
      atol: 1e-10,
      rtol: 1e-10,
      scope: "authored two-layer float64 CPU model, not observed_bridge",
    },
    null,
    2,
  ),
);
