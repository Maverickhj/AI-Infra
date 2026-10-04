import assert from "node:assert/strict";
import fs from "node:fs";
import { spawnSync } from "node:child_process";
import {
  parseTraceJSON,
  validateTrace,
  compareTraces,
  hashText,
} from "../web/runtime/trace.ts";
const env = { ...process.env, CUDA_VISIBLE_DEVICES: "" };
delete env.PYTHONPATH;
function python(args, input) {
  const r = spawnSync("python", args, {
    env,
    input,
    encoding: "utf8",
    maxBuffer: 16 * 1024 * 1024,
    timeout: 180000,
  });
  assert.equal(r.status, 0, r.stdout + "\n" + r.stderr);
  return r.stdout;
}
python([
  "-S",
  "-m",
  "unittest",
  "tests.test_runtime_contracts",
  "tests.test_runtime_capture",
  "-v",
]);
python(["-m", "tests.runtime_adapter_cpu_checks"]);
python(["-m", "tests.runtime_config_checks"]);
const fresh = JSON.parse(
  python([
    "-c",
    "import json;from experiments.runtime.reference_trace import generate;print(json.dumps(generate(),allow_nan=False))",
  ]),
);
const fixture = JSON.parse(
  fs.readFileSync(
    new URL("../content/fixtures/runtime-reference.json", import.meta.url),
    "utf8",
  ),
);
const inputs = [];
for (let i = 0; i < fresh.traces.length; i++) {
  const r = fresh.traces[i],
    stored = fixture.traces[i];
  assert.equal(r.config_sha256, stored.config_sha256);
  assert.equal(r.input_sha256, stored.input_sha256);
  assert.deepEqual(r.measurements, stored.measurements);
  inputs.push({
    name: r.task + " fresh CPU",
    text: JSON.stringify(r),
    valid: true,
  });
  const value = await validateTrace(JSON.stringify(r));
  assert.equal(compareTraces(value, value).max_abs_difference, 0);
  for (const [name, mutate] of [
    ["bad hash", (t) => (t.input_sha256 = "0".repeat(64))],
    ["unknown mapping", (t) => (t.manifest.adapter = "unknown_v9")],
    ["noncanonical layout", (t) => (t.manifest.layout = "SB")],
    [
      "loss mismatch",
      (t) =>
        (t.measurements[t.task === "sft" ? "loss_mean" : "policy_loss"] += 1),
    ],
    [
      "synthetic observed",
      (t) => {
        t.provenance = t.task === "sft" ? "observed_bridge" : "observed_rl";
        t.manifest.evidence_kind = "synthetic_contract";
      },
    ],
    ["same version wrong dtype", (t) => (t.manifest.dtype = "guess")],
  ]) {
    const bad = structuredClone(r);
    mutate(bad);
    inputs.push({
      name: r.task + " " + name,
      text: JSON.stringify(bad),
      valid: false,
    });
  }
  const changed = structuredClone(r);
  changed.manifest.software.torch =
    "different-version-for-synthetic-contract-test";
  changed.manifest.evidence_kind = "synthetic_contract";
  changed.manifest.execution.status = "synthetic";
  changed.manifest.execution.synthetic = true;
  inputs.push({
    name: r.task + " version independent synthetic",
    text: JSON.stringify(changed),
    valid: true,
  });
  const d = JSON.parse(r.input_json);
  delete d[r.task === "sft" ? "loss_mask" : "response_mask"];
  const bad = structuredClone(r);
  bad.input_json = JSON.stringify(d);
  bad.input_sha256 = await hashText(bad.input_json);
  inputs.push({
    name: r.task + " missing mask",
    text: JSON.stringify(bad),
    valid: false,
  });
}
for (const text of [
  '{"x":1,"x":2}',
  '{"constructor":{}}',
  '{"x":NaN}',
  "[".repeat(40) + "0" + "]".repeat(40),
  " ".repeat(1048577),
  '{\u00a0"x":1}',
])
  inputs.push({ name: "bounded strict JSON", text, valid: false });
const script = `import json,sys
from experiments.runtime.contracts import read_trace
out=[]
for text in json.load(sys.stdin):
 try:read_trace(text);out.append(True)
 except ValueError:out.append(False)
print(json.dumps(out))
`;
const accepts = JSON.parse(
  python(["-S", "-c", script], JSON.stringify(inputs.map((x) => x.text))),
);
for (let i = 0; i < inputs.length; i++) {
  const t = inputs[i];
  let valid;
  try {
    await validateTrace(t.text);
    valid = true;
  } catch {
    valid = false;
  }
  assert.equal(valid, t.valid, t.name + " TypeScript");
  assert.equal(accepts[i], t.valid, t.name + " Python");
}
assert.throws(() => parseTraceJSON('{"x":1,"x":2}'), /duplicate/);
console.log(
  JSON.stringify(
    {
      status: "passed",
      cross_language_cases: inputs.length,
      fresh_cpu_traces: fresh.traces.length,
      adapter_cpu_tests: 11,
      python_contract_tests: 13,
      python_capture_tests: 7,
      configuration_only_tests: 6,
      scope:
        "authored CPU and explicitly synthetic contracts; production runtime not executed",
    },
    null,
    2,
  ),
);
