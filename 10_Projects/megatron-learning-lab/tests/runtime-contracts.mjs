import assert from "node:assert/strict";
import fs from "node:fs";
import { spawnSync } from "node:child_process";
import {
  parseTraceJSON,
  validateTrace,
  compareTraces,
  communicationTheory,
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
  "tests.test_runtime_plan",
  "tests.test_runtime_token_data",
  "tests.test_runtime_launch",
  "tests.test_runtime_nemo_plan",
  "tests.test_runtime_nemo_entry",
  "tests.test_runtime_nemo_extension",
  "tests.test_runtime_nemo_events",
  "tests.test_runtime_nemo_loading",
  "tests.test_runtime_nemo_startup",
  "-v",
]);
python(["-m", "tests.runtime_adapter_cpu_checks"]);
python(["-m", "tests.runtime_config_checks"]);
python(["-m", "tests.runtime_bridge_config_checks"]);
python(["-m", "tests.runtime_nemo_capture_checks"]);
python(["-m", "tests.runtime_nemo_observer_checks"]);
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
const acknowledged = structuredClone(fresh.traces.find((t) => t.task === "rl"));
acknowledged.manifest.evidence_kind = "synthetic_contract";
acknowledged.manifest.execution = { status: "synthetic", synthetic: true };
acknowledged.measurements.refit = {
  status: "acknowledged",
  completed: true,
  generation_version: 2,
  evidence: "official_delegate_return",
  weight_hash_verified: false,
  event: 9,
};
inputs.push({
  name: "synthetic refit call acknowledgment",
  text: JSON.stringify(acknowledged),
  valid: true,
});
for (const change of [
  { weight_hash_verified: true },
  { export_hash: "a".repeat(64) },
  { generation_version: 1 },
  { evidence: "guessed" },
  { event: true },
  { status: "synchronized" },
]) {
  const bad = structuredClone(acknowledged);
  Object.assign(bad.measurements.refit, change);
  inputs.push({
    name: "invalid refit evidence " + JSON.stringify(change),
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

const wide = JSON.stringify({
  model: Object.fromEntries(
    Array.from({ length: 308 }, (_, i) => ["field_" + i, i]),
  ),
  threshold: { float_sentinel: "+inf" },
});
for (const [name, inner, hashOverride, valid] of [
  ["308-field effective config", wide, null, true],
  ["wrong inner hash", wide, "0".repeat(64), false],
  ["duplicate inner key", '{"x":1,"x":2}', null, false],
  ["unsafe inner key", '{"constructor":{}}', null, false],
  ["nonfinite inner value", '{"x":NaN}', null, false],
  ["inner is not object", "[]", null, false],
  ["invalid float tag", '{"float_sentinel":"NaN"}', null, false],
  [
    "wide inner object",
    JSON.stringify(
      Object.fromEntries(Array.from({ length: 513 }, (_, i) => ["x" + i, i])),
    ),
    null,
    false,
  ],
  [
    "deep inner object",
    '{"x":' + "[".repeat(34) + "0" + "]".repeat(34) + "}",
    null,
    false,
  ],
  [
    "large inner array",
    JSON.stringify({ x: Array(65537).fill(0) }),
    null,
    false,
  ],
]) {
  const trace = structuredClone(fresh.traces[0]);
  trace.manifest.evidence_kind = "synthetic_contract";
  trace.manifest.execution = { status: "synthetic", synthetic: true };
  const config = JSON.parse(trace.config_json);
  Object.assign(config, {
    effective_config_json: inner,
    effective_config_sha256: hashOverride ?? (await hashText(inner)),
  });
  trace.config_json = JSON.stringify(config);
  trace.config_sha256 = await hashText(trace.config_json);
  inputs.push({ name, text: JSON.stringify(trace), valid });
}
const nestedBadConfig = structuredClone(fresh.traces[0]);
nestedBadConfig.measurements.worker_loading = {
  effective_config_json: wide,
  effective_config_sha256: "0".repeat(64),
};
inputs.push({
  name: "nested worker config hash",
  text: JSON.stringify(nestedBadConfig),
  valid: false,
});
const wideOuter = structuredClone(fresh.traces[0]);
wideOuter.manifest.unbounded = Object.fromEntries(
  Array.from({ length: 257 }, (_, i) => ["x" + i, i]),
);
inputs.push({
  name: "outer object bound unchanged",
  text: JSON.stringify(wideOuter),
  valid: false,
});

const rlValid = await validateTrace(JSON.stringify(fresh.traces[1]));
const same = compareTraces(rlValid, rlValid);
assert.equal(same.loss.absolute_difference, 0);
assert.equal(same.gradient.absolute_difference, 0);
assert.equal(same.gradient.update_absolute_difference, 0);
assert.deepEqual(same.shape, [4, 13]);
const budget = communicationTheory(rlValid, 4);
assert.equal(budget.elements, 224); // 27*8 actor head + 8 critic
assert.equal(budget.bytes, 1792); // float64 gradients
assert.equal(budget.sent_bytes_per_rank, 2688); // 2*(4-1)/4*1792
assert.equal(budget.received_bytes_per_rank, 2688);
assert.equal(budget.observed, null);
assert.equal(communicationTheory(rlValid, 1).sent_bytes_per_rank, 0);
const sftValid = await validateTrace(JSON.stringify(fresh.traces[0]));
assert.equal(communicationTheory(sftValid, 4), null);
assert.match(compareTraces(sftValid, sftValid).gradient.reason, /未采集/);
const changedGradient = structuredClone(fresh.traces[1]);
changedGradient.manifest.evidence_kind = "synthetic_contract";
changedGradient.manifest.execution = { status: "synthetic", synthetic: true };
changedGradient.measurements.update.gradient += 0.125;
changedGradient.measurements.update.after += 0.25;
const changedValid = await validateTrace(JSON.stringify(changedGradient));
assert.ok(
  Math.abs(
    compareTraces(rlValid, changedValid).gradient.absolute_difference - 0.125,
  ) < 1e-12,
);
assert.ok(
  Math.abs(
    compareTraces(rlValid, changedValid).gradient.update_absolute_difference -
      0.25,
  ) < 1e-12,
);
changedGradient.measurements.update.parameter = "head[11,0]";
assert.match(
  compareTraces(rlValid, await validateTrace(JSON.stringify(changedGradient)))
    .gradient.reason,
  /参数坐标/,
);
changedGradient.measurements.update.parameter = "head[10,0]";
changedGradient.measurements.update.before += 0.1;
assert.match(
  compareTraces(rlValid, await validateTrace(JSON.stringify(changedGradient)))
    .gradient.reason,
  /更新前值/,
);
for (const [name, mutate] of [
  ["wrong gradient scalar", (t) => (t.measurements.update.gradient = "0.1")],
  [
    "unknown parameter coordinate",
    (t) => (t.measurements.update.parameter = "head[unknown]"),
  ],
  [
    "null optimizer master",
    (t) => {
      t.manifest.adapter = "nemo_full_token_v1";
      t.measurements.update.optimizer_parameter = null;
    },
  ],
]) {
  const t = structuredClone(fresh.traces[1]);
  mutate(t);
  inputs.push({ name, text: JSON.stringify(t), valid: false });
}
for (const [name, mutate] of [
  ["unknown gradient geometry", (g) => (g.scope = "guessed")],
  ["empty gradient shape", (g) => (g.tensors[0].shape = [])],
  ["boolean gradient shape", (g) => (g.tensors[0].shape = [true])],
  [
    "oversized gradient shape",
    (g) => (g.tensors[0].shape = [10000000, 10000000]),
  ],
  ["unknown gradient dtype", (g) => (g.tensors[0].dtype = "packed")],
  [
    "duplicate gradient name",
    (g) => (g.tensors[1].parameter = g.tensors[0].parameter),
  ],
  ["unknown geometry field", (g) => (g.tensors[0].sent_bytes = 12)],
]) {
  const t = structuredClone(fresh.traces[1]),
    config = JSON.parse(t.config_json);
  mutate(config.gradient_geometry);
  t.config_json = JSON.stringify(config);
  t.config_sha256 = await hashText(t.config_json);
  inputs.push({ name, text: JSON.stringify(t), valid: false });
}

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
      python_contract_tests: 18,
      nemo_worker_extension_tests: 5,
      nemo_event_ordering_tests: 7,
      python_capture_tests: 8,
      nemo_checkpoint_loading_tests: 4,
      nemo_observer_cpu_tests: 4,
      configuration_only_tests: 7,
      nemo_bound_plan_tests: 7,
      nemo_launcher_delegation_tests: 6,
      nemo_loss_and_worker_cpu_tests: 9,
      runtime_plan_tests: 7,
      tokenizer_contract_tests: 4,
      subprocess_guard_tests: 7,
      nemo_startup_tests: 6,
      bridge_configuration_and_cpu_tests: 4,
      scope:
        "authored CPU and explicitly synthetic contracts; production runtime not executed",
    },
    null,
    2,
  ),
);
