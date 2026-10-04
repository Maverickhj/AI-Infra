import assert from "node:assert/strict";
import fs from "node:fs";
import { spawnSync } from "node:child_process";
import {
  tokenize,
  align,
  batch,
  unpack,
  referenceLoss,
  validateAlignment,
  parseTokenizerTrace,
} from "../web/sft/compute.ts";
const fixture = JSON.parse(
  fs.readFileSync(
    new URL("../content/fixtures/sft-data.json", import.meta.url),
  ),
);
const requests = [];
for (const mode of ["assistant", "last_turn", "full"])
  for (const indices of [[0], [1], [2], [1, 2]])
    for (const limit of [2, 8, 12, 128])
      for (const padding of [1, 8])
        requests.push({ indices, mode, limit, padding });
const env = { ...process.env };
delete env.PYTHONPATH;
const py = spawnSync(
  "uv",
  ["run", "--no-project", "python", "-S", "experiments/sft_data_reference.py"],
  { input: JSON.stringify(requests), encoding: "utf8", env },
);
assert.equal(py.status, 0, py.stderr);
const references = JSON.parse(py.stdout);
let compared = 0,
  maxError = 0;
const near = (a, b) => {
  assert.ok(Number.isFinite(a) && Number.isFinite(b));
  const e = Math.abs(a - b);
  maxError = Math.max(maxError, e);
  compared++;
  assert.ok(e <= 1e-11 + 1e-11 * Math.abs(b), a + " vs " + b);
};
requests.forEach((r, k) => {
  const sequences = r.indices.map((i) =>
    align(
      tokenize(fixture, fixture.samples[i], r.mode, r.limit),
      fixture.samples[i].id,
    ),
  );
  const packed = batch(sequences, r.padding),
    loss = referenceLoss(packed, fixture.vocabulary.length),
    ref = references[k];
  for (const [key, fn] of Object.entries({
    ids: (r) => r.input.id,
    targets: (r) => r.target.id,
    labels: (r) => r.label,
    masks: (r) => r.mask,
    positions: (r) => r.position,
    segments: (r) => r.sample,
    valid: (r) => r.valid,
  }))
    assert.deepEqual(packed.rows.map(fn), ref[key]);
  for (const key of ["attention", "cuSeqlens", "cuSeqlensPadded", "lengths"])
    assert.deepEqual(packed[key], ref[key]);
  for (const key of ["contexts", "losses"])
    loss[key].forEach((n, i) => near(n, ref[key][i]));
  near(loss.sum, ref.sum);
  assert.equal(loss.count, ref.count);
  assert.equal(loss.status, ref.status);
  if (loss.mean === null) assert.equal(ref.mean, null);
  else near(loss.mean, ref.mean);
  assert.deepEqual(unpack(packed), sequences);
  const singles = sequences.map((s) =>
    referenceLoss(batch([s]), fixture.vocabulary.length),
  );
  near(
    loss.sum,
    singles.reduce((s, x) => s + x.sum, 0),
  );
});
const tokens = tokenize(fixture, fixture.samples[0], "assistant"),
  rows = align(tokens, "one");
assert.deepEqual(
  rows.slice(7, 10).map((r) => [r.input.text, r.target.text, r.mask]),
  [
    ["<assistant>", "5", 1],
    ["5", "。", 1],
    ["。", "<eos>", 1],
  ],
);
assert.equal(
  rows.reduce((s, r) => s + r.mask, 0),
  6,
);
validateAlignment(tokens, rows);
const double = rows.map((r, i) => ({
  ...r,
  target: tokens[i + 2] || r.target,
}));
assert.throws(() => validateAlignment(tokens, double), /shift/);
const wrongMask = rows.map((r) => ({ ...r, mask: r.input.supervised }));
assert.throws(() => validateAlignment(tokens, wrongMask), /shift/);
assert.throws(() => align(tokens, "one", 2), /shift/);
const seq = [1, 2].map((i) =>
  align(tokenize(fixture, fixture.samples[i], "full"), fixture.samples[i].id),
);
const wrongCross = seq[0].map((r, i) =>
  i === seq[0].length - 1
    ? { ...r, target: seq[1][0].input, mask: 1, label: 1 }
    : r,
);
assert.throws(
  () =>
    validateAlignment(
      tokenize(fixture, fixture.samples[1], "full"),
      wrongCross,
    ),
  /跨样本/,
);
const correct = referenceLoss(batch(seq), fixture.vocabulary.length),
  leaky = referenceLoss(batch(seq, 1, true), fixture.vocabulary.length);
assert.ok(Math.abs(correct.sum - leaky.sum) > 1e-4);
const two = seq.map((s) =>
  referenceLoss(batch([s]), fixture.vocabulary.length),
);
assert.ok(Math.abs(correct.mean - (two[0].mean + two[1].mean) / 2) > 1e-5);
const empty = structuredClone(fixture.samples[1]);
empty.messages[1].pieces = [];
empty.messages[1].content = "";
assert.equal(
  referenceLoss(
    batch([align(tokenize(fixture, empty, "assistant"), empty.id)]),
    fixture.vocabulary.length,
  ).count,
  1,
);
const trace = {
  schema_version: 1,
  kind: "tokenizer_trace",
  model_id: "synthetic/parser-test",
  tokenizer_revision: "a".repeat(40),
  template_sha256: "b".repeat(64),
  loss_mode: "assistant",
  rendered_text: "p a",
  messages: [
    { role: "user", content: "p" },
    { role: "assistant", content: "a" },
  ],
  tokens: [
    { id: 5, text: "p" },
    { id: 6, text: "a" },
  ],
  loss_mask: [0, 1],
};
const imported = parseTokenizerTrace(JSON.stringify(trace));
assert.equal(imported.provenance, "external_unverified");
assert.equal(imported.rows[0].label, 6);
assert.equal(imported.rows[1].label, -100);
for (const patch of [
  { loss_mask: [1] },
  { tokenizer_revision: "main" },
  { tokens: [{ id: -1, text: "bad" }] },
  { schema_version: 2 },
])
  assert.throws(() =>
    parseTokenizerTrace(JSON.stringify({ ...trace, ...patch })),
  );
assert.throws(() => parseTokenizerTrace("null"));
console.log(
  JSON.stringify(
    {
      status: "passed",
      scenarios: requests.length,
      finite_values_compared: compared,
      max_abs_error: maxError,
      atol: 1e-11,
      rtol: 1e-11,
      counterexamples: [
        "double shift",
        "input-aligned mask",
        "cross-sample prediction",
        "cross-sample attention",
        "mean of means",
      ],
      provenance:
        "authored token fixture; independent Python vs production TS; not official tokenizer",
    },
    null,
    2,
  ),
);

const padded = batch(seq, 8, false, 24);
assert.deepEqual(padded.cuSeqlensPadded, [0, 24, 48]);
near(referenceLoss(padded, fixture.vocabulary.length).sum, correct.sum);
