import modelsData from "../content/models.json";
import caseData from "../content/generated/model-cases.json";
import sourceData from "../research/source-evidence.json";
import lock from "../source.lock.json";
import qwen from "../content/cases/01_qwen3_complete_sft.md?raw";
import deepseek from "../content/cases/02_deepseek_complete_model.md?raw";
import rl from "../content/cases/03_complete_rl_update.md?raw";
import samplesRaw from "../content/samples.jsonl?raw";

export const models = modelsData.models;
export type Model = (typeof models)[number];
export const cases = caseData.cases;
export const sources = sourceData.entries;
export const sourceLock = lock;
export const samples: {
  id: string;
  messages: { role: string; content: string }[];
}[] = samplesRaw
  .trim()
  .split("\n")
  .map((line) => JSON.parse(line));
export const lessons = { qwen, deepseek, rl };
export const clean = (text: string) =>
  text.replace(/^---\n[\s\S]*?\n---\n/, "").trim();
export const sftSteps = [
  "input",
  "embedding",
  "decoder",
  "final_norm",
  "lm_head",
  "loss",
  "backward",
  "update",
] as const;
export const rlSteps = [
  "rollout",
  "reward",
  "logprobs",
  "advantage",
  "policy_update",
  "refit",
] as const;
export const labels: Record<string, string> = {
  input: "样本与目标",
  embedding: "Embedding",
  decoder: "Decoder",
  final_norm: "Final norm",
  lm_head: "LM head",
  loss: "Masked loss",
  backward: "Backward",
  update: "更新与恢复",
  rollout: "Rollout",
  reward: "Reward",
  logprobs: "四类 logprob",
  advantage: "Advantage",
  policy_update: "Policy update",
  refit: "Export / refit",
};
export type State = {
  model: string;
  scenario: "sft" | "rl";
  layer: number;
  step: string;
  sourceLane: "reference" | "runtime";
  view: "walkthrough" | "atlas" | "sample" | "course" | "basics";
  mode: "assistant" | "last_turn" | "full";
  mla: "train" | "decode";
  sample: number;
};
export const defaults: State = {
  model: "qwen3-06b",
  scenario: "sft",
  layer: 0,
  step: "decoder",
  sourceLane: "reference",
  view: "walkthrough",
  mode: "assistant",
  mla: "train",
  sample: 0,
};
export function normalize(s: State): State {
  const model = models.find((m) => m.id === s.model) || models[1];
  const scenario = s.scenario === "rl" ? "rl" : "sft";
  const steps: readonly string[] = scenario === "rl" ? rlSteps : sftSteps;
  return {
    ...defaults,
    ...s,
    model: model.id,
    scenario,
    layer: Number.isFinite(s.layer)
      ? Math.max(0, Math.min(model.layers - 1, Math.floor(s.layer)))
      : 0,
    step: steps.includes(s.step) ? s.step : steps[0],
    sourceLane: s.sourceLane === "runtime" ? "runtime" : "reference",
    view: ["walkthrough", "atlas", "sample", "course", "basics"].includes(
      s.view,
    )
      ? s.view
      : "walkthrough",
    mode: ["assistant", "last_turn", "full"].includes(s.mode)
      ? s.mode
      : "assistant",
    mla: s.mla === "decode" ? "decode" : "train",
    sample:
      Number.isInteger(s.sample) && s.sample >= 0 && s.sample < samples.length
        ? s.sample
        : 0,
  };
}
export function readState(): State {
  const p = new URLSearchParams(location.hash.slice(1));
  return normalize({
    ...defaults,
    ...Object.fromEntries(p),
    layer: Number(p.get("layer") || 0),
    sample: Number(p.get("sample") || 0),
  } as State);
}
export function reducer(s: State, patch: Partial<State>): State {
  return normalize({
    ...s,
    ...patch,
    ...(patch.scenario &&
    patch.scenario !== s.scenario &&
    patch.step === undefined
      ? { step: patch.scenario === "rl" ? "rollout" : "input" }
      : {}),
  });
}
export function sourceIds(s: State): string[] {
  if (s.scenario === "rl")
    return (
      (
        {
          rollout: ["R-WORKER"],
          reward: ["R-GRPO"],
          logprobs: ["R-LOSS", "R-TRAIN"],
          advantage: ["R-GRPO", "R-LOSS"],
          policy_update: ["R-LOSS", "R-TRAIN", "R-PPOCFG"],
          refit: ["R-WORKER"],
        } as Record<string, string[]>
      )[s.step] || []
    );
  const c = cases.find((c) => c.id === s.model)!;
  return [
    ...new Set([
      ...(c.pipeline.find((p) => p.id === s.step)?.source_ids || []),
      ...(s.step === "decoder" ? c.attention.source_ids : []),
    ]),
  ];
}
