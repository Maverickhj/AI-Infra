import { steps as gqaSteps, stepFor } from "./gqa/steps";
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
  rlAlgorithm: "grpo" | "ppo";
  rlTrajectory: number;
  rlToken: number;
  rlReduction: "token" | "sequence";
  rlKl: "on" | "off";
  rlForce: "on" | "off";
  rlEqual: "on" | "off";
  rlFault: "none" | "constant_ratio" | "detach_kl_weight" | "wrong_clip";
  ep: number;
  etp: number;
  familyToken: number;
  familyExpert: number;
  familyHead: number;
  familyPadding: number;
  familyOp: string;
  familyFault:
    | "none"
    | "wrong_weights"
    | "duplicate_dispatch"
    | "missing_shared"
    | "wrong_norm"
    | "wrong_scale"
    | "missing_rope";
  model: string;
  scenario: "sft" | "rl";
  layer: number;
  step: string;
  sourceLane: "reference" | "runtime";
  view: "walkthrough" | "atlas" | "sample" | "course" | "basics" | "runtime";
  mode: "assistant" | "last_turn" | "full";
  mla: "train" | "decode";
  sample: number;
  operator: string;
  query: number;
  gqaHead: number;
  sftLayout: "single" | "unpacked" | "packed";
  sftLimit: number;
  sftPadding: number;
  sftQuery: number;
  decoderOp: string;
  decoderToken: number;
  tinyLayer: number;
  decoderTied: "tied" | "untied";
  tp: number;
  dp: number;
  parallelRank: number;
  parallelOp: string;
  pp: number;
  microbatches: number;
  pipelineMicrobatch: number;
  sequenceLayout: "ordinary" | "thd";
  cp: number;
  sequenceRank: number;
  sequenceQuery: number;
  sequenceFault: "none" | "local_kv" | "leak";
};
export const defaults: State = {
  rlAlgorithm: "grpo",
  rlTrajectory: 0,
  rlToken: 8,
  rlReduction: "token",
  rlKl: "on",
  rlForce: "off",
  rlEqual: "off",
  rlFault: "none",
  ep: 1,
  etp: 1,
  familyToken: 7,
  familyExpert: 0,
  familyHead: 0,
  familyPadding: 0,
  familyOp: "router",
  familyFault: "none",
  model: "qwen3-06b",
  scenario: "sft",
  layer: 0,
  step: "decoder",
  sourceLane: "reference",
  view: "walkthrough",
  mode: "assistant",
  mla: "train",
  sample: 0,
  operator: "overview",
  query: 0,
  gqaHead: 0,
  sftLayout: "single",
  sftLimit: 128,
  sftPadding: 1,
  sftQuery: 0,
  decoderOp: "ffn_norm",
  decoderToken: 7,
  tinyLayer: 0,
  decoderTied: "tied",
  tp: 2,
  dp: 2,
  parallelRank: 0,
  parallelOp: "qkv",
  pp: 2,
  microbatches: 4,
  pipelineMicrobatch: 0,
  sequenceLayout: "ordinary",
  cp: 2,
  sequenceRank: 0,
  sequenceQuery: 19,
  sequenceFault: "none",
};
export function normalize(s: State): State {
  const model = models.find((m) => m.id === s.model) || models[1];
  const scenario = s.scenario === "rl" ? "rl" : "sft";
  const steps: readonly string[] = scenario === "rl" ? rlSteps : sftSteps;
  return {
    ...defaults,
    ...s,
    rlAlgorithm: s.rlAlgorithm === "ppo" ? "ppo" : "grpo",
    rlTrajectory: [0, 1, 2, 3].includes(s.rlTrajectory) ? s.rlTrajectory : 0,
    rlToken:
      Number.isInteger(s.rlToken) && s.rlToken >= 0 && s.rlToken < 13
        ? s.rlToken
        : 8,
    rlReduction: s.rlReduction === "sequence" ? "sequence" : "token",
    rlKl: s.rlKl === "off" ? "off" : "on",
    rlForce: s.rlForce === "on" ? "on" : "off",
    rlEqual: s.rlEqual === "on" ? "on" : "off",
    rlFault: [
      "none",
      "constant_ratio",
      "detach_kl_weight",
      "wrong_clip",
    ].includes(s.rlFault)
      ? s.rlFault
      : "none",
    ep: s.ep === 2 ? 2 : 1,
    etp: s.etp === 2 ? 2 : 1,
    familyToken:
      Number.isInteger(s.familyToken) &&
      s.familyToken >= 0 &&
      s.familyToken < 64
        ? s.familyToken
        : 7,
    familyExpert: [0, 1, 2, 3].includes(s.familyExpert) ? s.familyExpert : 0,
    familyHead: s.familyHead === 1 ? 1 : 0,
    familyPadding: s.familyPadding === 2 ? 2 : 0,
    familyOp: [
      "router",
      "dispatch",
      "combine",
      "aux",
      "bias",
      ...(model.attention === "mla"
        ? ["mla-norm", "mla-expanded", "mla-cache"]
        : []),
    ].includes(s.familyOp)
      ? s.familyOp
      : "router",
    familyFault: [
      "none",
      "wrong_weights",
      "duplicate_dispatch",
      ...(model.shared_experts > 0 ? ["missing_shared"] : []),
      ...(model.attention === "mla"
        ? ["wrong_norm", "wrong_scale", "missing_rope"]
        : []),
    ].includes(s.familyFault)
      ? s.familyFault
      : "none",
    pp: s.pp === 1 ? 1 : 2,
    microbatches:
      Number.isInteger(s.microbatches) &&
      s.microbatches >= 1 &&
      s.microbatches <= 8
        ? s.microbatches
        : 4,
    pipelineMicrobatch:
      Number.isInteger(s.pipelineMicrobatch) &&
      s.pipelineMicrobatch >= 0 &&
      s.pipelineMicrobatch < 8
        ? s.pipelineMicrobatch
        : 0,
    sequenceLayout: s.sequenceLayout === "thd" ? "thd" : "ordinary",
    cp: s.cp === 1 ? 1 : 2,
    sequenceRank: s.cp !== 1 && s.sequenceRank === 1 ? 1 : 0,
    sequenceQuery:
      Number.isInteger(s.sequenceQuery) &&
      s.sequenceQuery >= 0 &&
      s.sequenceQuery < 64
        ? s.sequenceQuery
        : 19,
    sequenceFault: ["none", "local_kv", "leak"].includes(s.sequenceFault)
      ? s.sequenceFault
      : "none",
    tp: s.tp === 1 ? 1 : 2,
    dp: s.dp === 1 ? 1 : 2,
    parallelRank:
      Number.isInteger(s.parallelRank) &&
      s.parallelRank >= 0 &&
      s.parallelRank < (s.tp === 1 ? 1 : 2) * (s.dp === 1 ? 1 : 2)
        ? s.parallelRank
        : 0,
    parallelOp: [
      "qkv",
      "attention_output",
      "ffn_pair",
      "ffn_output",
      "vocab",
    ].includes(s.parallelOp)
      ? s.parallelOp
      : "qkv",
    operator: gqaSteps.some((step) => step.id === s.operator)
      ? s.operator
      : "overview",
    query:
      Number.isInteger(s.query) && s.query >= 0 && s.query < 4 ? s.query : 0,
    gqaHead:
      Number.isInteger(s.gqaHead) && s.gqaHead >= 0 && s.gqaHead < 4
        ? s.gqaHead
        : 0,
    sftLayout: ["single", "unpacked", "packed"].includes(s.sftLayout)
      ? s.sftLayout
      : "single",
    sftLimit:
      Number.isInteger(s.sftLimit) && s.sftLimit >= 2 && s.sftLimit <= 128
        ? s.sftLimit
        : 128,
    sftPadding: s.sftPadding === 8 ? 8 : 1,
    sftQuery:
      Number.isInteger(s.sftQuery) && s.sftQuery >= 0 && s.sftQuery < 256
        ? s.sftQuery
        : 0,
    decoderOp: [
      "input_norm",
      "attention",
      "ffn_norm",
      "gate",
      "up",
      "silu",
      "product",
      "down",
      "residual",
    ].includes(s.decoderOp)
      ? s.decoderOp
      : "ffn_norm",
    decoderToken:
      Number.isInteger(s.decoderToken) &&
      s.decoderToken >= 0 &&
      s.decoderToken < 64
        ? s.decoderToken
        : 7,
    tinyLayer: s.tinyLayer === 1 ? 1 : 0,
    decoderTied: s.decoderTied === "untied" ? "untied" : "tied",
    model: model.id,
    scenario,
    layer: Number.isFinite(s.layer)
      ? Math.max(0, Math.min(model.layers - 1, Math.floor(s.layer)))
      : 0,
    step: steps.includes(s.step) ? s.step : steps[0],
    sourceLane: s.sourceLane === "runtime" ? "runtime" : "reference",
    view: [
      "walkthrough",
      "atlas",
      "sample",
      "course",
      "basics",
      "runtime",
    ].includes(s.view)
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
    query: Number(p.get("query") || 0),
    gqaHead: Number(p.get("gqaHead") || 0),
    sftLimit: Number(p.get("sftLimit") || 128),
    sftPadding: Number(p.get("sftPadding") || 1),
    sftQuery: Number(p.get("sftQuery") || 0),
    decoderToken: Number(p.get("decoderToken") || 7),
    tinyLayer: Number(p.get("tinyLayer") || 0),
    rlTrajectory: Number(p.get("rlTrajectory") ?? 0),
    rlToken: Number(p.get("rlToken") ?? 8),
    ep: Number(p.get("ep") || 1),
    etp: Number(p.get("etp") || 1),
    familyToken: Number(p.get("familyToken") ?? 7),
    familyExpert: Number(p.get("familyExpert") ?? 0),
    familyHead: Number(p.get("familyHead") ?? 0),
    familyPadding: Number(p.get("familyPadding") ?? 0),
    tp: Number(p.get("tp") || 2),
    dp: Number(p.get("dp") || 2),
    parallelRank: Number(p.get("parallelRank") || 0),
    pp: Number(p.get("pp") || 2),
    microbatches: Number(p.get("microbatches") || 4),
    pipelineMicrobatch: Number(p.get("pipelineMicrobatch") || 0),
    cp: Number(p.get("cp") || 2),
    sequenceRank: Number(p.get("sequenceRank") || 0),
    sequenceQuery: Number(p.get("sequenceQuery") || 19),
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
  const model = models.find((m) => m.id === s.model)!;
  if (
    s.step === "decoder" &&
    model.attention === "gqa" &&
    s.operator !== "overview"
  ) {
    const substep = stepFor(s.operator);
    return [
      substep.id === "qknorm" && !model.qk_norm ? "B-Q2" : substep.sourceId,
    ];
  }
  return [
    ...new Set([
      ...(c.pipeline.find((p) => p.id === s.step)?.source_ids || []),
      ...(s.step === "decoder" ? c.attention.source_ids : []),
    ]),
  ];
}

// Stable excerpt IDs identify curated semantic entry points, never array positions.
// An absent match deliberately leaves the reader unselected instead of implying
// that an arbitrary excerpt implements the current step.
export function sourceExcerptId(
  s: State,
  sourceId: string,
): string | undefined {
  const model = models.find((m) => m.id === s.model);
  if (!model) return undefined;
  if (sourceId === "C-SCHEDULE") return "pp-1f1b";
  if (sourceId === "C-TPMAP") return "sp-gather";
  if (sourceId === "C-COREUTIL")
    return s.sequenceLayout === "thd"
      ? "cp-document-zigzag"
      : "cp-sequence-zigzag";
  if (sourceId === "C-TPLINEAR")
    return ["attention_output", "ffn_output"].includes(s.parallelOp)
      ? "tp-row-sum"
      : "tp-column-gather";
  if (sourceId === "C-VOCABCE") return "tp-vocab-ce";
  if (sourceId === "C-FINALGRAD") return "dp-token-normalize";
  if (s.scenario === "sft" && sourceId === "C-BLOCK")
    return "decoder-final-norm";
  if (s.scenario === "sft" && s.step === "decoder" && sourceId === "C-MLP")
    return "decoder-swiglu";
  if (
    s.scenario === "sft" &&
    s.step === "decoder" &&
    s.operator === "overview" &&
    sourceId === "C-LAYER"
  )
    return s.decoderOp === "residual"
      ? "decoder-ffn-residual"
      : s.decoderOp === "ffn_norm"
        ? "decoder-pre-ffn-norm"
        : "gqa-input-norm";
  if (s.scenario === "rl") {
    const routes: Record<string, Record<string, string>> = {
      logprobs: { "R-LOSS": "r-loss-l158", "R-TRAIN": "r-train-l117" },
      advantage: { "R-GRPO": "r-grpo-l1785" },
      policy_update: {
        "R-LOSS": "r-loss-l310",
        "R-TRAIN": "r-train-l117",
        "R-PPOCFG": "r-ppocfg-l16",
      },
      refit: { "R-WORKER": "r-worker-l1187" },
    };
    return routes[s.step]?.[sourceId];
  }
  if (
    s.step === "decoder" &&
    model.attention === "gqa" &&
    s.operator !== "overview"
  ) {
    const substep = stepFor(s.operator);
    if (substep.id === "qknorm" && !model.qk_norm)
      return sourceId === "B-Q2"
        ? "b-q2-l46"
        : sourceId === "C-ATTN"
          ? "c-attn-l1920"
          : undefined;
    if (substep.id === "rope" && sourceId === "C-ATTN") return "gqa-rope-call";
    return sourceId === substep.sourceId ? substep.excerptId : undefined;
  }
  if (s.step === "decoder") {
    const c = cases.find((c) => c.id === s.model)!;
    if (sourceId === "C-MLA" && c.attention.source_ids.includes(sourceId))
      return s.mla === "decode" ? "c-mla-l321" : "c-mla-l638";
    if (sourceId === "C-ATTN" && c.attention.source_ids.includes(sourceId))
      return "c-attn-l1887";
    if (sourceId === "C-MOE" && c.layers[s.layer]?.feed_forward === "moe")
      return "c-moe-l437";
    if (sourceId === "C-GPT") return "c-gpt-l583";
  }
  const routes: Record<string, Record<string, string>> = {
    input: {
      "B-SFTDATA": "b-sftdata-l45",
      "B-DIRECTSFT": "sft-dataset-collate",
      "B-SFTCOLLATE": "sft-collate-shift",
      "B-CONVERSATION": "sft-shift",
      "B-PACK": "sft-pack-boundaries",
      "B-STEP": "b-step-l482",
      "B-LOSS": "b-loss-l62",
    },
    embedding: { "C-GPT": "gpt-embedding" },
    lm_head: { "C-GPT": "gpt-output-projection" },
    loss: {
      "C-GPT": "c-gpt-l790",
      "B-LOSS": "b-loss-l62",
      "B-STEP": "b-step-l482",
    },
    backward: { "B-TRAIN": "b-train-l914" },
    update: { "B-TRAIN": "b-train-l914", "B-PRETRAIN": "b-pretrain-l133" },
  };
  return routes[s.step]?.[sourceId];
}
