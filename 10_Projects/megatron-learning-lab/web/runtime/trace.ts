// Bounded JSON data only. No eval, imports, shell execution or URL loading.
export type RuntimeTrace = Record<string, any>;
export type ValidTrace = {
  trace: RuntimeTrace;
  config: Record<string, any>;
  data: Record<string, any>;
  token_count: number | null;
  trust: "imported_claim";
};
function check(ok: unknown, message: string): asserts ok {
  if (!ok) throw Error(message);
}
const num = (v: unknown): v is number =>
  typeof v === "number" && Number.isFinite(v);
const int = (v: unknown): v is number => num(v) && Number.isInteger(v);
const obj = (v: any) =>
  v !== null && typeof v === "object" && !Array.isArray(v);
const sum = (a: number[]) => a.reduce((s, v) => s + v, 0);
const hex = (v: any, n: number) =>
  typeof v === "string" && new RegExp("^[a-f0-9]{" + n + "}$").test(v);
const close = (a: any, b: number, name: string) =>
  check(
    num(a) && Math.abs(a - b) <= 1e-7 + 1e-6 * Math.abs(b),
    name + " semantic mismatch",
  );
const adapters = [
  "authored_cpu_v1",
  "hf_next_token_v1",
  "bridge_bsh_v1",
  "bridge_sbh_v1",
  "nemo_full_token_v1",
];
export function parseTraceJSON(text: string): any {
  check(
    typeof text === "string" &&
      new TextEncoder().encode(text).length <= 1048576,
    "trace exceeds 1 MiB",
  );
  let pos = 0;
  const space = () => {
    while (/[ \t\r\n]/.test(text[pos] ?? "") && pos < text.length) pos++;
  };
  function string(): string {
    const start = pos++;
    while (pos < text.length) {
      if (text[pos] === "\\") {
        pos += 2;
        continue;
      }
      if (text[pos++] === '"') return JSON.parse(text.slice(start, pos));
    }
    throw Error("unterminated JSON string");
  }
  function value(depth: number): any {
    check(depth <= 32, "JSON nesting exceeds limit");
    space();
    if (text[pos] === '"') return string();
    if (text[pos] === "{") {
      pos++;
      space();
      const result: Record<string, any> = {};
      let count = 0;
      if (text[pos] === "}") {
        pos++;
        return result;
      }
      while (true) {
        space();
        check(text[pos] === '"', "invalid JSON key");
        const key = string();
        check(
          !["__proto__", "prototype", "constructor"].includes(key),
          "unsafe JSON key",
        );
        check(!Object.hasOwn(result, key), "duplicate JSON key: " + key);
        check(++count <= 256, "object exceeds limit");
        space();
        check(text[pos++] === ":", "invalid JSON object");
        result[key] = value(depth + 1);
        space();
        const sep = text[pos++];
        if (sep === "}") return result;
        check(sep === ",", "invalid JSON object");
      }
    }
    if (text[pos] === "[") {
      pos++;
      space();
      const result: any[] = [];
      if (text[pos] === "]") {
        pos++;
        return result;
      }
      while (true) {
        result.push(value(depth + 1));
        check(result.length <= 65536, "array exceeds limit");
        space();
        const sep = text[pos++];
        if (sep === "]") return result;
        check(sep === ",", "invalid JSON array");
      }
    }
    const tail = text.slice(pos);
    for (const [token, v] of [
      ["true", true],
      ["false", false],
      ["null", null],
    ] as const)
      if (tail.startsWith(token)) {
        pos += token.length;
        return v;
      }
    const match = tail.match(/^-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?/);
    check(match, "invalid JSON value");
    pos += match[0].length;
    const n = Number(match[0]);
    check(Number.isFinite(n), "nonfinite numeric value");
    return n;
  }
  const parsed = value(0);
  space();
  check(pos === text.length, "unexpected JSON suffix");
  return parsed;
}
export async function hashText(text: string) {
  const digest = await crypto.subtle.digest(
    "SHA-256",
    new TextEncoder().encode(text),
  );
  return [...new Uint8Array(digest)]
    .map((v) => v.toString(16).padStart(2, "0"))
    .join("");
}
function matrix(
  v: any,
  b: number,
  s: number,
  name: string,
  predicate: (x: any) => boolean = num,
): number[][] {
  check(Array.isArray(v) && v.length === b, name + " batch shape mismatch");
  v.forEach((row) =>
    check(
      Array.isArray(row) && row.length === s && row.every(predicate),
      name + " sequence shape/element mismatch",
    ),
  );
  return v;
}
function identity(m: any, p: string) {
  check(obj(m), "manifest missing");
  check(adapters.includes(m.adapter), "unknown adapter/version mapping");
  check(
    ["float64", "float32", "bfloat16", "float16"].includes(m.dtype),
    "unknown dtype",
  );
  check(m.layout === "BS", "trace must use canonical BS layout");
  check(typeof m.backend === "string" && m.backend, "backend missing");
  for (const k of ["model", "tokenizer"]) {
    check(
      obj(m[k]) && typeof m[k].id === "string" && m[k].id,
      "identity missing",
    );
    check(
      typeof m[k].revision === "string" && m[k].revision,
      "revision missing",
    );
    if (p.startsWith("observed") || m.evidence_kind === "hf_runtime")
      check(
        /^[a-f0-9]{40,64}$/.test(m[k].revision),
        "immutable revision required",
      );
  }
  check(
    hex(m.tokenizer.chat_template_sha256, 64),
    "chat template hash missing",
  );
  check(
    [
      "authored",
      "hf_checkpoint",
      "training_checkpoint",
      "random_initialized",
    ].includes(m.model.weights_origin),
    "weights origin missing",
  );
  check(
    ["authored_scaled", "hf_config"].includes(m.model.architecture_origin),
    "architecture origin missing",
  );
  const g = m.parallel;
  check(obj(g), "parallel groups missing");
  for (const k of ["tp", "pp", "dp", "cp", "ep", "world_size"])
    check(int(g[k]) && g[k] > 0, "invalid parallel dimension");
  check(
    g.tp * g.pp * g.dp * g.cp === g.world_size && g.world_size % g.ep === 0,
    "parallel world size mismatch",
  );
  check(
    ["runtime", "in_memory_reference"].includes(g.groups_origin),
    "group origin missing",
  );
  check(obj(m.software) && m.software.python, "software versions missing");
  check(
    Array.isArray(m.runtime_sources) && m.runtime_sources.length,
    "runtime source mapping missing",
  );
  for (const x of m.runtime_sources) {
    check(
      obj(x) &&
        ["component", "path", "symbol"].every(
          (k) => typeof x[k] === "string" && x[k],
        ),
      "invalid runtime source",
    );
    check(hex(x.sha256, 64), "runtime source hash missing");
  }
  check(
    obj(m.capture) &&
      m.capture.timing === "not_measured" &&
      m.capture.scope === "selected_slices" &&
      Array.isArray(m.capture.unobserved),
    "capture boundary missing",
  );
  const e = m.execution;
  check(obj(e), "execution metadata missing");
  if (p === "derived")
    check(
      e.status === "not_run" && m.evidence_kind === "formula",
      "derived cannot claim execution",
    );
  else if (p === "reference") {
    check(
      ["authored_cpu", "synthetic_contract", "hf_runtime"].includes(
        m.evidence_kind,
      ),
      "wrong reference provenance",
    );
    check(
      e.status ===
        (m.evidence_kind === "synthetic_contract" ? "synthetic" : "executed"),
      "reference execution mismatch",
    );
  } else {
    const bridge = p === "observed_bridge";
    check(
      m.backend === (bridge ? "megatron-bridge" : "nemo-rl") &&
        m.evidence_kind === (bridge ? "bridge_runtime" : "rl_runtime") &&
        m.adapter.startsWith(bridge ? "bridge_" : "nemo_"),
      "observed provenance/backend/adapter mismatch",
    );
    check(
      e.status === "executed" &&
        e.synthetic === false &&
        g.groups_origin === "runtime",
      "synthetic/unexecuted data cannot be observed",
    );
    check(
      ["hf_checkpoint", "training_checkpoint"].includes(m.model.weights_origin),
      "observed requires real checkpoint",
    );
    check(
      Array.isArray(e.command) &&
        e.command.length &&
        e.started_at &&
        e.completed_at,
      "actual argv/timestamps missing",
    );
  }
  check(Array.isArray(m.limitations), "limitations missing");
}
export async function validateTrace(text: string): Promise<ValidTrace> {
  const t = parseTraceJSON(text);
  check(
    obj(t) &&
      t.schema === "megatron-learning-lab.trace" &&
      t.schema_version === 1,
    "unknown trace schema/version",
  );
  check(
    typeof t.run_id === "string" &&
      t.run_id.length > 0 &&
      t.run_id.length <= 128,
    "run id missing",
  );
  check(["sft", "rl"].includes(t.task), "unsupported trace task");
  const p = t.provenance;
  check(
    ["derived", "reference", "observed_bridge", "observed_rl"].includes(p),
    "unknown provenance",
  );
  check(
    (p !== "observed_bridge" || t.task === "sft") &&
      (p !== "observed_rl" || t.task === "rl"),
    "provenance/task mismatch",
  );
  identity(t.manifest, p);
  const config = parseTraceJSON(t.config_json),
    data = parseTraceJSON(t.input_json);
  check(obj(config) && obj(data), "config/input must encode objects");
  check(
    (await hashText(t.config_json)) === t.config_sha256,
    "config SHA256 mismatch",
  );
  check(
    (await hashText(t.input_json)) === t.input_sha256,
    "input SHA256 mismatch",
  );
  const ids = data.input_ids;
  check(
    Array.isArray(ids) && ids.length >= 1 && ids.length <= 8,
    "invalid input batch",
  );
  check(
    Array.isArray(ids[0]) && ids[0].length >= 2 && ids[0].length <= 512,
    "invalid input length",
  );
  const b = ids.length,
    s = ids[0].length,
    vocab = config.vocab_size;
  check(int(vocab) && vocab > 1, "vocab size missing");
  matrix(ids, b, s, "input_ids", (v) => int(v) && v >= 0 && v < vocab);
  const mask = matrix(
      data[t.task === "sft" ? "loss_mask" : "response_mask"],
      b,
      s,
      "mask",
      (v) => int(v) && (v === 0 || v === 1),
    ),
    count = sum(mask.map(sum));
  check(count > 0, "global mask is empty");
  if (t.task === "sft") {
    check(data.alignment === "next_token", "unknown SFT shift/mapping");
    const labels = matrix(
      data.labels,
      b,
      s,
      "labels",
      (v) => int(v) && (v === -100 || (v >= 0 && v < vocab)),
    );
    matrix(data.position_ids, b, s, "position_ids", (v) => int(v) && v >= 0);
    const docs = matrix(
      data.document_ids,
      b,
      s,
      "document_ids",
      (v) => int(v) && v >= -1,
    );
    for (let i = 0; i < b; i++) {
      check(mask[i][s - 1] === 0, "last input has no next-token target");
      for (let j = 0; j < s - 1; j++)
        if (mask[i][j]) {
          check(
            labels[i][j] === ids[i][j + 1],
            "double/missing shift or wrong label",
          );
          check(
            docs[i][j] === docs[i][j + 1] && docs[i][j] >= 0,
            "cross-document target leakage",
          );
        }
    }
  } else {
    check(
      data.alignment === "action_position" && mask.every((row) => row[0] === 0),
      "RL action-position alignment mismatch",
    );
    for (const k of ["trajectory_ids", "prompt_ids", "group_ids"])
      check(
        Array.isArray(data[k]) &&
          data[k].length === b &&
          data[k].every((x: any) => typeof x === "string" && x),
        "trajectory identity missing",
      );
    check(new Set(data.trajectory_ids).size === b, "duplicate trajectory id");
    check(
      mask.every((row) => sum(row) > 0),
      "empty RL response",
    );
    check(
      Array.isArray(data.sample_mask) &&
        data.sample_mask.length === b &&
        data.sample_mask.every((v: any) => v === 1),
      "unsupported sample mask",
    );
    const v = data.policy_versions;
    check(
      obj(v) &&
        ["generation", "previous", "current", "after"].every(
          (k) => int(v[k]) && v[k] >= 0,
        ),
      "policy versions missing",
    );
    check(v.after === v.current + 1, "one-step policy version mismatch");
  }
  const m = t.measurements;
  if (p === "derived") {
    check(m === null, "derived traces cannot carry measured numbers");
    return {
      trace: t,
      config,
      data,
      token_count: null,
      trust: "imported_claim",
    };
  }
  check(obj(m), "measurements missing");
  const lpOK = (v: any) => num(v) && v <= 1e-6;
  if (t.task === "sft") {
    const lp = matrix(m.token_logprobs, b, s, "token_logprobs", lpOK),
      total = sum(lp.map((row, i) => sum(row.map((v, j) => -v * mask[i][j]))));
    check(
      int(m.token_count) && m.token_count === count,
      "loss token count mismatch",
    );
    close(m.loss_sum, total, "loss sum");
    close(m.loss_mean, total / count, "loss reduction");
  } else {
    check(
      ["grpo", "ppo"].includes(config.algorithm),
      "unsupported RL algorithm",
    );
    check(
      config.loss_variant === "clipped_pg_k3" &&
        config.offpolicy_correction === false,
      "unknown RL loss mapping",
    );
    check(
      ["token", "sequence"].includes(config.reduction),
      "unknown RL reduction",
    );
    const c = config.ratio_clip,
      beta = config.kl_beta;
    check(
      num(c) && c > 0 && c < 1 && num(beta) && beta >= 0,
      "invalid RL loss parameters",
    );
    const prev = matrix(m.previous_logprobs, b, s, "previous_logprobs", lpOK),
      curr = matrix(m.current_logprobs, b, s, "current_logprobs", lpOK);
    matrix(m.generation_logprobs, b, s, "generation_logprobs", lpOK);
    const adv = matrix(m.advantages, b, s, "advantages"),
      ratio = matrix(m.ratio, b, s, "ratio"),
      pg = matrix(m.pg_token, b, s, "pg_token");
    let ref: number[][] = [];
    if (beta)
      ref = matrix(m.reference_logprobs, b, s, "reference_logprobs", lpOK);
    else
      check(
        m.reference_logprobs === null,
        "disabled reference must be explicitly absent",
      );
    check(
      typeof config.force_on_policy === "boolean",
      "force-on-policy flag missing",
    );
    if (config.force_on_policy) {
      const v = data.policy_versions;
      check(
        v.generation === v.previous && v.previous === v.current,
        "force-on-policy batch version mismatch",
      );
    }
    for (let i = 0; i < b; i++)
      for (let j = 0; j < s; j++) {
        check(
          Math.abs(curr[i][j] - prev[i][j]) <= 50,
          "logprob ratio overflow/invalid mapping",
        );
        const r = config.force_on_policy
          ? 1
          : Math.exp(curr[i][j] - prev[i][j]);
        close(ratio[i][j], r, "policy ratio");
        close(
          pg[i][j],
          Math.max(
            -adv[i][j] * r,
            -adv[i][j] * Math.min(1 + c, Math.max(1 - c, r)),
          ),
          "PG clipping",
        );
        if (!mask[i][j]) close(adv[i][j], 0, "masked advantage");
      }
    const reduce = (rows: number[][]) => {
      const values = rows.map((row, i) =>
        sum(row.map((v, j) => v * mask[i][j])),
      );
      return config.reduction === "token"
        ? sum(values) / count
        : sum(values.map((v, i) => v / sum(mask[i]))) / b;
    };
    const actor = reduce(pg);
    close(m.actor_loss, actor, "actor reduction");
    check(num(m.kl_loss) && m.kl_loss >= 0, "invalid KL loss");
    if (!beta) close(m.kl_loss, 0, "disabled KL");
    else {
      const bound = config.kl_input_clamp,
        cap = config.kl_output_clamp;
      check(
        num(bound) && bound > 0 && bound <= 50 && num(cap) && cap > 0,
        "KL clamp configuration missing",
      );
      check(
        config.kl_sampling === "non_is_score_gradient",
        "unknown KL sampling mapping",
      );
      const kl = curr.map((row, i) =>
        row.map((v, j) => {
          const d = Math.min(bound, Math.max(-bound, ref[i][j] - v));
          return Math.min(cap, Math.exp(d) - 1 - d);
        }),
      );
      close(m.kl_loss, beta * reduce(kl), "KL penalty");
    }
    close(m.policy_loss, actor + m.kl_loss, "policy loss");
    if (config.algorithm === "ppo") {
      const values = matrix(m.values, b, s, "values"),
        old = matrix(m.old_values, b, s, "old_values"),
        ret = matrix(m.returns, b, s, "returns"),
        c = config.value_clip;
      check(num(c) && c > 0, "value clip missing");
      const total = sum(
        values.map((row, i) =>
          sum(
            row.map((v, j) => {
              const clip = Math.min(old[i][j] + c, Math.max(old[i][j] - c, v));
              return (
                0.5 *
                mask[i][j] *
                Math.max((v - ret[i][j]) ** 2, (clip - ret[i][j]) ** 2)
              );
            }),
          ),
        ),
      );
      close(m.value_loss, total / count, "value loss");
    } else
      check(
        m.values === null && m.returns === null && m.value_loss === null,
        "GRPO must not claim critic measurements",
      );
    const r = m.refit;
    check(
      obj(r) && ["not_run", "acknowledged", "synchronized"].includes(r.status),
      "refit state missing",
    );
    if (r.status === "acknowledged") {
      check(
        r.completed === true &&
          r.generation_version === data.policy_versions.after,
        "refit version/ack mismatch",
      );
      check(
        r.evidence === "official_delegate_return" &&
          r.weight_hash_verified === false &&
          !Object.hasOwn(r, "export_hash") &&
          !Object.hasOwn(r, "ack_hash") &&
          int(r.event) &&
          r.event >= 0,
        "refit call acknowledgment cannot claim verified weight hashes",
      );
    }
    if (r.status === "synchronized") {
      check(
        r.completed === true &&
          r.generation_version === data.policy_versions.after,
        "refit version/ack mismatch",
      );
      check(
        hex(r.export_hash, 64) && r.export_hash === r.ack_hash,
        "refit weight acknowledgment mismatch",
      );
    }
  }
  return {
    trace: t,
    config,
    data,
    token_count: count,
    trust: "imported_claim",
  };
}
export function compareTraces(left: ValidTrace, right: ValidTrace) {
  const a = left.trace,
    b = right.trace;
  check(a.task === b.task, "不同任务不能逐 token 比较");
  check(a.input_sha256 === b.input_sha256, "输入、mask 或版本不一致");
  for (const k of ["model", "tokenizer"])
    check(
      a.manifest[k].id === b.manifest[k].id &&
        a.manifest[k].revision === b.manifest[k].revision,
      k + " 身份或 revision 不一致",
    );
  check(
    a.manifest.tokenizer.chat_template_sha256 ===
      b.manifest.tokenizer.chat_template_sha256,
    "chat template 不一致",
  );
  check(
    a.provenance !== "derived" && b.provenance !== "derived",
    "derived 没有实测数值可作差",
  );
  const field = a.task === "sft" ? "token_logprobs" : "current_logprobs";
  const x = a.measurements[field] as number[][],
    y = b.measurements[field] as number[][];
  return {
    field,
    max_abs_difference: Math.max(
      ...x.flat().map((v, i) => Math.abs(v - y.flat()[i])),
    ),
    evidence: [a.provenance, b.provenance],
    claim: "仅比较导入文件中的数值，不证明运行身份或跨引擎等价",
  };
}
