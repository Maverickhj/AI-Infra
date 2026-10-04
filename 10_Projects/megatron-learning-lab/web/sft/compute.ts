// Data-mechanism reference only. Does not run a tokenizer or a decoder.
export type Mode = "assistant" | "last_turn" | "full";
export type Message = { role: string; content: string; pieces: string[] };
export type Sample = { id: string; messages: Message[] };
export type Fixture = { vocabulary: string[]; samples: Sample[] };
export type Token = {
  id: number;
  text: string;
  message: number;
  supervised: number;
};
export type Row = {
  sample: string;
  position: number;
  input: Token;
  target: Token;
  label: number;
  mask: number;
  valid: boolean;
};
export type Batch = {
  rows: Row[];
  cuSeqlens: number[];
  cuSeqlensPadded: number[];
  lengths: number[];
  attention: boolean[][];
};
const PAD: Token = { id: 0, text: "<pad>", message: -1, supervised: 0 };
const END: Token = { id: -100, text: "<ignored>", message: -1, supervised: 0 };

export function tokenize(
  f: Fixture,
  sample: Sample,
  mode: Mode,
  maxTokens = 128,
): Token[] {
  if (!["assistant", "last_turn", "full"].includes(mode))
    throw Error("未知监督模式");
  if (!Number.isInteger(maxTokens) || maxTokens < 2 || maxTokens > 4096)
    throw Error("截断长度必须为 2–4096 的整数");
  const last = sample.messages.map((m) => m.role).lastIndexOf("assistant");
  const result: Token[] = [];
  const add = (text: string, message: number, response: boolean) => {
    const id = f.vocabulary.indexOf(text);
    if (id < 0) throw Error("fixture 词表缺少 token");
    result.push({
      id,
      text,
      message,
      supervised: Number(
        mode === "full" ||
          (response && (mode === "assistant" || message === last)),
      ),
    });
  };
  add("<bos>", -1, false);
  sample.messages.forEach((m, i) => {
    if (
      !["user", "assistant", "system"].includes(m.role) ||
      m.pieces.join("") !== m.content
    )
      throw Error("fixture 与原始 messages 不一致");
    add("<" + m.role + ">", i, false);
    m.pieces.forEach((p) => add(p, i, m.role === "assistant"));
    add(m.role === "assistant" ? "<eos>" : "<turn>", i, m.role === "assistant");
  });
  // Right truncation doesn't invent EOS or retarget last_turn to an earlier reply.
  return result.slice(0, maxTokens);
}

export function align(tokens: Token[], sample: string, shift = 1): Row[] {
  if (shift !== 1) throw Error("只允许一次 shift；重复 shift 会跳过紧邻目标");
  if (tokens.length < 2) throw Error("至少需要两个 token");
  return tokens.map((input, i) => {
    const target = tokens[i + 1] || END;
    return {
      sample,
      position: i,
      input,
      target,
      label: target.supervised ? target.id : -100,
      mask: target.supervised,
      valid: true,
    };
  });
}

export function validateAlignment(tokens: Token[], rows: Row[]) {
  if (
    tokens.length !== rows.length ||
    rows.some((r, i) => {
      const target = tokens[i + 1] || END;
      return (
        r.input.id !== tokens[i].id ||
        r.target.id !== target.id ||
        r.mask !== target.supervised ||
        r.label !== (target.supervised ? target.id : -100)
      );
    })
  )
    throw Error("input/target/mask 不满足单次 shift，或发生跨样本预测");
}

export function batch(
  sequences: Row[][],
  padMultiple = 1,
  leak = false,
  padTo = 0,
): Batch {
  if (!sequences.length || sequences.some((s) => !s.length))
    throw Error("每个样本至少需要一对 input/target");
  if (!Number.isInteger(padMultiple) || padMultiple < 1 || padMultiple > 64)
    throw Error("padding 倍数非法");
  if (new Set(sequences.map((s) => s[0].sample)).size !== sequences.length)
    throw Error("batch sample ID 必须唯一");
  if (
    !Number.isInteger(padTo) ||
    padTo < 0 ||
    (padTo > 0 && sequences.some((s) => s.length > padTo))
  )
    throw Error("padding 宽度不能截断样本");
  const rows: Row[] = [],
    cuSeqlens = [0],
    cuSeqlensPadded = [0],
    lengths = sequences.map((s) => s.length);
  sequences.forEach((s) => {
    rows.push(...s);
    const padding = padTo
      ? padTo - s.length
      : (padMultiple - (s.length % padMultiple)) % padMultiple;
    for (let i = 0; i < padding; i++)
      rows.push({
        sample: s[0].sample,
        position: s.length + i,
        input: PAD,
        target: END,
        label: -100,
        mask: 0,
        valid: false,
      });
    cuSeqlens.push(cuSeqlens[cuSeqlens.length - 1] + s.length);
    cuSeqlensPadded.push(rows.length);
  });
  const attention = rows.map((q, i) =>
    rows.map(
      (k, j) => q.valid && k.valid && j <= i && (leak || k.sample === q.sample),
    ),
  );
  return { rows, cuSeqlens, cuSeqlensPadded, lengths, attention };
}

export function unpack(b: Batch): Row[][] {
  return b.cuSeqlensPadded
    .slice(0, -1)
    .map((start, i) =>
      b.rows.slice(start, b.cuSeqlensPadded[i + 1]).filter((r) => r.valid),
    );
}

export function referenceLoss(b: Batch, vocabSize: number) {
  // Toy causal context q=id/10, k=id/10, v=id/7. Not the GQA/decoder model.
  const contexts: number[] = [],
    losses: number[] = [];
  for (let i = 0; i < b.rows.length; i++) {
    const row = b.rows[i];
    if (!row.valid) {
      contexts.push(0);
      losses.push(0);
      continue;
    }
    const allowed = b.rows.map((_, j) => j).filter((j) => b.attention[i][j]);
    if (!allowed.length) throw Error("有效 query 无可见 key");
    const scores = allowed.map(
      (j) => (row.input.id * b.rows[j].input.id) / 100,
    );
    const mx = Math.max(...scores),
      exp = scores.map((s) => Math.exp(s - mx));
    const den = exp.reduce((a, v) => a + v, 0);
    const context = allowed.reduce(
      (a, j, n) => a + ((exp[n] / den) * b.rows[j].input.id) / 7,
      0,
    );
    const logits = Array.from(
      { length: vocabSize },
      (_, v) => Math.cos((v + 1) * (context + 1) + row.position / 3) / 4,
    );
    const top = Math.max(...logits);
    const logZ =
      top + Math.log(logits.reduce((a, v) => a + Math.exp(v - top), 0));
    contexts.push(context);
    losses.push(row.target.id < 0 ? 0 : logZ - logits[row.target.id]);
  }
  const sum = losses.reduce((a, l, i) => a + l * b.rows[i].mask, 0);
  const count = b.rows.reduce((a, r) => a + r.mask, 0);
  return {
    contexts,
    losses,
    sum,
    count,
    mean: count ? sum / count : null,
    status: count ? "reference" : "no_supervision",
  };
}

// Imported data is not proof of official tokenizer execution.
export function parseTokenizerTrace(text: string) {
  if (text.length > 200000) throw Error("trace 超过 200KB 限制");
  const t = JSON.parse(text);
  if (
    !t ||
    t.schema_version !== 1 ||
    t.kind !== "tokenizer_trace" ||
    typeof t.model_id !== "string" ||
    !t.model_id ||
    typeof t.tokenizer_revision !== "string" ||
    !/^[a-f0-9]{40}$/.test(t.tokenizer_revision) ||
    typeof t.template_sha256 !== "string" ||
    !/^[a-f0-9]{64}$/.test(t.template_sha256) ||
    !["assistant", "last_turn", "full"].includes(t.loss_mode) ||
    typeof t.rendered_text !== "string" ||
    !Array.isArray(t.messages) ||
    !t.messages.length ||
    !t.messages.every(
      (m: Record<string, unknown>) =>
        m &&
        ["user", "assistant", "system"].includes(String(m.role)) &&
        typeof m.content === "string",
    ) ||
    !Array.isArray(t.tokens) ||
    t.tokens.length < 2 ||
    t.tokens.length > 4096 ||
    !Array.isArray(t.loss_mask) ||
    t.loss_mask.length !== t.tokens.length ||
    !t.loss_mask.every((m: unknown) => m === 0 || m === 1) ||
    !t.tokens.every(
      (v: Record<string, unknown>) =>
        v &&
        Number.isSafeInteger(v.id) &&
        Number(v.id) >= 0 &&
        typeof v.text === "string",
    )
  )
    throw Error(
      "tokenizer trace 缺少有效 revision/template/messages/token/mask 合约",
    );
  const tokens: Token[] = t.tokens.map((v: Token, i: number) => ({
    id: v.id,
    text: v.text,
    message: -1,
    supervised: t.loss_mask[i],
  }));
  return {
    model: t.model_id as string,
    revision: t.tokenizer_revision as string,
    templateHash: t.template_sha256 as string,
    rendered: t.rendered_text as string,
    messages: t.messages as { role: string; content: string }[],
    mode: t.loss_mode as Mode,
    provenance: "external_unverified" as const,
    rows: align(tokens, "imported"),
  };
}
