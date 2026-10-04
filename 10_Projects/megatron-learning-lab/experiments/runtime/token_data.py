"""Actual tokenizer output mapped to the explicit Qwen IM chat branch.

This adapter never edits a chat template. It validates the rendered turn
boundaries and fast-tokenizer offsets; unsupported layouts fail closed.
"""
from __future__ import annotations
import hashlib
import re

from .contracts import require, validate_input

START = "<|im_start|>"
END = "<|im_end|>"
TURN = re.compile(re.escape(START) + r"(system|user|assistant)\n")


def canonical_chat(tokenizer, document, *, mask_mode, forward_sequence_length,
                   vocab_size):
    require(mask_mode in ("assistant", "last_turn", "full"), "unknown supervision mode")
    require(isinstance(document, dict) and set(document) == {"sample_id", "messages"},
            "expected one sample_id/messages document")
    require(isinstance(document["sample_id"], str) and document["sample_id"], "sample_id missing")
    messages = document["messages"]
    require(isinstance(messages, list) and 2 <= len(messages) <= 32, "invalid conversation length")
    for message in messages:
        require(isinstance(message, dict) and set(message) == {"role", "content"},
                "only plain role/content messages are supported")
        require(message["role"] in ("system", "user", "assistant")
                and isinstance(message["content"], str) and message["content"],
                "unknown role or empty message")
        require(START not in message["content"] and END not in message["content"],
                "literal chat delimiters in message content need a separate mapping")
    require(messages[-1]["role"] == "assistant", "SFT sample must end in an assistant response")
    require(getattr(tokenizer, "is_fast", False), "offset mapping requires a fast tokenizer")
    template = tokenizer.get_chat_template()
    require(isinstance(template, str) and template, "actual chat template missing")
    kwargs = dict(add_generation_prompt=False, enable_thinking=False)
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, **kwargs)
    require(isinstance(rendered, str) and len(rendered.encode("utf-8")) <= 1_048_576,
            "rendered chat exceeds capture limit")
    turns = list(TURN.finditer(rendered))
    require([turn.group(1) for turn in turns] == [m["role"] for m in messages],
            "unknown rendered chat role/header mapping")
    require(rendered.count(START) == len(turns) and rendered.count(END) == len(turns),
            "unknown rendered chat delimiter count")
    spans = []
    for i, turn in enumerate(turns):
        stop = turns[i + 1].start() if i + 1 < len(turns) else len(rendered)
        eos = rendered.find(END, turn.end(), stop)
        require(eos >= turn.end() and rendered[eos + len(END):stop].strip() == "",
                "unknown end-of-turn mapping")
        if turn.group(1) == "assistant":
            spans.append((turn.end(), eos + len(END)))
    require(spans, "assistant spans missing")
    chosen = spans[-1:] if mask_mode == "last_turn" else spans
    encoded = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
    ids = list(encoded["input_ids"])
    offsets = list(encoded["offset_mapping"])
    # Independently ask the real template API to tokenize the complete message.
    template_ids = tokenizer.apply_chat_template(messages, tokenize=True, **kwargs)
    require(ids == list(template_ids), "render and template tokenization differ")
    require(len(ids) == len(offsets) and 2 <= len(ids) <= forward_sequence_length + 1,
            "sample is too long or too short; this smoke profile does not truncate")
    target_mask = []
    for start, end in offsets:
        require(type(start) is int and type(end) is int and 0 <= start < end <= len(rendered),
                "unsupported tokenizer offset")
        if mask_mode == "full":
            target_mask.append(1)
            continue
        intersects = [(a, b) for a, b in chosen if max(a, start) < min(b, end)]
        require(not intersects or any(a <= start and end <= b for a, b in intersects),
                "token crosses a supervision boundary; explicit mapping required")
        target_mask.append(int(bool(intersects)))
    pad = tokenizer.pad_token_id
    if pad is None:
        pad = tokenizer.eos_token_id
    require(type(pad) is int and 0 <= pad < vocab_size, "valid padding/EOS token id required")
    real_length = len(ids)
    padding = forward_sequence_length + 1 - real_length
    ids += [pad] * padding
    target_mask += [0] * padding
    s = len(ids)
    # One canonical extra target slot lets Bridge retain its last prediction.
    data = dict(alignment="next_token", input_ids=[ids],
                labels=[ids[1:] + [-100]], loss_mask=[target_mask[1:] + [0]],
                position_ids=[list(range(s))], document_ids=[[0] * s])
    validate_input(data, {"vocab_size": vocab_size}, "sft")
    return data, dict(sample_id=document["sample_id"], mapping="qwen_im_chat_v1",
                      mask_mode=mask_mode, real_token_count=real_length,
                      padding_token_id=pad, padding_side="right",
                      rendered_text=rendered, target_supervision=target_mask,
                      chat_template=template,
                      chat_template_sha256=hashlib.sha256(template.encode("utf-8")).hexdigest(),
                      template_options=kwargs, source="actual_tokenizer_output")
