"""Explicit, bounded tensor mappings for inspected runtime interfaces.

These functions copy selected scalar/tensor slices for evidence. They do not
implement a trainer, change the caller's tensors, or certify a runtime. Imports
are lazy so source/config dry-runs do not import torch.
"""
from __future__ import annotations
from .contracts import MAX_BATCH, MAX_SEQUENCE, require, validate_input


def tensor(value, shape, name):
    import torch
    require(isinstance(value, torch.Tensor), name + " must be a tensor")
    require(tuple(value.shape) == tuple(shape), name + " shape mismatch")
    require(bool(torch.isfinite(value).all()), name + " contains nonfinite values")
    return value


def bounded_shape(ids):
    import torch
    require(isinstance(ids, torch.Tensor) and ids.ndim == 2, "input_ids must be BS")
    b, s = ids.shape
    require(1 <= b <= MAX_BATCH and 2 <= s <= MAX_SEQUENCE, "capture exceeds BS limits")
    require(ids.dtype in (torch.int32, torch.int64), "input_ids must have integer dtype")
    return b, s


def copy_rows(value):
    return value.detach().cpu().tolist()


def hf_sft_slice(output, data, config, *, mapping="hf_next_token_v1"):
    """HF logits are [B,S,V]; canonical labels/mask already refer to target j+1.

    Do not pass shifted labels to HF's built-in loss (which would shift again).
    Compute only selected target logprobs from returned logits; no logits dump.
    """
    import torch
    require(mapping == "hf_next_token_v1", "unknown HF mapping")
    b, s, mask = validate_input(data, config, "sft")
    logits = output.get("logits") if isinstance(output, dict) else getattr(output, "logits", None)
    tensor(logits, (b, s, config["vocab_size"]), "HF logits")
    require(logits.is_floating_point(), "HF logits must be floating point")
    with torch.no_grad():
        values = logits.detach()
        values = values.double() if values.dtype == torch.float64 else values.float()
        labels = torch.tensor(data["labels"], device=values.device, dtype=torch.long)
        weights = torch.tensor(mask, device=values.device, dtype=values.dtype)
        # Unsupervised labels may be -100. Their diagnostic logprob is omitted
        # as zero, rather than inventing a logprob for the ignore index.
        safe = labels.clamp_min(0)
        lp = values.gather(-1, safe.unsqueeze(-1)).squeeze(-1) - values.logsumexp(-1)
        lp = torch.where(labels >= 0, lp, torch.zeros_like(lp))
        total = -(lp * weights).sum(dtype=torch.float64).item()
    count = sum(map(sum, mask))
    return dict(token_logprobs=copy_rows(lp), token_count=count, loss_sum=total, loss_mean=total/count)


def bridge_sft_slice(batch, output, *, vocab_size, mapping, output_kind,
                     document_ids=None):
    """Adapt a normal, unpacked Bridge batch; its labels are ALREADY shifted.

    Append the final target as a context-only canonical slot, with mask=0.
    This preserves the actual last supervised prediction. Nothing is shifted a
    second time. For THD/packing an explicit adapter is required instead.
    """
    import torch
    require(mapping in ("bridge_bsh_v1", "bridge_sbh_v1"), "unknown Bridge mapping")
    require(output_kind in ("token_loss", "logits"), "unknown Bridge output kind")
    require(not any(batch.get(k) is not None for k in ("cu_seqlens", "packed_seq_params")), "packed Bridge adapter unavailable")
    ids = batch.get("tokens")
    b, s = bounded_shape(ids)
    require(s + 1 <= MAX_SEQUENCE, "canonical target slot exceeds capture limit")
    labels = tensor(batch.get("labels"), (b, s), "labels")
    mask = tensor(batch.get("loss_mask"), (b, s), "loss_mask")
    pos = tensor(batch.get("position_ids"), (b, s), "position_ids")
    require(labels.dtype in (torch.int32, torch.int64), "labels must have integer dtype")
    require(pos.dtype in (torch.int32, torch.int64), "position_ids must have integer dtype")
    require(bool(((labels >= 0) & (labels < vocab_size)).all()), "Bridge labels must be real target ids")
    require(bool(((mask == 0) | (mask == 1)).all()), "loss_mask must be binary")
    require(bool((ids[:, 1:] == labels[:, :-1]).all()), "Bridge labels missing/double shift")
    docs = document_ids if document_ids is not None else torch.zeros_like(ids)
    tensor(docs, (b, s), "document_ids")
    # This initial adapter deliberately supports unpacked, reset-free sequences.
    require(bool((docs == 0).all()), "packed documents require explicit adapter")
    require(bool((pos[:, 1:] == pos[:, :-1] + 1).all()), "reset positions require explicit adapter")
    data = dict(alignment="next_token",
                input_ids=copy_rows(torch.cat((ids, labels[:, -1:]), 1)),
                labels=[row+[-100] for row in copy_rows(labels)],
                loss_mask=[row+[0] for row in copy_rows(mask.to(torch.int64))],
                position_ids=copy_rows(torch.cat((pos, pos[:, -1:]+1), 1)),
                document_ids=[[0]*(s+1) for _ in range(b)])
    validate_input(data, {"vocab_size":vocab_size}, "sft")
    if output_kind == "token_loss":
        # GPT with labels returns [B,S] token CE, independent of hidden layout.
        tensor(output, (b, s), "Bridge token_loss")
        require(output.is_floating_point() and bool((output >= 0).all()), "token_loss must be nonnegative floating point")
        lp = -output.detach()
    else:
        raw_shape = (b, s, vocab_size) if mapping == "bridge_bsh_v1" else (s, b, vocab_size)
        tensor(output, raw_shape, "Bridge logits")
        require(output.is_floating_point(), "logits must be floating point")
        raw = output.detach() if mapping == "bridge_bsh_v1" else output.detach().transpose(0, 1)
        raw = raw.double() if raw.dtype == torch.float64 else raw.float()
        lp = raw.gather(-1, labels.long().unsqueeze(-1)).squeeze(-1) - raw.logsumexp(-1)
    total = -(lp.double()*mask).sum().item()
    count = int(mask.sum().item())
    measurements = dict(token_logprobs=[row+[0.0] for row in copy_rows(lp)],
                        token_count=count, loss_sum=total, loss_mean=total/count)
    return data, measurements


def nemo_action_slice(next_token_logprobs, data, *, config, identities, versions,
                      mapping="nemo_full_token_v1"):
    """Map inspected NeMo [B,S-1] current LP + [B,S] batch to action positions.

    No PG/GAE is evaluated here. Collected advantages/logprobs remain the actual
    upstream values. Names and alignment are selected explicitly, never guessed.
    """
    import torch
    require(mapping == "nemo_full_token_v1", "unknown NeMo mapping")
    ids = data.get("input_ids")
    b, s = bounded_shape(ids)
    tensor(next_token_logprobs, (b, s-1), "current next-token logprobs")
    mask = tensor(data.get("token_mask"), (b, s), "token_mask")
    sample = tensor(data.get("sample_mask"), (b,), "sample_mask")
    require(bool(((mask == 0) | (mask == 1)).all()) and bool((sample == 1).all()), "unknown filtered/masked batch mapping")
    canonical = dict(input_ids=copy_rows(ids), response_mask=copy_rows(mask.to(torch.int64)),
                     sample_mask=copy_rows(sample.to(torch.int64)), alignment="action_position",
                     policy_versions=dict(versions), **identities)
    validate_input(canonical, config, "rl")
    current = torch.cat((torch.zeros_like(next_token_logprobs[:, :1]), next_token_logprobs.detach()), 1)
    values = dict(current_logprobs=copy_rows(current))
    for source, target in (("generation_logprobs","generation_logprobs"),
                           ("advantages","advantages")):
        values[target] = copy_rows(tensor(data.get(source), (b,s), source))
    if config.get("force_on_policy"):
        # Actual loss branch uses current.detach(), not an absent previous field.
        values["previous_logprobs"] = copy_rows(current)
    else:
        values["previous_logprobs"] = copy_rows(tensor(data.get("prev_logprobs"),(b,s),"prev_logprobs"))
    values["reference_logprobs"] = (
        copy_rows(tensor(data.get("reference_policy_logprobs"), (b,s),"reference_policy_logprobs"))
        if config.get("kl_beta") else None)
    if config.get("algorithm") == "ppo":
        values["old_values"] = copy_rows(tensor(data.get("values"), (b,s),"old values"))
        values["returns"] = copy_rows(tensor(data.get("returns"), (b,s),"returns"))
    else:
        require(config.get("algorithm") == "grpo", "unknown RL algorithm")
    # Zero only the canonical dummy slot. It has no upstream current prediction.
    # Do not zero masked gaps: GAE returns and other diagnostics may be nonzero.
    for key in ("generation_logprobs","previous_logprobs","reference_logprobs","advantages"):
        if values.get(key) is not None:
            for row in values[key]: row[0] = 0.0
    return canonical, values


class LossTap:
    """Composition hook: call the real loss once and return its exact objects.

    The sink receives detached *selected* slices via the mapping above. A sink
    failure stops the run rather than silently dropping required evidence.
    """
    def __init__(self, delegate, sink):
        self.delegate, self.sink = delegate, sink

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "delegate"), name)

    def __call__(self, *args, **kwargs):
        result = self.delegate(*args, **kwargs)
        self.sink(args, kwargs, result)
        return result
