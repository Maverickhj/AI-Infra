"""Canonical one-sample data through Bridge's public DatasetProvider API.

Imported only by the guarded configuration probe or authorized Bridge worker.
The official Bridge loader/sampler remains responsible for iteration/resume.
"""
from dataclasses import dataclass
from megatron.bridge.training.config import DatasetProvider
from .contracts import require, validate_input
from .plan import read_document


class RepeatedCanonicalSample:
    def __init__(self, data, samples, vocab_size):
        validate_input(data, {"vocab_size": vocab_size}, "sft")
        require(len(data["input_ids"]) == 1, "one-sample capture dataset required")
        require(type(samples) is int and samples > 0, "invalid official train sample count")
        self.data, self.samples = data, samples

    def __len__(self):
        return self.samples

    def __getitem__(self, index):
        import torch
        require(type(index) is int and 0 <= index < self.samples, "sample index out of range")
        data = self.data
        # Original labels already target the next token. Bridge must not shift.
        return dict(tokens=torch.tensor(data["input_ids"][0][:-1], dtype=torch.long),
                    labels=torch.tensor(data["labels"][0][:-1], dtype=torch.long),
                    loss_mask=torch.tensor(data["loss_mask"][0][:-1], dtype=torch.float32),
                    position_ids=torch.tensor(data["position_ids"][0][:-1], dtype=torch.long))


@dataclass(kw_only=True)
class CanonicalDatasetProvider(DatasetProvider):
    seq_length: int
    canonical_path: str
    vocab_size: int
    skip_getting_attention_mask_from_dataset: bool = True

    def build_datasets(self, context):
        data = read_document(self.canonical_path)
        require(len(data["input_ids"][0]) == self.seq_length + 1, "canonical context target slot mismatch")
        require(context.valid_samples == 0 and context.test_samples == 0,
                "this capture provider has no validation/test dataset")
        return RepeatedCanonicalSample(data, context.train_samples, self.vocab_size), None, None
