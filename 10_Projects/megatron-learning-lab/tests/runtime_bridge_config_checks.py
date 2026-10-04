"""Real Bridge config/API + synthetic CPU data/optimizer contracts; no training."""
import json
from pathlib import Path
import socket
import tempfile
from types import SimpleNamespace
import unittest

def forbidden(*args, **kwargs):
    raise AssertionError("configuration test forbids network, CUDA and distributed initialization")

socket.socket.connect=forbidden
socket.create_connection=forbidden
import torch
blocked_cuda_attempts=[]
def cuda_unavailable(*args,**kwargs):
    blocked_cuda_attempts.append("blocked before initialization")
    # FlashInfer explicitly handles the normal CUDA-unavailable RuntimeError
    # for optional kernels. The real initializer is still never called.
    raise RuntimeError("configuration probe forbids CUDA initialization")
torch.cuda.init=cuda_unavailable
torch.cuda._lazy_init=cuda_unavailable
torch.distributed.init_process_group=forbidden

from transformers import AutoConfig
from experiments.runtime.bridge_entry import build_config, optimizer_parameter
from experiments.runtime.bridge_dataset import CanonicalDatasetProvider
from experiments.runtime.capture import effective_config
from experiments.runtime.adapters import bridge_sft_slice

ROOT=Path(__file__).resolve().parents[1]


class BridgeConfigOnlyChecks(unittest.TestCase):
    def test_complete_qwen_config_uses_real_provider_but_load_hook_is_not_executed(self):
        plan=json.loads((ROOT/"profiles/runtime-plan.example.json").read_text())
        values=json.loads((ROOT/"tests/fixtures/runtime-model-config/qwen3-0.6b.json").read_text())
        config=AutoConfig.for_model(values.pop("model_type"),**values)
        cfg,bridge,events=build_config(plan,hf_config=config)
        self.assertEqual(events,[])
        self.assertEqual(cfg.model.hidden_size,1024)
        self.assertEqual(cfg.model.num_layers,28)
        self.assertEqual(cfg.model.num_attention_heads,16)
        self.assertEqual(cfg.model.kv_channels,128)
        self.assertFalse(cfg.model.perform_initialization)
        self.assertIsNotNone(cfg.model.pre_wrap_hook)
        self.assertTrue(cfg.model.calculate_per_token_loss)
        self.assertEqual(cfg.model.seq_length,128)
        self.assertEqual(cfg.dataset.seq_length,128)
        self.assertEqual(cfg.train.train_iters,2)
        self.assertFalse(cfg.ddp.use_distributed_optimizer)
        self.assertTrue(cfg.checkpoint.save_optim and cfg.checkpoint.save_rng)
        self.assertTrue(cfg.checkpoint.load_optim and cfg.checkpoint.load_rng)
        actual,sha=effective_config(cfg)
        self.assertEqual(actual["optimizer"]["grad_norm_skip_threshold"],{"float_sentinel":"+inf"})
        self.assertGreater(len(actual["model"]),256)
        self.assertEqual(len(sha),64)
        self.assertFalse(torch.cuda.is_initialized())
        plan["training"]["resume_from"]="/not-loaded/checkpoint"
        restored,_,events=build_config(plan,hf_config=config)
        self.assertIsNone(restored.model.pre_wrap_hook)
        self.assertTrue(restored.checkpoint.exit_on_missing_checkpoint)
        self.assertEqual(events,[])

    def test_dataset_provider_uses_official_sample_count_and_once_shifted_labels(self):
        data=dict(alignment="next_token",input_ids=[[1,2,3]],labels=[[2,3,-100]],
                  loss_mask=[[1,1,0]],position_ids=[[0,1,2]],document_ids=[[0,0,0]])
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"input.json";path.write_text(json.dumps(data))
            cfg=CanonicalDatasetProvider(seq_length=2,canonical_path=str(path),vocab_size=8,
                                         dataloader_type="single",num_workers=0)
            dataset,valid,test=cfg.build_datasets(SimpleNamespace(train_samples=3,valid_samples=0,test_samples=0))
            self.assertEqual(len(dataset),3)
            self.assertIsNone(valid);self.assertIsNone(test)
            batch=next(iter(torch.utils.data.DataLoader(dataset,batch_size=1,num_workers=0)))
            logits=torch.arange(16,dtype=torch.float64).reshape(1,2,8).requires_grad_()
            losses=torch.nn.functional.cross_entropy(logits.transpose(1,2),batch["labels"],reduction="none")
            got,measured=bridge_sft_slice(batch,losses,vocab_size=8,mapping="bridge_bsh_v1",output_kind="token_loss")
            self.assertEqual(got,data)
            self.assertAlmostEqual(measured["loss_mean"],losses.mean().item(),places=14)
            self.assertEqual(batch["tokens"].device.type,"cpu")
            losses.mean().backward()
            self.assertGreater(logits.grad.norm().item(),0)
            with self.assertRaises(ValueError):dataset[3]

    def test_bf16_model_rounding_is_not_misreported_as_no_master_update(self):
        parameter=torch.nn.Parameter(torch.tensor([1.],dtype=torch.bfloat16))
        master=torch.nn.Parameter(parameter.detach().float())
        class SyntheticFloat16Optimizer:
            float16_groups=[[parameter]]
            fp32_from_float16_groups=[[master]]
            def get_parameters(self):return [master]
        wrapped=SimpleNamespace(chained_optimizers=[SyntheticFloat16Optimizer()])
        actual=optimizer_parameter(wrapped,parameter)
        self.assertIs(actual,master)
        before=master.detach().clone()
        opt=torch.optim.SGD([master],lr=1e-5)
        parameter.float().square().sum().backward()
        master.grad=parameter.grad.float()
        opt.step()
        with torch.no_grad():parameter.copy_(master)
        self.assertNotEqual(master.item(),before.item())
        self.assertEqual(parameter.item(),1.)
        self.assertEqual(master.grad.item(),2.)

    def test_unknown_and_ambiguous_optimizer_mappings_are_rejected(self):
        parameter=torch.nn.Parameter(torch.tensor([1.]))
        class SyntheticFP32Optimizer:
            def get_parameters(self):return [parameter]
        direct=SyntheticFP32Optimizer()
        self.assertIs(optimizer_parameter(direct,parameter),parameter)
        with self.assertRaisesRegex(ValueError,"ambiguous"):
            optimizer_parameter(SimpleNamespace(chained_optimizers=[direct,direct]),parameter)
        with self.assertRaisesRegex(ValueError,"missing"):
            optimizer_parameter(direct,torch.nn.Parameter(torch.tensor([2.])))


if __name__=="__main__":
    print(json.dumps({"blocked_cuda_initialization_attempts":len(blocked_cuda_attempts),
                      "cuda_initialized":torch.cuda.is_initialized(),
                      "scope":"configuration-only and synthetic CPU contracts"}))
    unittest.main(verbosity=2)
