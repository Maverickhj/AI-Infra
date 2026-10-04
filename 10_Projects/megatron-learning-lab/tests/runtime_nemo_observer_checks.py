"""End-to-end observer contracts with real CPU loss/backward and synthetic engines.

The launcher, rollout/reward, model and worker orchestration are synthetic.
Pinned official loss bodies execute in Torch; nothing imports NeMo/Ray or CUDA.
"""
import copy
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from experiments.runtime.capture import callable_source, effective_config
from experiments.runtime.contracts import read_trace
from experiments.runtime.nemo_config import validate_bound_config
from experiments.runtime.nemo_observer import NeMoObserver
from experiments.runtime.nemo_worker_observation import TrainObservationMixin
from tests import runtime_nemo_capture_checks as loss_checks
from tests import test_runtime_nemo_plan as plan_checks
from tests import test_runtime_nemo_entry as entry_checks

ROOT=Path(__file__).resolve().parents[1]
COEFFICIENTS=torch.tensor([[-1.,-2.],[-.5,-1.]],dtype=torch.float64)
IDS=torch.tensor([[1,2,3],[1,4,0]])
MASK=torch.tensor([[0.,1.,1.],[0.,1.,0.]],dtype=torch.float64)


class SyntheticCPUWorker:
    def __init__(self,role):
        self.role=role
        self.model=torch.nn.Linear(1,1,bias=False,dtype=torch.float64)
        with torch.no_grad():self.model.weight.fill_(1. if role=="actor" else .2)
        self.optimizer=loss_checks.SyntheticOptimizer(self.model)
        self.calls=0

    def train(self,data,loss_fn,**kwargs):
        self.calls+=1
        self.optimizer.sgd.zero_grad()
        total=torch.zeros((),dtype=torch.float64)
        for i in range(2):
            micro={k:v[i:i+1] for k,v in data.items()}
            output=(COEFFICIENTS[i:i+1]*self.model.weight[0,0] if self.role=="actor"
                    else torch.ones((1,3,1),dtype=torch.float64)*self.model.weight[0,0])
            loss,_=loss_fn(output,micro,torch.tensor(2.),torch.tensor(3.))
            total+=loss.detach()
            loss.backward()
        norm=self.model.weight.grad.norm().detach().clone()
        self.optimizer.sgd.step()
        self.last_result={"loss":total.view(1),"grad_norm":norm.view(1)}
        return self.last_result


class ObservedSyntheticCPUWorker(TrainObservationMixin,SyntheticCPUWorker):
    pass


class SyntheticPolicy:
    def __init__(self):
        self.worker=ObservedSyntheticCPUWorker("actor")
        self.logprob_error=0.

    def train(self,data,loss_fn,**kwargs):
        return self.worker.train(data,loss_fn,**kwargs)

    def prepare_for_lp_inference(self):pass
    def offload_after_refit(self):pass

    def get_logprobs(self,data):
        out=torch.cat((torch.zeros((2,1),dtype=torch.float64),
                       COEFFICIENTS*self.worker.model.weight.detach()[0,0]),1)
        out[:,1:]+=self.logprob_error
        return {"logprobs":out}


class SyntheticCritic:
    def __init__(self):
        self.worker=SyntheticCPUWorker("critic")

    def train(self,data,loss_fn,**kwargs):
        return self.worker.train(data,loss_fn,**kwargs)

    def finish_training(self):pass


class SyntheticGeneration:
    def __init__(self):
        self.weight=None;self.calls=0
        self.refits=0;self.fail_refit=False

    def generate(self,data,*,greedy=False):
        self.calls+=1
        self.last=dict(output_ids=IDS.clone(),generation_lengths=torch.tensor([2,1]),
            unpadded_sequence_lengths=torch.tensor([3,2]),
            logprobs=torch.cat((torch.zeros((2,1),dtype=torch.float64),COEFFICIENTS*self.weight),1))
        self.last["logprobs"][1,2]=0.
        return self.last

    def prepare_for_generation(self):pass
    def finish_generation(self):pass


class SyntheticExtensions:
    def open(self):self.closed=False
    def close(self):self.closed=True


class NeMoObserverCPUChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        loss_checks.NeMoCaptureCPUChecks.setUpClass.__func__(cls)

    def setUp(self):
        self.profile=plan_checks.NeMoBoundPlanTests()
        self.profile.setUp();self.addCleanup(self.profile.doCleanups)
        self.unwrap=patch("experiments.runtime.nemo_worker_observation.unwrap_training_model",lambda model:model)
        self.unwrap.start();self.addCleanup(self.unwrap.stop)

    def scenario(self,algorithm):
        plan,config=self.profile.bound(algorithm)
        plan["capture"].update(parameter_name="weight",parameter_index=[0,0])
        folder=Path(plan["execution"]["output_path"]);folder.mkdir()
        mapping=validate_bound_config(config,plan)
        policy,generation=SyntheticPolicy(),SyntheticGeneration()
        critic=SyntheticCritic() if algorithm=="ppo" else None
        actor_loss=self.module.ClippedPGLossFn(self.module.ClippedPGLossConfig(
            token_level_loss=True,reference_policy_kl_penalty=mapping["loss"]["kl_beta"]))
        value_loss=self.module.MseValueLossFn(self.module.MseValueLossConfig(scale=1.,cliprange=.2))
        module=ModuleType("synthetic_observed_"+algorithm)
        self.addCleanup(sys.modules.pop,module.__name__,None);sys.modules[module.__name__]=module
        exec(f"def {algorithm}_train():\n pass\n",module.__dict__)
        def refit(p,g,colocated,**kwargs):
            if g.fail_refit:raise RuntimeError("synthetic failed refit")
            g.weight=float(p.worker.model.weight.detach().item());g.refits+=1
        def rollout(*,policy_generation,**kwargs):
            inputs=dict(input_ids=torch.tensor([[1],[1]]),input_lengths=torch.tensor([1,1]))
            result=policy_generation.generate(inputs,greedy=False)
            self.assertIs(result,policy_generation.last)
            batch=dict(total_reward=torch.tensor([1.,0.],dtype=torch.float64),
                       message_log=[[{"role":"user","token_ids":torch.tensor([1])},
                         {"role":"assistant","token_ids":row[1:total]}]
                         for row,total in zip(result["output_ids"],result["unpadded_sequence_lengths"])])
            module.last_rollout=(batch,{"synthetic":True})
            return module.last_rollout
        module.refit_policy_generation,module.run_multi_turn_rollout=refit,rollout
        fixture=json.loads((ROOT/"content/fixtures/runtime-reference.json").read_text())["traces"][1]["manifest"]
        def metadata(plan,tokenizer):
            base=copy.deepcopy(fixture)
            base.update(evidence_kind="synthetic_contract",backend="synthetic-cpu",
                        dtype="float64",adapter="nemo_full_token_v1")
            base["execution"]={"status":"synthetic","synthetic":True}
            base["model"].update(id="synthetic-single-parameter-logprob",revision="synthetic-v1")
            base["runtime_sources"]=[callable_source(SyntheticPolicy.train,"synthetic_policy")]
            base["limitations"]=["synthetic engines/reward; actual extracted loss and CPU SGD only"]
            return base,16
        observer=NeMoObserver(plan,folder,mapping,extensions=SyntheticExtensions(),
                              manifest_builder=metadata,batch_factory=dict)
        _,sha=effective_config(config);observer.configure(config,sha,object())
        self.addCleanup(observer.close)
        master=entry_checks.SyntheticMaster(**copy.deepcopy(config))
        if algorithm=="grpo":
            setup=(policy,generation,None,object(),object(),None,actor_loss,None,None,{},master,{}, {})
        else:
            setup=(policy,generation,critic,object(),object(),None,actor_loss,value_loss,None,None,{},master)
        launcher=SimpleNamespace(**{algorithm+"_train":getattr(module,algorithm+"_train")})
        self.assertIs(observer.attach(setup,launcher),setup)
        return observer,module,policy,generation,critic,actor_loss,value_loss

    def iterate(self,scenario):
        observer,module,policy,generation,critic,actor_loss,value_loss=scenario
        module.refit_policy_generation(policy,generation,True)
        original=module.run_multi_turn_rollout(policy_generation=generation)
        self.assertIs(original,module.last_rollout)
        previous=policy.get_logprobs({})["logprobs"]
        batch=dict(input_ids=IDS.clone(),token_mask=MASK.clone(),
            sample_mask=torch.ones(2,dtype=torch.float64),
            generation_logprobs=generation.last["logprobs"].clone(),
            prev_logprobs=previous.clone(),
            reference_policy_logprobs=torch.cat((torch.zeros((2,1),dtype=torch.float64),COEFFICIENTS),1),
            advantages=torch.tensor([[0.,1.,1.],[0.,-1.,0.]],dtype=torch.float64))
        if critic is not None:
            batch.update(rewards=original[0]["total_reward"],
                values=torch.ones((2,3),dtype=torch.float64)*critic.worker.model.weight.detach()[0,0],
                returns=torch.tensor([[0.,1.,-.2],[0.,.8,0.]],dtype=torch.float64))
            returned=critic.train(batch,value_loss)
            self.assertIs(returned,critic.worker.last_result)
        returned=policy.train(batch,actor_loss)
        self.assertIs(returned,policy.worker.last_result)

    def test_grpo_and_ppo_delegate_two_updates_and_verify_final_refit_probabilities(self):
        for algorithm in ("grpo","ppo"):
            with self.subTest(algorithm=algorithm):
                scenario=self.scenario(algorithm)
                observer,module,policy,generation,critic,*_=scenario
                self.iterate(scenario);self.iterate(scenario)
                result=observer.finish()
                self.assertEqual(result["status"],"synthetic_nemo_contract")
                self.assertEqual(result["iterations"],2)
                self.assertEqual(generation.calls,4)
                self.assertEqual(generation.refits,3)
                self.assertEqual(result["fixed_input_verification"]["max_abs_difference"],0)
                self.assertEqual(policy.worker.calls,2)
                if critic:self.assertEqual(critic.worker.calls,2)
                for record in result["traces"]:
                    trace=json.loads(Path(record["path"]).read_text())
                    self.assertEqual(read_trace(json.dumps(trace))["trust"],"imported_claim")
                    self.assertEqual(trace["provenance"],"reference")
                    self.assertEqual(trace["manifest"]["evidence_kind"],"synthetic_contract")
                    self.assertEqual(trace["measurements"]["refit"]["status"],"acknowledged")
                    self.assertFalse(trace["measurements"]["refit"]["weight_hash_verified"])
                observer.close()
                self.assertNotIn("train",policy.__dict__)
                self.assertNotIn("generate",generation.__dict__)
        self.assertFalse(torch.cuda.is_initialized())

    def test_failed_refit_never_advances_generation_version_or_writes_completion(self):
        scenario=self.scenario("grpo")
        observer,module,policy,generation,*_=scenario
        self.iterate(scenario);self.iterate(scenario)
        generation.fail_refit=True
        with self.assertRaisesRegex(RuntimeError,"failed refit"):
            observer.finish()
        self.assertEqual(observer.events.generation_version,1)
        self.assertFalse((observer.output/"result.json").exists())
        observer.close()
        self.assertNotIn("train",policy.__dict__)

    def test_after_refit_logprob_mismatch_rejects_completion_without_relaxing_tolerance(self):
        scenario=self.scenario("grpo")
        observer,module,policy,generation,*_=scenario
        self.iterate(scenario);self.iterate(scenario)
        policy.logprob_error=-.5
        with self.assertRaisesRegex(ValueError,"alignment failed"):
            observer.finish()
        self.assertFalse((observer.output/"result.json").exists())
        self.assertEqual(observer.collector.records,[])

    def test_missing_iteration_cannot_emit_a_successful_synthetic_or_observed_run(self):
        scenario=self.scenario("grpo")
        observer,*_=scenario
        self.iterate(scenario)
        with self.assertRaisesRegex(ValueError,"iterations are incomplete"):
            observer.finish()
        self.assertFalse((observer.output/"result.json").exists())


if __name__=="__main__":
    unittest.main(verbosity=2)
