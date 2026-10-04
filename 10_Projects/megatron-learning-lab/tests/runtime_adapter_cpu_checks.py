"""Real CPU tensor checks; fabricated framework containers are synthetic contracts."""
import copy
import math
from types import SimpleNamespace
import unittest
import torch
from experiments.runtime.adapters import bridge_sft_slice, hf_sft_slice, nemo_action_slice, LossTap
from experiments.runtime.reference_trace import generate


def synthetic_sink(*args, **kwargs):
    pass


class RuntimeAdapterCPUChecks(unittest.TestCase):
    def setUp(self):
        self.data=dict(alignment="next_token",input_ids=[[0,1,2],[2,1,0]],
                       labels=[[1,2,-100],[1,0,-100]],loss_mask=[[1,1,0],[1,0,0]],
                       position_ids=[[0,1,2],[0,1,2]],document_ids=[[0,0,0],[0,0,0]])
        self.logits=torch.tensor([[[0.,math.log(2),math.log(3)]]*3,
                                  [[math.log(3),math.log(2),0.]]*3],dtype=torch.float64,requires_grad=True)
        self.batch=dict(tokens=torch.tensor([[0,1],[2,1]]),labels=torch.tensor([[1,2],[1,0]]),
                        loss_mask=torch.tensor([[1.,1.],[1.,0.]],dtype=torch.float64),
                        position_ids=torch.tensor([[0,1],[0,1]]))

    def test_hf_target_probabilities_and_global_reduction_match_hand_values(self):
        expected=(-math.log(2/6)-math.log(3/6)-math.log(2/6))/3
        for output in ({"logits":self.logits},SimpleNamespace(logits=self.logits)):
            result=hf_sft_slice(output,self.data,{"vocab_size":3})
            self.assertAlmostEqual(result["loss_mean"],expected,places=14)
            self.assertEqual(result["token_count"],3)
            self.assertEqual(result["token_logprobs"][0][-1],0)
        self.assertIsNone(self.logits.grad)

    def test_bridge_token_loss_is_not_reinterpreted_as_logits_or_shifted_twice(self):
        target=self.batch["labels"]
        output=torch.nn.functional.cross_entropy(self.logits[:,:2].transpose(1,2),target,reduction="none")
        data,actual=bridge_sft_slice(self.batch,output,vocab_size=3,mapping="bridge_bsh_v1",output_kind="token_loss")
        expected=hf_sft_slice({"logits":self.logits},self.data,{"vocab_size":3})
        self.assertEqual(data,self.data)
        self.assertAlmostEqual(actual["loss_mean"],expected["loss_mean"],places=14)
        self.assertEqual(actual["token_count"],3)
        self.assertTrue(output.requires_grad)
        output.sum().backward()
        self.assertGreater(self.logits.grad.norm().item(),0)
        with self.assertRaisesRegex(ValueError,"logits.*shape"):
            bridge_sft_slice(self.batch,output,vocab_size=3,mapping="bridge_bsh_v1",output_kind="logits")

    def test_explicit_bsh_and_sbh_mappings_have_equal_values(self):
        bsh=self.logits[:,:2]
        left=bridge_sft_slice(self.batch,bsh,vocab_size=3,mapping="bridge_bsh_v1",output_kind="logits")
        right=bridge_sft_slice(self.batch,bsh.transpose(0,1),vocab_size=3,mapping="bridge_sbh_v1",output_kind="logits")
        self.assertEqual(left,right)
        # Deliberately non-square BS to catch a mistaken layout.
        batch={k:v[:1] for k,v in self.batch.items()}
        with self.assertRaisesRegex(ValueError,"shape"):
            bridge_sft_slice(batch,bsh[:1].transpose(0,1),vocab_size=3,mapping="bridge_bsh_v1",output_kind="logits")

    def test_mapping_missing_mask_packing_and_double_shift_are_rejected(self):
        for mutate in (
            lambda b:b.pop("loss_mask"),
            lambda b:b.update(cu_seqlens=torch.tensor([0,2])),
            lambda b:b["labels"].__setitem__((0,0),2),
            lambda b:b["position_ids"].__setitem__((0,1),0),
            lambda b:b["loss_mask"].zero_(),
        ):
            batch=copy.deepcopy(self.batch);mutate(batch)
            with self.assertRaises(ValueError):
                bridge_sft_slice(batch,self.logits[:,:2],vocab_size=3,mapping="bridge_bsh_v1",output_kind="logits")
        with self.assertRaisesRegex(ValueError,"unknown"):
            hf_sft_slice({"logits":self.logits},self.data,{"vocab_size":3},mapping="auto_guess")
        with self.assertRaisesRegex(ValueError,"HF logits must be a tensor"):
            hf_sft_slice({"loss":torch.tensor(1.)},self.data,{"vocab_size":3})

    def test_capture_preserves_original_loss_object_and_exact_backward(self):
        seen=[]
        class SyntheticLoss:
            input_type="synthetic_logits"
            def __call__(self,x,target):
                loss=torch.nn.functional.cross_entropy(x,target)
                self.result=(loss,{"loss":loss.item()})
                return self.result
        fn=SyntheticLoss()
        tap=LossTap(fn,lambda args,kwargs,result:seen.append(float(result[0].detach())))
        logits=self.logits[0,:2].detach().clone().requires_grad_()
        expected=logits.detach().clone().requires_grad_()
        got=tap(logits,torch.tensor([1,2]))
        self.assertIs(got,fn.result)
        self.assertEqual(tap.input_type,"synthetic_logits")
        got[0].backward()
        torch.nn.functional.cross_entropy(expected,torch.tensor([1,2])).backward()
        torch.testing.assert_close(logits.grad,expected.grad,rtol=0,atol=0)
        self.assertEqual(len(seen),1)

    def test_sink_failure_is_not_silently_ignored(self):
        def sink(*unused):raise ValueError("capture invalid")
        with self.assertRaisesRegex(ValueError,"capture invalid"):
            LossTap(lambda:torch.tensor(1.),sink)()

    def test_nemo_action_mapping_matches_fresh_real_cpu_reference(self):
        import json
        r=generate()["traces"][1]
        data=json.loads(r["input_json"]);config=json.loads(r["config_json"]);m=r["measurements"]
        batch={k:torch.tensor(data[k]) for k in ("input_ids","sample_mask")}
        batch["token_mask"]=torch.tensor(data["response_mask"])
        for field, key in (("generation_logprobs","generation_logprobs"),("prev_logprobs","previous_logprobs"),
                           ("reference_policy_logprobs","reference_logprobs"),("advantages","advantages"),
                           ("values","old_values"),("returns","returns")):
            batch[field]=torch.tensor(m[key],dtype=torch.float64)
        current=torch.tensor(m["current_logprobs"],dtype=torch.float64)[:,1:].requires_grad_()
        identities={k:data[k] for k in ("trajectory_ids","prompt_ids","group_ids")}
        actual,values=nemo_action_slice(current,batch,config=config,identities=identities,versions=data["policy_versions"])
        self.assertEqual(actual,data)
        for key in values:
            expected=copy.deepcopy(m[key])
            if key in ("generation_logprobs","previous_logprobs","reference_logprobs","advantages"):
                for row in expected:row[0]=0.
            self.assertEqual(values[key],expected,key)
        self.assertIsNone(current.grad)
        self.assertNotEqual(values["returns"][3][10],0) # masked gap must remain actual value
        with self.assertRaisesRegex(ValueError,"shape"):
            nemo_action_slice(current[:,:-1],batch,config=config,identities=identities,versions=data["policy_versions"])
        bad=dict(batch);bad["sample_mask"]=torch.tensor([1,1,1,0])
        with self.assertRaisesRegex(ValueError,"filtered"):
            nemo_action_slice(current,bad,config=config,identities=identities,versions=data["policy_versions"])

    def test_no_framework_import_from_adapter_module(self):
        import os,subprocess,sys
        env={k:v for k,v in os.environ.items() if k!="PYTHONPATH"}
        code="import sys;import experiments.runtime.adapters;assert 'torch' not in sys.modules;assert 'ray' not in sys.modules"
        subprocess.run([sys.executable,"-S","-c",code],env=env,check=True)


    def test_bridge_forward_tap_preserves_official_style_return_and_backward(self):
        from experiments.runtime.capture import BridgeForwardTap
        recorded=[]
        class SyntheticForward:
            def __call__(self,state,data_iterator,model,return_schedule_plan=False):
                batch=next(data_iterator)
                output=torch.nn.functional.cross_entropy(model.transpose(1,2),batch["labels"],reduction="none")
                self.result=(output,lambda x:(x*batch["loss_mask"]).sum()/batch["loss_mask"].sum())
                return self.result
        delegate=SyntheticForward()
        tap=BridgeForwardTap(delegate,lambda data,m,state,model:recorded.append((data,m)),vocab_size=3)
        result=tap(None,iter([self.batch]),self.logits[:,:2])
        self.assertIs(result,delegate.result)
        self.assertEqual(recorded[0][0],self.data)
        loss=result[1](result[0]);loss.backward()
        expected=self.logits.detach().clone().requires_grad_()
        ce=torch.nn.functional.cross_entropy(expected[:,:2].transpose(1,2),self.batch["labels"],reduction="none")
        ((ce*self.batch["loss_mask"]).sum()/3).backward()
        torch.testing.assert_close(self.logits.grad,expected.grad,rtol=0,atol=0)
        self.assertAlmostEqual(recorded[0][1]["loss_mean"],loss.item(),places=14)
        with self.assertRaisesRegex(ValueError,"schedule-plan"):
            tap(None,iter([self.batch]),self.logits[:,:2],return_schedule_plan=True)

    def test_parameter_slice_observes_actual_sgd_without_changing_gradients(self):
        from experiments.runtime.capture import ParameterSlice
        parameter=torch.nn.Parameter(torch.tensor([1.,2.],dtype=torch.float64))
        observer=ParameterSlice(parameter,[0])
        optimizer=torch.optim.SGD([parameter],lr=.1)
        observer.start()
        parameter.square().sum().backward();optimizer.step()
        observed=observer.finish(optimizer_executed=True)
        observer.close()
        self.assertEqual(observed["before"],1.)
        self.assertEqual(observed["gradient"],2.)
        self.assertAlmostEqual(observed["after"],.8,places=14)
        self.assertEqual(observed["backward_calls"],1)
        self.assertTrue(observed["optimizer_executed"])
        self.assertTrue(observed["changed"])
        torch.testing.assert_close(parameter,torch.tensor([.8,1.6],dtype=torch.float64),rtol=0,atol=1e-15)


    def test_loss_tap_serialization_preserves_delegation_for_worker_transport(self):
        import pickle
        tap=LossTap(torch.nn.MSELoss(),synthetic_sink)
        restored=pickle.loads(pickle.dumps(tap))
        x=torch.tensor([2.],requires_grad=True)
        got=restored(x,torch.tensor([1.]))
        got.backward()
        self.assertEqual(got.item(),1.)
        self.assertEqual(x.grad.item(),2.)


if __name__=="__main__":unittest.main(verbosity=2)
