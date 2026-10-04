"""Synthetic load-call evidence; no checkpoint, model or framework execution."""
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from experiments.runtime.nemo_loading import CheckpointLoadScope


class NeMoLoadingTests(unittest.TestCase):
    def setUp(self):
        folder=tempfile.TemporaryDirectory();self.addCleanup(folder.cleanup)
        self.root=Path(folder.name)
        self.snapshot=self.root/"hf";self.snapshot.mkdir()
        self.cache=self.root/"owned";self.cache.mkdir()
        self.destination=self.cache/"converted"
        self.calls=[];self.returned=object()
        def synthetic_import(source,destination,*args,**kwargs):
            self.calls.append(("import",source,destination))
            Path(destination).mkdir()
            return self.returned
        def synthetic_load(state,model,optimizer,scheduler,*args,**kwargs):
            self.calls.append(("load",model,optimizer))
            if kwargs.get("synthetic_failure"):
                raise RuntimeError("synthetic loading failure")
            return self.returned
        self.module=SimpleNamespace(import_model_from_hf_name=synthetic_import,load_checkpoint=synthetic_load)
        self.originals=(synthetic_import,synthetic_load)
        self.state=SimpleNamespace(cfg=SimpleNamespace(checkpoint=SimpleNamespace(
            pretrained_checkpoint=str(self.destination),load=None)))

    def scope(self,reference=True):
        return CheckpointLoadScope(self.module,snapshot=self.snapshot,cache_root=self.cache,
                                   reference_model=reference)

    def test_actual_call_objects_and_actor_reference_roles_are_preserved(self):
        model,optimizer,scheduler=object(),object(),object()
        with self.scope() as scope:
            self.assertIs(self.module.import_model_from_hf_name(str(self.snapshot),str(self.destination),{}),self.returned)
            self.assertIs(self.module.load_checkpoint(self.state,model,optimizer,scheduler,
                skip_load_to_model_and_opt=False),self.returned)
            self.assertIs(self.module.load_checkpoint(self.state,model,None,None),self.returned)
            result=scope.complete({"known_config":1})
            self.assertEqual({x["role"] for x in result["loads"]},{"actor","reference"})
            self.assertEqual(result["status"],"official_conversion_and_load_completed")
            self.assertEqual(len(result["effective_config_sha256"]),64)
        self.assertEqual((self.module.import_model_from_hf_name,self.module.load_checkpoint),self.originals)
        self.assertIs(self.calls[1][1],model)
        self.assertIs(self.calls[1][2],optimizer)

    def test_conversion_must_use_the_exact_source_and_fresh_owned_cache(self):
        with self.scope(False) as scope:
            for source,destination in ((self.root/"other",self.destination),
                                       (self.snapshot,self.root/"escaped"),
                                       (self.snapshot,self.cache)):
                with self.assertRaises(ValueError):
                    self.module.import_model_from_hf_name(str(source),str(destination),{})
            self.destination.mkdir()
            with self.assertRaisesRegex(ValueError,"fresh"):
                self.module.import_model_from_hf_name(str(self.snapshot),str(self.destination),{})
            self.assertEqual(self.calls,[])
            with self.assertRaisesRegex(ValueError,"incomplete"):
                scope.complete({})

    def test_skipped_or_wrong_checkpoint_cannot_be_reported_as_loaded(self):
        with self.scope(False) as scope:
            self.module.import_model_from_hf_name(str(self.snapshot),str(self.destination),{})
            with self.assertRaisesRegex(ValueError,"skipped"):
                self.module.load_checkpoint(self.state,object(),object(),None,skip_load_to_model_and_opt=True)
            self.state.cfg.checkpoint.pretrained_checkpoint=str(self.root/"different")
            with self.assertRaisesRegex(ValueError,"differs"):
                self.module.load_checkpoint(self.state,object(),object(),None)
            self.assertEqual(scope.loads,[])
            with self.assertRaisesRegex(ValueError,"incomplete"):
                scope.complete({})

    def test_load_failure_restores_original_functions_and_records_no_success(self):
        scope=self.scope(False)
        with self.assertRaisesRegex(RuntimeError,"loading failure"):
            with scope:
                self.module.import_model_from_hf_name(str(self.snapshot),str(self.destination),{})
                self.module.load_checkpoint(self.state,object(),object(),None,synthetic_failure=True)
        self.assertEqual(scope.loads,[])
        self.assertEqual((self.module.import_model_from_hf_name,self.module.load_checkpoint),self.originals)
