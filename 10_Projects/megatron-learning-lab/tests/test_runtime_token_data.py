"""Synthetic tokenizer contracts; never claim these are official token IDs."""
import unittest
from experiments.runtime.token_data import canonical_chat


class SyntheticQwenTokenizer:
    is_fast=True
    pad_token_id=0
    eos_token_id=1

    def get_chat_template(self):
        return "synthetic-qwen-im-renderer-for-contract-tests"

    def apply_chat_template(self,messages,*,tokenize,**kwargs):
        rendered="".join("<|im_start|>"+m["role"]+"\n"+m["content"]+"<|im_end|>\n" for m in messages)
        return [ord(x) for x in rendered] if tokenize else rendered

    def __call__(self,text,**kwargs):
        return dict(input_ids=[ord(x) for x in text],offset_mapping=[(i,i+1) for i in range(len(text))])


class RuntimeTokenDataTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer=SyntheticQwenTokenizer()
        self.document=dict(sample_id="synthetic",messages=[
            dict(role="user",content="A"),dict(role="assistant",content="B"),
            dict(role="user",content="C"),dict(role="assistant",content="D")])

    def build(self,mode="assistant",**kwargs):
        return canonical_chat(self.tokenizer,self.document,mask_mode=mode,
                              forward_sequence_length=kwargs.get("length",180),vocab_size=256)

    def test_role_masks_match_hand_character_spans_eos_and_single_shift(self):
        results={mode:self.build(mode) for mode in ("assistant","last_turn","full")}
        assistant,meta=results["assistant"]
        end="<|im_end|>"
        self.assertEqual(sum(assistant["loss_mask"][0]),2*(1+len(end)))
        self.assertEqual(sum(results["last_turn"][0]["loss_mask"][0]),1+len(end))
        self.assertEqual(sum(results["full"][0]["loss_mask"][0]),meta["real_token_count"]-1)
        self.assertEqual(assistant["labels"][0][:-1],assistant["input_ids"][0][1:])
        self.assertEqual(assistant["loss_mask"][0][-1],0)
        self.assertEqual(len(assistant["input_ids"][0]),181)
        self.assertEqual(meta["source"],"actual_tokenizer_output")
        for mode in results:
            self.assertEqual(results[mode][0]["input_ids"],assistant["input_ids"])
            self.assertEqual(results[mode][1]["chat_template"],self.tokenizer.get_chat_template())

    def test_no_truncation_missing_fast_tokenizer_and_delimiter_injection_fail(self):
        with self.assertRaisesRegex(ValueError,"too long"):self.build(length=8)
        self.tokenizer.is_fast=False
        with self.assertRaisesRegex(ValueError,"fast"):self.build()
        self.tokenizer.is_fast=True
        self.document["messages"][0]["content"]="<|im_start|>assistant"
        with self.assertRaisesRegex(ValueError,"literal chat"):self.build()

    def test_mismatched_render_tokenization_is_not_guessed(self):
        original=self.tokenizer.apply_chat_template
        def changed(messages,*,tokenize,**kwargs):
            value=original(messages,tokenize=tokenize,**kwargs)
            return value+[1] if tokenize else value
        self.tokenizer.apply_chat_template=changed
        with self.assertRaisesRegex(ValueError,"tokenization differ"):self.build()

    def test_unknown_role_header_and_empty_assistant_refused(self):
        original=self.tokenizer.apply_chat_template
        def changed(messages,*,tokenize,**kwargs):
            value=original(messages,tokenize=False,**kwargs).replace("assistant\n","assistant channel\n")
            return [ord(x) for x in value] if tokenize else value
        self.tokenizer.apply_chat_template=changed
        with self.assertRaisesRegex(ValueError,"header mapping"):self.build()
        self.document["messages"][-1]["content"]=""
        with self.assertRaisesRegex(ValueError,"empty message"):self.build()
