"""Real pinned tokenizer tests; fixture text only, no pretrained model answers."""

import importlib.util
import os
from pathlib import Path
import re
import shutil
import tempfile
import unittest
from unittest.mock import patch

from training.qwen_tokenizer import CompletionDecodeError, NONTHINKING_SUFFIX, QwenTextTokenizer


ARTIFACTS = Path(os.environ.get("QWEN_TOKENIZER_ARTIFACTS", str(Path(__file__).resolve().parent / "runs/qwen-tokenizer-artifacts")))
AVAILABLE = importlib.util.find_spec("transformers") is not None and (ARTIFACTS / "tokenizer.json").is_file()


@unittest.skipUnless(AVAILABLE, "optional pinned tokenizer environment and public artifacts required")
class QwenTokenizerTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = QwenTextTokenizer(ARTIFACTS)

    def test_actual_nonthinking_prompt_roundtrip_and_fresh_roles(self):
        tokenizer = self.tokenizer
        messages = (("system", "Return one decimal probability."), ("user", "A fair coin lands heads. 🪙"))
        rendered, ids = tokenizer.render_and_encode(messages)
        self.assertTrue(rendered.endswith(NONTHINKING_SUFFIX))
        self.assertEqual(rendered.count("<|im_start|>"), 3)
        self.assertIn("A fair coin lands heads. 🪙", rendered)
        self.assertEqual(ids, tokenizer.encode_messages(messages))
        self.assertEqual(tokenizer._tokenizer.decode(ids, skip_special_tokens=False,
                                                    clean_up_tokenization_spaces=False), rendered)
        original, original_ids = tokenizer.render_and_encode((messages[1],), original_user_only=True)
        self.assertEqual(original.count("<|im_start|>"), 2)
        self.assertLess(len(original_ids), len(ids))
        self.assertNotIn("Return one decimal probability.", original)
        for bad in ((messages[1],), ((),), (None,), messages + (("assistant", "0.5"),),
                    (("system", "rule"), ("tool", "response"))):
            with self.assertRaises(ValueError):
                tokenizer.encode_messages(bad)
        tokenizer.verify_runtime()

    def test_decimal_fixture_removes_only_one_terminal_eos(self):
        tokenizer = self.tokenizer
        prefix = tokenizer.encode_text("0.5")
        eos, pad = tokenizer.eos_token_id, tokenizer.pad_token_id
        self.assertEqual(tokenizer.decode_completion(prefix), "0.5")
        self.assertEqual(tokenizer.decode_completion(prefix + (eos,)), "0.5")
        self.assertEqual(tokenizer.decode_completion(prefix + (eos, eos)), "0.5<|im_end|>")
        self.assertEqual(tokenizer.decode_completion(prefix + (pad, eos)), "0.5<|endoftext|>")
        self.assertEqual(tokenizer.decode_completion((eos,) + prefix), "<|im_end|>0.5")
        self.assertEqual(tokenizer.model_config["text_config"]["eos_token_id"], pad)
        self.assertNotEqual(eos, pad)

    def test_reasoning_refusal_and_controls_stay_visible(self):
        tokenizer = self.tokenizer
        for text in ("<think>reasoning</think>0.5", "ABSTAIN", "0.5 commentary",
                     "<|im_start|>assistant\n0.5", "0.5<|image_pad|>"):
            tokens = tokenizer.encode_text(text) + (tokenizer.eos_token_id,)
            decoded = tokenizer.decode_completion(tokens)
            self.assertEqual(decoded, text)
            self.assertIsNone(re.fullmatch(r"(?:0(?:\.[0-9]+)?|1(?:\.0+)?)", decoded.strip()))

    def test_every_unmapped_model_output_id_is_rejected_with_raw_tokens(self):
        tokenizer = self.tokenizer
        mapped = set(tokenizer._tokenizer.get_vocab().values())
        all_ids = range(tokenizer.model_config["text_config"]["vocab_size"])
        unmapped = [token for token in all_ids if token not in mapped]
        self.assertEqual(unmapped, list(range(248077, 248320)))
        prefix = tokenizer.encode_text("0.5")
        for token in unmapped:
            tokens = prefix + (token, tokenizer.eos_token_id)
            with self.assertRaises(CompletionDecodeError) as caught:
                tokenizer.decode_completion(tokens)
            self.assertEqual(caught.exception.completion_token_ids, tokens)
            self.assertEqual(tokenizer._tokenizer.decode(prefix + (token,), skip_special_tokens=False,
                                                        clean_up_tokenization_spaces=False), "0.5")

    def test_malformed_tokens_never_decode_as_a_probability(self):
        tokenizer = self.tokenizer
        prefix = tokenizer.encode_text("0.5")
        for bad in (-1, 248320, True, 1.0, "15"):
            with self.assertRaises(CompletionDecodeError) as caught:
                tokenizer.decode_completion(prefix + (bad,))
            self.assertEqual(caught.exception.completion_token_ids, prefix + (bad,))

    def test_artifact_special_id_template_and_backend_drift_reject(self):
        tokenizer = self.tokenizer
        with patch.object(tokenizer, "eos_token_id", 248044):
            with self.assertRaisesRegex(ValueError, "drift"):
                tokenizer.verify_runtime()
        with patch.object(tokenizer._tokenizer, "chat_template", "changed"):
            with self.assertRaisesRegex(ValueError, "drift"):
                tokenizer.verify_runtime()
        with patch.dict(tokenizer.model_config["text_config"], {"eos_token_id": 0}):
            with self.assertRaisesRegex(ValueError, "drift"):
                tokenizer.verify_runtime()
        tokenizer._tokenizer.backend_tokenizer.enable_truncation(2)
        with self.assertRaisesRegex(ValueError, "drift"):
            tokenizer.encode_messages((("system", "rule"), ("user", "coin")))
        tokenizer._tokenizer.backend_tokenizer.no_truncation()
        tokenizer._tokenizer.add_tokens(["<synthetic-new-token>"])
        with self.assertRaisesRegex(ValueError, "drift"):
            tokenizer.verify_runtime()
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            for path in ARTIFACTS.glob("*.json"):
                shutil.copyfile(path, directory / path.name)
            with (directory / "config.json").open("ab") as stream:
                stream.write(b" ")
            with self.assertRaisesRegex(ValueError, "artifact differs"):
                QwenTextTokenizer(directory)


if __name__ == "__main__":
    unittest.main()
