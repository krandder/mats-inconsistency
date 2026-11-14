"""Exact local-loader mechanics using tiny serialized fixtures, never Qwen weights."""

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

from training.qwen_checkpoint import (MODEL_ID, PINNED_FILES, REVISION, UPSTREAM_METADATA_SHA256,
                                      _assign_verified_tensors, _manifest, _realize_rotary_buffers,
                                      _scan_safetensors, tensor_sha256)


TENSOR_ENVIRONMENT = all(importlib.util.find_spec(name) is not None for name in
                         ("torch", "safetensors", "transformers"))


class CheckpointManifestTests(unittest.TestCase):
    def test_manifest_identity_and_authoritative_file_pins(self):
        manifest = {"schema_version": 1, "model_id": MODEL_ID, "revision": REVISION,
                    "upstream_metadata_sha256": UPSTREAM_METADATA_SHA256,
                    "files": {name: {"bytes": size, "sha256": digest}
                              for name, (size, digest) in PINNED_FILES.items()}}
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "manifest.json"
            raw = json.dumps(manifest).encode()
            path.write_bytes(raw)
            actual, identity = _manifest(path)
            self.assertEqual(actual, manifest)
            self.assertEqual(identity, hashlib.sha256(raw).hexdigest())
            variants = []
            wrong = copy.deepcopy(manifest)
            wrong["revision"] = "main"
            variants.append(wrong)
            wrong = copy.deepcopy(manifest)
            wrong["files"][next(n for n in PINNED_FILES if n.endswith(".safetensors"))]["sha256"] = "0" * 64
            variants.append(wrong)
            wrong = copy.deepcopy(manifest)
            del wrong["files"]["config.json"]
            variants.append(wrong)
            for wrong in variants:
                path.write_text(json.dumps(wrong))
                with self.assertRaises(ValueError):
                    _manifest(path)
            path.write_text('{"schema_version":1,"schema_version":1}')
            with self.assertRaisesRegex(ValueError, "duplicate JSON"):
                _manifest(path)

    def test_header_structure_is_checked_even_with_matching_file_hash(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "small.safetensors"
            for row in ({"dtype": "BF16", "shape": [1], "data_offsets": [1, 3]},
                        {"dtype": "BF16", "shape": [True], "data_offsets": [0, 2]},
                        {"dtype": "F64", "shape": [1], "data_offsets": [0, 8]}):
                header = json.dumps({"weight": row}).encode()
                raw = struct.pack("<Q", len(header)) + header + b"\0\0"
                path.write_bytes(raw)
                with self.assertRaises(ValueError):
                    _scan_safetensors(path, expected_size=len(raw),
                                      expected_hash=hashlib.sha256(raw).hexdigest(), expected_names={"weight"})


@unittest.skipUnless(TENSOR_ENVIRONMENT, "optional CPU plus pinned tokenizer libraries required")
class SerializedCheckpointMechanicsTests(unittest.TestCase):
    def setUp(self):
        import torch
        from safetensors.torch import save_file

        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.path = self.root / "small.safetensors"
        self.values = {"embedding": torch.tensor([[1., 2., 3.], [4., 5., 6.]], dtype=torch.bfloat16),
                       "head": torch.tensor([[6., 5., 4.], [3., 2., 1.]], dtype=torch.bfloat16),
                       "scale": torch.tensor(0.375, dtype=torch.float32)}
        save_file(self.values, str(self.path))
        raw = self.path.read_bytes()
        self.inventory, self.receipt, _ = _scan_safetensors(
            self.path, expected_size=len(raw), expected_hash=hashlib.sha256(raw).hexdigest(),
            expected_names=set(self.values), chunk_bytes=7)
        self.inventory = {name: {**row, "checkpoint_name": name} for name, row in self.inventory.items()}
        self.model = torch.nn.Module()
        for name, value in self.values.items():
            self.model.register_parameter(name, torch.nn.Parameter(torch.empty(value.shape, device="meta")))

    def test_bf16_and_scalar_hashes_match_serialized_bytes_without_conversion(self):
        import torch

        for name, value in self.values.items():
            self.assertEqual(tensor_sha256(value, chunk_bytes=3), self.inventory[name]["sha256"])
        scalar = torch.tensor(1., dtype=torch.bfloat16)
        self.assertEqual(tensor_sha256(scalar), hashlib.sha256(b"\x80\x3f").hexdigest())
        with self.assertRaises(ValueError):
            tensor_sha256(self.values["embedding"].t())

    def test_strict_assign_preserves_mixed_dtype_and_storages_after_archive_closes(self):
        import torch

        assigned_pointers = {}
        original_load = self.model.load_state_dict
        def observe(state, **kwargs):
            self.assertEqual(kwargs, {"strict": True, "assign": True})
            assigned_pointers.update({name: tensor.data_ptr() for name, tensor in state.items()})
            return original_load(state, **kwargs)
        with patch.object(self.model, "load_state_dict", observe):
            _assign_verified_tensors(self.model, self.root, self.inventory)
        for name, parameter in self.model.named_parameters():
            self.assertFalse(parameter.requires_grad)
            self.assertEqual(parameter.dtype, self.values[name].dtype)
            self.assertEqual(parameter.data_ptr(), assigned_pointers[name])
            torch.testing.assert_close(parameter, self.values[name], rtol=0, atol=0)
        self.assertNotEqual(self.model.embedding.data_ptr(), self.model.head.data_ptr())
        self.assertFalse(self.model.training)

    def test_changed_payload_is_rejected_by_file_and_loaded_tensor_checks(self):
        raw = self.path.read_bytes()
        with self.path.open("r+b") as stream:
            stream.seek(-1, 2)
            stream.write(bytes([raw[-1] ^ 1]))
        with self.assertRaisesRegex(ValueError, "shard hash mismatch"):
            _scan_safetensors(self.path, expected_size=len(raw),
                              expected_hash=hashlib.sha256(raw).hexdigest(), expected_names=set(self.values))
        with self.assertRaisesRegex(ValueError, "loaded tensor differs"):
            _assign_verified_tensors(self.model, self.root, self.inventory)

    def test_meta_keys_shape_and_loaded_dtype_are_exact(self):
        for change, message in (("missing", "state keys"), ("extra", "state keys"),
                                ("shape", "shape mismatch"), ("dtype", "loaded tensor differs")):
            inventory = copy.deepcopy(self.inventory)
            if change == "missing":
                del inventory["scale"]
            elif change == "extra":
                inventory["extra"] = inventory["scale"]
            elif change == "shape":
                inventory["embedding"]["shape"] = [3, 2]
            else:
                inventory["embedding"]["dtype"] = "F32"
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, message):
                _assign_verified_tensors(self.model, self.root, inventory)

    def test_only_nonpersistent_rotary_buffers_are_realized(self):
        import torch
        from transformers import Qwen3_5TextConfig
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextRotaryEmbedding

        model = torch.nn.Module()
        model.config = Qwen3_5TextConfig(rope_parameters={"rope_type": "default", "rope_theta": 1000000.,
                                                        "partial_rotary_factor": .25})
        model.model = torch.nn.Module()
        model.register_parameter("fixture_weight", torch.nn.Parameter(torch.tensor([1.]), requires_grad=False))
        pointer = model.fixture_weight.data_ptr()
        with torch.device("meta"):
            model.model.rotary_emb = Qwen3_5TextRotaryEmbedding(model.config)
        evidence = _realize_rotary_buffers(model)
        self.assertEqual(set(evidence), {"model.rotary_emb.inv_freq", "model.rotary_emb.original_inv_freq"})
        self.assertEqual(model.fixture_weight.data_ptr(), pointer)
        self.assertEqual(set(model.state_dict()), {"fixture_weight"})
        expected = 1. / (1000000. ** (torch.arange(0, 64, 2, dtype=torch.float32) / 64))
        for name, buffer in model.named_buffers():
            self.assertEqual(buffer.device.type, "cpu")
            self.assertEqual(buffer.dtype, torch.float32)
            torch.testing.assert_close(buffer, expected, rtol=0, atol=0)
            self.assertEqual(tensor_sha256(buffer), evidence[name]["sha256"])
        model.register_buffer("unsupported_extra", torch.empty(1, device="meta"), persistent=False)
        with self.assertRaisesRegex(ValueError, "unexpected nonpersistent"):
            _realize_rotary_buffers(model)


if __name__ == "__main__":
    unittest.main()
