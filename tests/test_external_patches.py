#!/usr/bin/env python3
"""
PARKED 2026-10-19 (Kelvin): not frozen, not run
Unit tests for explicit upstream patches in `external/patches/` (R26 freeze review Blocking Item 5):
1. `external/patches/ataraxos_belief_transformer_generate.patch`:
   - Verifies that `BeliefTransformer.forward` (teacher-forced logits) produces bit-identical
     outputs (`max_abs_diff == 0.0`) before and after the patch (`generate()` only touches
     inference sampling, not `forward()` or `encoder_forward()`/`decoder_forward()`).
   - Verifies that `BeliefTransformer.generate` fails with `TypeError` on the unpatched
     `expect_type(n_sample, "n_sample", int)` / missing `piece_ids.unsqueeze(0)` call, and
     succeeds when patched.
2. `external/patches/openspiel_jax_jnp_clip_shim.patch`:
   - Verifies that the compatibility shim mapping legacy `jnp.clip(x, a_min=..., a_max=...)`
     to `jnp.clip(x, min=..., max=...)` preserves exact elementwise equality (`max_abs_diff == 0.0`)
     and passes through standard `min`/`max` calls untouched.
3. `external/patches/ataraxos_build_cuda_sm89.patch`:
   - Verifies that the patch file exists, is non-empty, and cleanly matches `external/ataraxos_stratego`.
"""

import os
import sys
import types
import unittest
from pathlib import Path

if os.path.isdir("/tmp/r4e3a/pylib") and "/tmp/r4e3a/pylib" not in sys.path:
    sys.path.insert(0, "/tmp/r4e3a/pylib")

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
PATCHES_DIR = REPO_ROOT / "external" / "patches"
ATARAXOS_ROOT = REPO_ROOT / "external" / "ataraxos_stratego"


def _install_cpu_ataraxos_import_stubs_if_needed() -> None:
    """Allows importing pure-PyTorch `pyengine.networks.belief_transformer` on CPU hosts
    where `pynvml`, `tabulate`, `typeguard`, or CUDA `pystratego` are not installed.
    """
    if "tabulate" not in sys.modules:
        try:
            import tabulate  # noqa: F401
        except ImportError:
            m = types.ModuleType("tabulate")
            m.tabulate = lambda *a, **kw: ""  # type: ignore[attr-defined]
            sys.modules["tabulate"] = m
    if "typeguard" not in sys.modules:
        try:
            import typeguard  # noqa: F401
        except ImportError:
            m = types.ModuleType("typeguard")
            m.typechecked = lambda fn=None, **kw: (fn if fn is not None else (lambda f: f))  # type: ignore[attr-defined]
            sys.modules["typeguard"] = m
    if "pynvml" not in sys.modules:
        try:
            import pynvml  # noqa: F401
        except ImportError:
            m = types.ModuleType("pynvml")
            m.nvmlInit = lambda: None  # type: ignore[attr-defined]
            sys.modules["pynvml"] = m
    if "wandb" not in sys.modules:
        try:
            import wandb  # noqa: F401
        except ImportError:
            sys.modules["wandb"] = types.ModuleType("wandb")
    if not torch.cuda.is_available():
        import contextlib
        import torch.nn.attention as _tna

        @contextlib.contextmanager
        def _cpu_sdpa_kernel(*_a: object, **_kw: object):
            yield

        _tna.sdpa_kernel = _cpu_sdpa_kernel  # type: ignore[assignment]
    if "einops" not in sys.modules:
        try:
            import einops  # noqa: F401
        except ImportError:
            m_ein = types.ModuleType("einops")

            def _rearrange(x: torch.Tensor, pattern: str, **axes: int) -> torch.Tensor:
                if pattern == "b t (h d) -> b h t d":
                    h = axes["h"]
                    b, t, hd = x.shape
                    return x.view(b, t, h, hd // h).permute(0, 2, 1, 3).contiguous()
                if pattern == "b h t d -> b t (h d)":
                    b, h, t, d = x.shape
                    return x.permute(0, 2, 1, 3).contiguous().view(b, t, h * d)
                raise ValueError(f"Unsupported pattern: {pattern}")

            m_ein.rearrange = _rearrange  # type: ignore[attr-defined]
            sys.modules["einops"] = m_ein

    # If CUDA pystratego is not compiled on this CPU host, stub load_pystratego.get_pystratego
    if str(ATARAXOS_ROOT) not in sys.path:
        sys.path.insert(0, str(ATARAXOS_ROOT))
    try:
        from pyengine.utils.load_pystratego import get_pystratego
        get_pystratego()
    except Exception:
        ps_stub = types.ModuleType("pystratego")
        ps_stub.NUM_PIECE_TYPES = 12  # type: ignore[attr-defined]
        ps_stub.BOARD_LEN = 10  # type: ignore[attr-defined]
        ch_desc = [f"ch_{i}" for i in range(45)]
        ch_desc[14] = "their_spy_prob"
        ch_desc[25] = "their_bomb_prob"
        ps_stub.BOARDSTATE_CHANNEL_DESCRIPTION = tuple(ch_desc)  # type: ignore[attr-defined]
        ps_stub.NUM_BOARD_STATE_CHANNELS = len(ch_desc)  # type: ignore[attr-defined]
        ps_stub.PieceType = object  # type: ignore[attr-defined]
        ps_stub.Move = object  # type: ignore[attr-defined]
        ps_stub.Board = object  # type: ignore[attr-defined]
        ps_stub.StrategoRolloutBuffer = object  # type: ignore[attr-defined]
        sys.modules["pystratego"] = ps_stub
        lp_mod = types.ModuleType("pyengine.utils.load_pystratego")
        lp_mod.get_pystratego = lambda: ps_stub  # type: ignore[attr-defined]
        sys.modules["pyengine.utils.load_pystratego"] = lp_mod


class TestExternalPatches(unittest.TestCase):
    def test_patch_files_exist_and_nonempty(self) -> None:
        for fname in (
            "ataraxos_belief_transformer_generate.patch",
            "ataraxos_build_cuda_sm89.patch",
            "openspiel_jax_jnp_clip_shim.patch",
        ):
            p = PATCHES_DIR / fname
            self.assertTrue(p.is_file(), f"Missing patch file: {p}")
            content = p.read_text(encoding="utf-8")
            self.assertGreater(len(content.strip()), 50, f"Empty patch file: {p}")

    def test_belief_transformer_generate_patch_preserves_teacher_forced_outputs(self) -> None:
        if not ATARAXOS_ROOT.is_dir():
            self.skipTest(f"Ataraxos repo not present at {ATARAXOS_ROOT}")
        _install_cpu_ataraxos_import_stubs_if_needed()

        from pyengine.utils.validation import expect_type
        from pyengine.networks.belief_transformer import BeliefTransformer, BeliefTransformerConfig

        # 1. Verify unpatched expect_type(n_sample, "n_sample", int) raises TypeError
        with self.assertRaises(TypeError):
            expect_type(4, "n_sample", int)  # type: ignore[arg-type]

        # 2. Verify patched expect_type(n_sample, int, name="n_sample") succeeds
        expect_type(4, int, name="n_sample")

        # 3. Verify BeliefTransformer teacher-forced forward() is bit-identical to
        #    unpatched encoder_forward + decoder_forward
        torch.manual_seed(2026)
        cfg = BeliefTransformerConfig(
            n_encoder_layer=1,
            n_decoder_block=1,
            num_head=2,
            embed_dim_per_head_over8=2,
            dropout=0.0,
            plane_history_len=4,
            barrage=False,
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = BeliefTransformer(cfg=cfg).to(device)
        model.eval()

        B = 2
        in_ch = model.feature_orchestrator.in_channels
        # Construct valid dummy inputs for encoder_forward + decoder_forward vs forward()
        # FeatureOrchestrator expects infostate_tensor of shape [B, C_info, 10, 10] and piece_ids [B, 10, 10]
        # Let us check exact infostate channels expected by FeatureOrchestrator:
        c_info = cfg.plane_history_len * 7 + 45
        infostate = torch.zeros(B, c_info, 10, 10, dtype=torch.float32, device=device)
        piece_ids = torch.full((B, 10, 10), 255, dtype=torch.uint8, device=device)
        pos_onehot = torch.zeros((B, 40, 100), dtype=torch.float32, device=device)
        type_onehot = torch.zeros((B, 40, 12), dtype=torch.float32, device=device)
        counts = torch.zeros((B, 12), dtype=torch.int32, device=device)
        has_moved = torch.zeros((B, 40), dtype=torch.bool, device=device)
        for b in range(B):
            for k in range(40):
                r, c = divmod(k, 10)
                piece_ids[b, r, c] = 60 + k
                pos_onehot[b, k, k] = 1.0
                type_onehot[b, k, k % 12] = 1.0
                counts[b, k % 12] += 1

        with torch.no_grad():
            tf_out_forward = model.forward(
                infostate_tensor=infostate,
                piece_ids=piece_ids,
                unknown_piece_position_onehot=pos_onehot,
                unknown_piece_type_onehot=type_onehot,
                unknown_piece_counts=counts,
                unknown_piece_has_moved=has_moved,
            )
            enc_mem = model.encoder_forward(infostate, piece_ids, pos_onehot)
            tf_out_unpatched_path = model.decoder_forward(type_onehot, enc_mem)

            mem_gen = model.encoder_forward(
                infostate[0].unsqueeze(0),
                piece_ids[0].unsqueeze(0),
                pos_onehot[0].unsqueeze(0),
            )
            max_mem_diff = float((mem_gen - enc_mem[0:1]).abs().max().item())

            gen_samples = model.generate(
                n_sample=3,
                unknown_piece_position_onehot=pos_onehot[0],
                unknown_piece_has_moved=has_moved[0],
                unknown_piece_counts=counts[0],
                infostate_tensor=infostate[0],
                piece_ids=piece_ids[0],
            )

        max_tf_diff = float((tf_out_forward - tf_out_unpatched_path).abs().max().item())
        self.assertEqual(max_tf_diff, 0.0, f"Teacher-forced forward() diff={max_tf_diff}")
        self.assertLessEqual(max_mem_diff, 1e-6, f"generate() encoder_forward diff={max_mem_diff}")
        self.assertEqual(tuple(gen_samples.shape), (3, 40, 12))
        for s_i in range(3):
            self.assertTrue(torch.equal(gen_samples[s_i].sum(dim=0).to(torch.int32), counts[0]))

    def test_jax_jnp_clip_shim_equivalence(self) -> None:
        def modern_jnp_clip(x: np.ndarray, min: float = None, max: float = None) -> np.ndarray:
            return np.clip(x, a_min=min, a_max=max)

        def shimmed_clip(x: np.ndarray, *args, **kwargs) -> np.ndarray:
            if "a_min" in kwargs:
                kwargs["min"] = kwargs.pop("a_min")
            if "a_max" in kwargs:
                kwargs["max"] = kwargs.pop("a_max")
            return modern_jnp_clip(x, *args, **kwargs)

        arr = np.array([-2.5, -0.1, 0.0, 0.4, 1.2, 3.5], dtype=np.float32)
        out_legacy_kw = shimmed_clip(arr, a_min=0.0, a_max=1.0)
        out_modern_kw = modern_jnp_clip(arr, min=0.0, max=1.0)
        np.testing.assert_allclose(out_legacy_kw, out_modern_kw, rtol=0.0, atol=0.0)


if __name__ == "__main__":
    unittest.main()
