"""Software-only sampler regressions; synthetic weights, never candidate logits."""

import hashlib
import math
from types import SimpleNamespace
import unittest

try:
    import torch
except ModuleNotFoundError as error:
    if error.name != "torch":
        raise
    torch = None

from training.categorical import categorical_sampler_identity
if torch is not None:
    from training.categorical import cdf_index, normalized_cdf, sample_categorical


@unittest.skipIf(torch is None, "optional pinned CPU environment required")
class CategoricalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_exact_zero_mass_boundaries_and_half_open_endpoint(self):
        weights = torch.tensor([0., 1., 0., 3., 0., 4., 0.])
        cdf = normalized_cdf(weights)
        torch.testing.assert_close(cdf, torch.tensor([0., .125, .125, .5, .5, 1., 1.], dtype=torch.float64),
                                   atol=0, rtol=0)
        for uniform, expected in ((0., 1), (math.nextafter(.125, 0.), 1), (.125, 3),
                                  (math.nextafter(.5, 0.), 3), (.5, 5), (math.nextafter(1., 0.), 5)):
            with self.subTest(uniform=uniform):
                self.assertEqual(cdf_index(cdf, torch.tensor(uniform, dtype=torch.float64)), expected)
        # An exact eighth-grid gives the declared masses without a noisy frequency test.
        self.assertEqual([cdf_index(cdf, torch.tensor(i / 8, dtype=torch.float64)) for i in range(8)],
                         [1, 3, 3, 3, 5, 5, 5, 5])
        for invalid in (1., math.nextafter(1., math.inf), -.1, math.nan, math.inf):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, "uniform"):
                cdf_index(cdf, torch.tensor(invalid, dtype=torch.float64))

    def test_fp64_accumulation_retains_mass_lost_by_float32_cdf(self):
        weights = torch.tensor([1., 2 ** -25, 2 ** -25, 0.], dtype=torch.float32)
        self.assertEqual(weights.cumsum(0)[1].item(), 1.)
        cdf = normalized_cdf(weights)
        self.assertLess(cdf[0].item(), cdf[1].item())
        self.assertLess(cdf[1].item(), cdf[2].item())
        self.assertEqual(cdf[2].item(), cdf[3].item())
        self.assertEqual(cdf[-1].item(), 1.)
        for index in range(3):
            left = 0. if index == 0 else cdf[index - 1].item()
            uniform = (left + cdf[index].item()) / 2
            self.assertEqual(cdf_index(cdf, torch.tensor(uniform, dtype=torch.float64)), index)

    def test_invalid_weights_and_logits_reject_before_rng_advances(self):
        for weights in (torch.tensor([]), torch.tensor([0., 0.]), torch.tensor([1., -1.]),
                        torch.tensor([math.nan, 1.]), torch.tensor([math.inf, 1.]),
                        torch.ones(2, 2), torch.ones(2, dtype=torch.float16)):
            with self.subTest(weights=weights), self.assertRaises(ValueError):
                normalized_cdf(weights)
        generator = torch.Generator().manual_seed(41)
        before = generator.get_state().clone()
        for logits in (torch.tensor([]), torch.tensor([math.nan]), torch.tensor([math.inf]),
                       torch.tensor([1, 2]), torch.ones(2, 2), torch.tensor([-math.inf], dtype=torch.float64)):
            with self.subTest(logits=logits), self.assertRaises(ValueError):
                sample_categorical(logits, generator=generator)
            self.assertTrue(torch.equal(generator.get_state(), before))

    def test_malformed_cdf_rejected(self):
        for values in ([], [.2, .1, 1.], [-.1, 1.], [.1, .9], [math.nan, 1.], [math.inf, 1.]):
            with self.subTest(values=values), self.assertRaisesRegex(ValueError, "CDF"):
                cdf_index(torch.tensor(values, dtype=torch.float64), torch.tensor(.5, dtype=torch.float64))
        with self.assertRaisesRegex(ValueError, "CDF"):
            cdf_index(torch.tensor([.5, 1.]), torch.tensor(.5, dtype=torch.float64))

    def test_generic_consumer_matches_independent_cdf_oracle_and_fp32_loss(self):
        from training.episode_adapter import _sampling_hash
        from training.torch_loop import (causal_policy_loss, generate_tokens, policy_manifest,
                                         policy_sha256, runtime_sampling_distribution)

        class FixedSyntheticModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.logits = torch.nn.Parameter(torch.tensor([-1.25, 0., 1., -1000.]))

            def forward(self, *, input_ids, attention_mask):
                return self.logits.to(torch.bfloat16).expand(*input_ids.shape, -1)

        model = FixedSyntheticModel().eval()
        reference = torch.Generator().manual_seed(718)
        weights = model.logits.detach().softmax(-1).tolist()
        cumulative = [math.fsum(weights[:index + 1]) / math.fsum(weights) for index in range(len(weights))]
        expected = []
        for _ in range(12):
            uniform = torch.rand((), dtype=torch.float64, generator=reference).item()
            expected.append(next(index for index, bound in enumerate(cumulative) if uniform < bound))
        generator = torch.Generator().manual_seed(718)
        generated = generate_tokens(model, (0,), eos_token_id=3, max_new_tokens=12, generator=generator)
        self.assertEqual(generated.completion_token_ids, tuple(expected))
        expected_logps = tuple(model.logits.detach().log_softmax(-1)[index].item() for index in expected)
        self.assertEqual(generated.token_logprobs, expected_logps)
        self.assertEqual(generated.finish_reason, "length")
        self.assertTrue(torch.equal(generator.get_state(), reference.get_state()))
        self.assertEqual(policy_manifest(model)["categorical_sampler"], categorical_sampler_identity())
        credits = SimpleNamespace(policy_sha256=policy_sha256(model),
            sampling_distribution_sha256=_sampling_hash(runtime_sampling_distribution(model)),
            input_ids=((0,) + generated.completion_token_ids,), attention_mask=((1,) * 13,),
            completion_loss_mask=((1,) * 12,), loss_coefficients=((-1.,) * 12,), trajectory_ids=("synthetic",))
        record = SimpleNamespace(trajectory=SimpleNamespace(request_id="synthetic"), token_logprobs=expected_logps)
        loss = causal_policy_loss(model, credits, records=(record,))
        self.assertEqual(loss.causal_logprobs.dtype, torch.float32)
        self.assertEqual(loss.max_behavior_logprob_error, 0.)
        self.assertTrue(loss.loss.requires_grad)
        self.assertAlmostEqual(loss.loss.item(), -math.fsum(expected_logps), places=5)

    def test_full_vocabulary_point_mass_at_multiple_indices(self):
        for selected in (0, 13, 70215, 123456, 248319):
            logits = torch.full((248320,), -1000., dtype=torch.float32)
            logits[selected] = 0.
            self.assertEqual(int(torch.count_nonzero(logits.softmax(-1))), 1)
            for seed in (0, 2026091310, 16432092232865781795):
                with self.subTest(selected=selected, seed=seed):
                    token, logprob = sample_categorical(logits, generator=torch.Generator().manual_seed(seed))
                    self.assertEqual(token, selected)
                    self.assertEqual(logprob, 0.)

    def test_deterministic_replay_and_exact_selected_fp32_logprob(self):
        logits = torch.tensor([-4., -.125, 0., 1.25, -1000.], dtype=torch.bfloat16)
        first, second = (torch.Generator().manual_seed(9182) for _ in range(2))
        outputs = [sample_categorical(logits, generator=first) for _ in range(32)]
        self.assertEqual(outputs, [sample_categorical(logits, generator=second) for _ in range(32)])
        self.assertTrue(torch.equal(first.get_state(), second.get_state()))
        self.assertGreater(len({token for token, _ in outputs}), 1)
        for token, logprob in outputs:
            self.assertNotEqual(token, 4)
            self.assertEqual(logprob, logits.float().log_softmax(-1)[token].item())
        # One uniform64 per token, independently verify total RNG advancement.
        reference = torch.Generator().manual_seed(9182)
        for _ in outputs:
            torch.rand((), dtype=torch.float64, generator=reference)
        self.assertTrue(torch.equal(first.get_state(), reference.get_state()))

    def test_fp64_logits_retain_normalization_precision(self):
        logits = torch.tensor([.2, -.1, .4], dtype=torch.float64)
        generator = torch.Generator().manual_seed(97)
        for _ in range(12):
            token, logprob = sample_categorical(logits, generator=generator)
            self.assertEqual(logprob, logits.log_softmax(-1)[token].item())
            self.assertNotEqual(logprob, logits.float().log_softmax(-1)[token].item())
        token, logprob = sample_categorical(torch.tensor([1e100], dtype=torch.float64), generator=generator)
        self.assertEqual((token, logprob), (0, 0.))

    @unittest.skipUnless(torch is not None and torch.__version__ == "2.6.0+cpu"
                         and torch.backends.mkl.is_available(), "historical defect is pinned to Torch 2.6 CPU/MKL")
    def test_retained_seed_full_vocabulary_exponential_floor_regression(self):
        # Retained campaign v2 index20: these are real RNG pins, not model logits.
        seed, size, rare = 16432092232865781795, 248320, 70215
        generator = torch.Generator().manual_seed(seed)
        torch.empty(size, dtype=torch.float32).exponential_(generator=generator)
        before_second = generator.get_state().clone()
        state_sha = lambda value: hashlib.sha256(value.numpy().tobytes()).hexdigest()
        self.assertEqual(state_sha(before_second), "8692323026cf42b8cb9c69aeadbf55f1e3c7dbf9df36ae96da050d774c91ce8a")
        noise = torch.empty(size, dtype=torch.float32).exponential_(
            generator=torch.Generator().set_state(before_second))
        self.assertEqual(noise[rare].item(), torch.finfo(torch.float32).tiny)
        self.assertEqual(noise.argmin().item(), rare)
        self.assertGreater(torch.topk(noise, 2, largest=False).values[1].item(), 4e-6)
        for dominant in (0, 13, 123456, size - 1):
            # Deliberately synthetic distribution: almost all mass at `dominant`.
            logits = torch.full((size,), -1000., dtype=torch.float32)
            logits[dominant] = 0.
            logits[rare] = -27.26654815673828
            weights = logits.softmax(-1)
            self.assertLess(weights[rare].item(), 2e-12)
            legacy_rng = torch.Generator().set_state(before_second)
            self.assertEqual(torch.multinomial(weights, 1, generator=legacy_rng).item(), rare)
            self.assertEqual(state_sha(legacy_rng.get_state()), "c915278a84697d544a08165a24a771fc0561101cd86d48670934905f1536175b")
            token, logprob = sample_categorical(logits, generator=torch.Generator().set_state(before_second))
            self.assertEqual(token, dominant)
            self.assertEqual(logprob, logits.log_softmax(-1)[dominant].item())


if __name__ == "__main__":
    unittest.main()
