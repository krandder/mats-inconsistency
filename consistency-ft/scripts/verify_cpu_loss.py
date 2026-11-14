"""Independent finite-difference check of the real tensor loss backend.

Uses explicit position-logit parameters, uneven role lengths and hand-specified
joint rewards. This is a synthetic software check, not a language-model result.
Run with the isolated environment: runs/cpu-dev-venv/bin/python scripts/verify_cpu_loss.py
The JSON on stdout records this synthetic software verification only.
"""
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if not __debug__:
    raise RuntimeError("Run this verification without Python optimization flags")
import torch
from anchors import canonical_bytes, sha256
from training.episode_adapter import TokenCreditBatch
from training.torch_loop import causal_policy_loss, policy_sha256

torch.set_num_threads(1)
prompts = [[3, 4], [4, 2, 5], [6, 3], [3, 5, 2, 4], [2, 6, 4]]
completions = [[2, 1], [3, 6, 1], [4, 5, 1], [5, 1], [6, 1]]
rewards = [-0.4] * 3 + [-0.6] * 2
joint_count, vocab = 2, 7
width = max(len(p) + len(c) for p, c in zip(prompts, completions))
ids, attention, masks, coefficients = [], [], [], []
for prompt, completion, reward in zip(prompts, completions, rewards):
    row = prompt + completion
    ids.append(tuple(row + [0] * (width - len(row))))
    attention.append(tuple([1] * len(row) + [0] * (width - len(row))))
    mask = tuple(int(len(prompt) <= target_pos < len(row)) for target_pos in range(1, width))
    masks.append(mask)
    coefficients.append(tuple(-reward / joint_count * active for active in mask))

raw = [[[math.sin((row + 1) * 11 + position * 7 + token) / 3
         for token in range(vocab)] for position in range(width)] for row in range(5)]


class PositionLogitFixture(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.raw = torch.nn.Parameter(torch.tensor(raw, dtype=torch.float64))

    def forward(self, *, input_ids, attention_mask):
        assert tuple(input_ids.shape) == (5, width)
        assert tuple(attention_mask.shape) == (5, width)
        return self.raw


sampling = {"schema_version": 1, "sampler": "categorical", "temperature": 1.0,
            "top_p": 1.0, "top_k": 0, "logits_processors": [], "logits_warpers": [],
            "model_mode": "eval", "dropout_enabled": False}
model = PositionLogitFixture().eval()
credits = TokenCreditBatch(
    "a" * 64, "b" * 64, policy_sha256(model), sha256(canonical_bytes(sampling)),
    tuple("trajectory-" + str(i) for i in range(5)), ("and",) * 3 + ("negation",) * 2,
    tuple(rewards), tuple(ids), tuple(attention), tuple(masks), tuple(coefficients),
    joint_count, ())


def independent_loss(values):
    total = 0.0
    # Independently establish the generated-target positions from prompt lengths;
    # do not use the implementation's mask, coefficients or log-probability output.
    for row, (prompt, completion, reward) in enumerate(zip(prompts, completions, rewards)):
        for offset, target in enumerate(completion):
            position = len(prompt) + offset - 1
            logits = values[row][position]
            largest = max(logits)
            log_probability = logits[target] - largest - math.log(math.fsum(math.exp(v - largest) for v in logits))
            total += -reward / joint_count * log_probability
    return total


result = causal_policy_loss(model, credits)
result.loss.backward()
expected_loss = independent_loss(raw)
assert abs(result.loss.item() - expected_loss) < 1e-12
epsilon, max_error, checked = 1e-5, 0.0, 0
for row in range(5):
    for position in range(width):
        for token in range(vocab):
            original = raw[row][position][token]
            raw[row][position][token] = original + epsilon
            plus = independent_loss(raw)
            raw[row][position][token] = original - epsilon
            minus = independent_loss(raw)
            raw[row][position][token] = original
            finite_difference = (plus - minus) / (2 * epsilon)
            observed = model.raw.grad[row, position, token].item()
            max_error = max(max_error, abs(observed - finite_difference))
            checked += 1
assert max_error < 1e-8, max_error

for row, (prompt, completion) in enumerate(zip(prompts, completions)):
    for offset, target in enumerate(completion):
        assert model.raw.grad[row, len(prompt) + offset - 1, target] > 0
    eos_position = len(prompt) + len(completion) - 2
    assert model.raw.grad[row, eos_position, 1] > 0

before = model.raw.detach().clone()
torch.optim.SGD(model.parameters(), lr=0.001).step()
after = model.raw.detach().tolist()
assert independent_loss(after) < expected_loss
assert not torch.equal(before, model.raw.detach())
try:
    causal_policy_loss(model, credits)
except ValueError as error:
    stale_error = str(error)
else:
    raise AssertionError("Updated policy was accepted for a stale rollout batch")

print(json.dumps({
    "passed": True, "synthetic_only": True, "joint_samples": joint_count, "role_trajectories": 5,
    "generated_tokens_including_eos": sum(map(len, completions)),
    "finite_difference_parameters_checked": checked, "maximum_gradient_absolute_error": max_error,
    "tensor_loss": result.loss.item(), "independent_loss": expected_loss,
    "independent_loss_after_sgd": independent_loss(after),
    "stale_policy_rejected": stale_error, "torch_version": torch.__version__,
    "scope": "Tensor loss, full joint reward credit, prompt/padding zeros, causal/EOS indexing and optimizer direction; no scientific model or training-quality claim."
}, indent=2))
