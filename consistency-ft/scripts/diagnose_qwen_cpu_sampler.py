#!/usr/bin/env python3
"""Replay only retained CPU RNG transitions; never load a model or emit answers."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import torch


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--collection', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--vocab-size', type=int, required=True)
    args = parser.parse_args()
    if args.output.exists() or not 2 <= args.vocab_size <= 1_000_000:
        parser.error('output must be new and vocabulary size bounded')
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    records = []
    inputs = {}
    for path in sorted((args.collection / 'attempts').glob('*/receipt.json')):
        raw = path.read_bytes()
        receipt = json.loads(raw)
        attempt_raw = path.with_name('attempt.json').read_bytes()
        attempt = json.loads(attempt_raw)
        events_raw = path.with_name('tokens.jsonl').read_bytes()
        events = [json.loads(line) for line in events_raw.splitlines()]
        assert receipt['id'] == attempt['id'] == path.parent.name
        assert len(events) == len(receipt['completion_token_ids'])
        assert len(events) == len(receipt['token_logprobs'])
        for name, content in [('receipt.json', raw), ('attempt.json', attempt_raw),
                              ('tokens.jsonl', events_raw)]:
            inputs[str(path.with_name(name).resolve())] = {
                'bytes': len(content), 'sha256': sha(content)}
        generator = torch.Generator(device='cpu').manual_seed(attempt['seed'])
        state_sha = lambda: sha(generator.get_state().numpy().tobytes())
        assert state_sha() == attempt['rng_before_sha256']
        for i, event in enumerate(events):
            token = receipt['completion_token_ids'][i]
            logprob = receipt['token_logprobs'][i]
            assert (event['token_index'], event['token_id'], event['sampled_logprob']) == (i, token, logprob)
            noise = torch.empty(args.vocab_size, dtype=torch.float32).exponential_(1.0, generator=generator)
            observed_sha = state_sha()
            assert observed_sha == event['rng_after_sha256'], (receipt['id'], i)
            smallest, indices = torch.topk(noise, k=2, largest=False)
            floors = (noise == torch.finfo(torch.float32).tiny).nonzero().flatten().tolist()
            records.append({
                'id': receipt['id'], 'catalog_index': attempt['catalog_index'],
                'seed': attempt['seed'], 'token_index': i, 'retained_token_id': token,
                'retained_logprob': logprob, 'readout_status': receipt['readout_status'],
                'rng_after_sha256': observed_sha, 'rng_after_matches_retained': True,
                'noise_sha256': sha(noise.numpy().tobytes()),
                'noise_at_retained_token': noise[token].item(),
                'smallest_noise': smallest.tolist(), 'smallest_noise_indices': indices.tolist(),
                'float32_floor_indices': floors, 'retained_token_has_floor_noise': token in floors,
            })
        assert state_sha() == receipt['rng_after_sha256']
    if not records:
        raise ValueError('no retained sampled tokens')
    report = {
        'recorded_at': datetime.now(timezone.utc).isoformat(),
        'scope': 'Deterministic software-only exponential-noise replay at retained RNG transitions; no model forward, benchmark request, response retry or new probability readout.',
        'source_sha256': sha(Path(__file__).read_bytes()),
        'torch_version': torch.__version__, 'torch_git_version': torch.version.git_version,
        'mkl_available': torch.backends.mkl.is_available(), 'dtype': 'float32',
        'vocab_size': args.vocab_size, 'cpu_threads': 2, 'interop_threads': 1,
        'attempts': len({row['id'] for row in records}), 'retained_token_transitions': len(records),
        'all_retained_rng_transitions_match': True,
        'transitions_with_float32_floor': sum(bool(row['float32_floor_indices']) for row in records),
        'selected_tokens_with_float32_floor': sum(row['retained_token_has_floor_noise'] for row in records),
        'input_artifacts': inputs, 'transitions': records,
        'limitations': 'Original full logits and probability vectors were not retained. Noise replay establishes the implementation trajectory, not model-logit replay or validity of a complete anchor.',
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        stream.write(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ('input_artifacts', 'transitions')}, indent=2))
    for row in records:
        if row['float32_floor_indices']:
            print(json.dumps(row))


if __name__ == '__main__':
    main()
