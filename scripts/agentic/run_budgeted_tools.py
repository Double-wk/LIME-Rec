"""Isolated prepare/fit/evaluate workflow for budgeted expert acquisition."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from lime_rec.agentic.budgeted import (PROTOCOL, TOOLS, LinearRouter, LLMSelector,
                                      fixed_policy, fixed_sequences, messages,
                                      rollout, training_states)
from lime_rec.agentic.metrics import evaluate_agentic


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8') as handle:
        json.dump(value, handle, indent=2)
        handle.write('\n')


def load_data(path):
    data = json.loads(Path(path).read_text())
    if data['protocol'] != PROTOCOL:
        raise ValueError('prepared protocol differs from runtime protocol')
    users = [r['user_id'] for r in data['rows']]
    if not users or len(users) != len(set(users)) or set(users) != set(data['targets']):
        raise ValueError('empty, duplicate, or unpaired users')
    return data


def prepare(args):
    # Import model dependencies only for preparation, never for router evaluation.
    from lime_rec.evaluation import ExpertEvaluator
    from lime_rec.protocols import prediction_context, evaluation_label
    from lime_rec.agentic.candidates import build_candidate_pool
    from lime_rec.agentic.tools import score_hash

    out = Path(args.out)
    if out.exists():
        raise FileExistsError(out)
    source = Path(args.candidate_cache)
    meta = json.loads(Path(str(source) + '.metadata.json').read_text())
    if meta['split'] != args.split or args.top_k > meta['candidate_per_expert'] or args.top_k < 10:
        raise ValueError('split mismatch or top-k outside [10, cached per-expert size]')
    artifacts = meta['artifacts']
    config = json.loads(Path(args.config).read_text())
    identity = {'dataset': meta['dataset'], 'config_sha256': sha(args.config),
                'top_k': args.top_k, 'artifact_sha256': {k: sha(v) for k, v in artifacts.items()},
                'interactions_sha256': sha(config['interactions_path']),
                'metadata_sha256': sha(config['metadata_path']) if config.get('metadata_path') else None}
    evaluator = ExpertEvaluator.from_paths(args.config, artifacts['sasrec'], artifacts['itemcf'],
                                           artifacts['semantic'], device=args.device, mask_history=False)
    dataset = evaluator.dataset
    if dataset.name != meta['dataset']:
        raise ValueError('dataset mismatch')
    raw_rows = [json.loads(l) for l in source.read_text().splitlines() if l.strip()]
    if len(raw_rows) != meta['num_users'] or len({r['user_id'] for r in raw_rows}) != len(raw_rows):
        raise ValueError('cache has duplicate users or incorrect row count')
    if args.max_users is not None:
        if args.max_users < 1:
            raise ValueError('max-users must be positive')
        indices = np.random.default_rng(args.sample_seed).permutation(len(raw_rows))[:args.max_users]
        raw_rows = [raw_rows[i] for i in sorted(indices)]
    rows, targets = [], {}
    for base in raw_rows:
        uid = base['user_id']
        context = prediction_context(dataset, uid, args.split)
        if base['split'] != args.split or list(context.history) != base['history']:
            raise ValueError(f'cache split/history mismatch: {uid}')
        # Recheck both pool generation and scores against the actual supplied experts.
        # Existing cache metadata hashes paths, not model bytes; this closes that gap.
        live = evaluator.normalized_scores(uid, context.history)
        pool = build_candidate_pool(uid, dataset.item_ids, dict(zip(TOOLS, live)),
                                    meta['candidate_per_expert'])
        if list(pool.item_ids) != base['candidate_ids'] or pool.candidate_hash != base['candidate_hash']:
            raise ValueError(f'candidate pool mismatch: {uid}')
        indices = [evaluator.sasrec.item_index[i] for i in base['candidate_ids']]
        cached = np.asarray(base['expert_scores'], dtype=np.float32)
        restricted = np.stack([v[indices] for v in live]).astype(np.float32)
        if cached.shape != restricted.shape or not np.isfinite(cached).all():
            raise ValueError(f'invalid cached scores: {uid}')
        if [score_hash(v) for v in cached] != base['expert_score_hashes']:
            raise ValueError(f'cache score hash mismatch: {uid}')
        if not np.allclose(cached, restricted, rtol=1e-5, atol=1e-6):
            raise ValueError(f'live expert scores differ from cache: {uid}')
        tools = {}
        for name, values in zip(TOOLS, cached):
            order = sorted(range(len(values)), key=lambda i: (-float(values[i]), base['candidate_ids'][i]))
            tools[name] = [{'item_id': base['candidate_ids'][i], 'score': float(values[i])}
                           for i in order[:args.top_k]]
        rows.append({'user_id': uid, 'history': base['history'], 'tools': tools})
        targets[uid] = evaluation_label(dataset, uid, args.split).target_item_id
    if not rows:
        raise ValueError('empty cache')
    out.parent.mkdir(parents=True, exist_ok=True)
    write_json(out, {'protocol': PROTOCOL, 'identity': identity, 'split': args.split,
                     'source_cache_sha256': sha(source), 'live_cache_equivalence_checked': True,
                     'sample_seed': args.sample_seed, 'rows': rows, 'targets': targets})
    print(f'Prepared {len(rows)} {args.split} users -> {out}', flush=True)


def fit(args):
    data = load_data(args.data)
    if data['split'] != 'validation':
        raise ValueError('fit requires validation data')
    out = Path(args.out_dir)
    if out.exists():
        raise FileExistsError(out)
    router = LinearRouter.fit(data, args.ridge)
    router.state['validation_sha256'] = sha(args.data)
    out.mkdir(parents=True)
    write_json(out / 'router.json', router.state)
    emitted = skipped = 0
    with (out / 'selector_sft.jsonl').open('x') as handle:
        for row in data['rows']:
            for view, rewards in training_states(row, data['targets'][row['user_id']]):
                ordered = sorted(rewards, key=lambda t: -rewards[t])
                # Ambiguous/no-signal states must not teach an arbitrary tool preference.
                if rewards[ordered[0]] == rewards[ordered[1]]:
                    skipped += 1
                    continue
                prompt = messages(view)
                handle.write(json.dumps({'system': prompt[0]['content'], 'prompt': prompt[1]['content'],
                    'completion': json.dumps({'tool': ordered[0]}), 'user_id': row['user_id']}) + '\n')
                emitted += 1
    write_json(out / 'fit_manifest.json', {'protocol': PROTOCOL, 'fit_split': 'validation',
        'identity': data['identity'], 'validation_sha256': sha(args.data),
        'router_sha256': sha(out / 'router.json'), 'sft_sha256': sha(out / 'selector_sft.jsonl'),
        'sft_examples': emitted, 'sft_tied_states_skipped': skipped,
        'supervision': 'held-out validation target; greedy next-step NDCG@10'})
    print(f'Router fitted; {emitted} SFT examples ({skipped} tied states skipped).', flush=True)


def paired(left, right, seed, resamples):
    if [r['user_id'] for r in left] != [r['user_id'] for r in right]:
        raise ValueError('paired comparison users differ')
    rng = np.random.default_rng(seed)
    result = {}
    for metric in ('hit10', 'ndcg10'):
        diff = np.array([a[metric] - b[metric] for a, b in zip(left, right)])
        samples = [float(rng.choice(diff, len(diff), replace=True).mean()) for _ in range(resamples)]
        result[metric] = {'difference': float(diff.mean()),
                          'ci95': np.quantile(samples, [0.025, 0.975]).tolist(),
                          'lcb95_one_sided': float(np.quantile(samples, 0.05))}
    return result


def evaluate(args):
    data = load_data(args.data)
    if data['split'] != 'test':
        raise ValueError('evaluation requires test data')
    router = LinearRouter(json.loads(Path(args.router).read_text()))
    if data['identity'] != router.state['identity']:
        raise ValueError('test and fitted router artifact/protocol identities differ')
    if args.bootstrap_resamples < 1 or args.max_tokens < 1:
        raise ValueError('resamples and max-tokens must be positive')
    out = Path(args.out_dir)
    if out.exists():
        raise FileExistsError(out)
    selector = None
    if args.llm:
        from lime_rec.agentic.llm_client import OpenAICompatibleClient
        if getattr(args, 'json_schema', False):
            from scripts.revision.strict_controller import SchemaClient, RetryingSelector
            selector = RetryingSelector(SchemaClient.from_environment(), args.max_tokens,
                                        args.format_retries)
        else:
            selector = LLMSelector(OpenAICompatibleClient.from_environment(), args.max_tokens)
    out.mkdir(parents=True)
    manifest = {'protocol': PROTOCOL, 'identity': data['identity'],
        'test_sha256': sha(args.data), 'router_sha256': sha(args.router),
        'code_sha256': sha(__file__), 'validation_sha256': router.state['validation_sha256'],
        'llm_model': selector.client.model_name if selector else None,
        'llm_artifact_sha256': sha(args.llm_artifact) if args.llm_artifact else None,
        'budgets': [1, 2], 'temperature': 0.0, 'max_tokens': args.max_tokens,
        'bootstrap_seed': args.seed, 'bootstrap_resamples': args.bootstrap_resamples,
        'timing_scope': 'cache replay wall time includes LLM calls; expert inference excluded',
        'comparison_scope': 'paired exploratory intervals, no multiplicity correction',
        'completion_marker': 'COMPLETED.json'}
    if getattr(args, 'json_schema', False):
        from scripts.revision import strict_controller
        manifest['decoding_variant'] = {'json_schema': True, 'format_retries': args.format_retries,
                                        'code_sha256': sha(strict_controller.__file__)}
    # Record the environment implementation too, not just this CLI.
    import lime_rec.agentic.budgeted as implementation
    manifest['environment_sha256'] = sha(implementation.__file__)
    write_json(out / 'manifest.json', manifest)
    conditions = [('all_tools_reference', 3, fixed_policy(TOOLS))]
    for budget in (1, 2):
        conditions.extend((f'fixed_{"_".join(seq)}_b{budget}', budget, fixed_policy(seq))
                          for seq in fixed_sequences(budget))
        conditions.append((f'router_b{budget}', budget, router))
        if selector:
            conditions.append((f'llm_b{budget}', budget, selector))
    outputs, summary = {}, {}
    from lime_rec.agentic.budgeted import utility
    for name, budget, policy in conditions:
        records = []
        with (out / f'{name}.jsonl').open('x') as handle:
            for row in data['rows']:
                if isinstance(policy, LLMSelector):
                    policy.events.clear()
                result = rollout(row, budget, policy)
                target = data['targets'][row['user_id']]
                events = list(policy.events) if isinstance(policy, LLMSelector) else []
                result.update({'condition': name, 'target_item_id': target,
                    'hit10': float(target in result['ranking'][:10]),
                    'ndcg10': utility(result['ranking'], target), 'llm_events': events,
                    'llm_calls': len(events),
                    'llm_request_seconds': sum(e['latency_seconds'] for e in events),
                    'input_tokens': sum(e['usage'].get('prompt_tokens', 0) for e in events),
                    'output_tokens': sum(e['usage'].get('completion_tokens', 0) for e in events)})
                handle.write(json.dumps(result) + '\n')
                handle.flush()
                records.append(result)
        outputs[name] = records
        metrics = evaluate_agentic(records)
        for field in ('cache_replay_wall_seconds', 'llm_request_seconds', 'input_tokens', 'output_tokens', 'llm_calls'):
            metrics['mean_' + field] = float(np.mean([r[field] for r in records]))
        metrics['mean_tool_calls'] = float(np.mean([len(r['tool_calls']) for r in records]))
        metrics['users'] = len(records)
        summary[name] = metrics
        print(name, json.dumps(metrics), flush=True)
    comparisons = {}
    if selector:
        for budget in (1, 2):
            target = outputs[f'llm_b{budget}']
            for name, rows in outputs.items():
                if name.endswith(f'_b{budget}') and not name.startswith('llm'):
                    comparisons[f'{name}_minus_llm_b{budget}'] = paired(rows, target, args.seed, args.bootstrap_resamples)
    write_json(out / 'summary.json', summary)
    write_json(out / 'paired_comparisons.json', comparisons)
    write_json(out / 'COMPLETED.json', {'users': len(data['rows']), 'conditions': list(summary)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare')
    p.add_argument('--config', required=True)
    p.add_argument('--candidate-cache', required=True)
    p.add_argument('--split', choices=['validation', 'test'], required=True)
    p.add_argument('--top-k', type=int, default=20)
    p.add_argument('--max-users', type=int)
    p.add_argument('--sample-seed', type=int, default=2027)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    p.add_argument('--out', required=True)
    p.set_defaults(function=prepare)
    p = sub.add_parser('fit')
    p.add_argument('--data', required=True)
    p.add_argument('--ridge', type=float, default=1.0)
    p.add_argument('--out-dir', required=True)
    p.set_defaults(function=fit)
    p = sub.add_parser('evaluate')
    p.add_argument('--data', required=True)
    p.add_argument('--router', required=True)
    p.add_argument('--llm', action='store_true')
    p.add_argument('--json-schema', action='store_true', help='new constrained-decoding variant; requires server schema support')
    p.add_argument('--format-retries', type=int, default=2)
    p.add_argument('--llm-artifact', help='Local model/training manifest to hash for provenance')
    p.add_argument('--max-tokens', type=int, default=128)
    p.add_argument('--bootstrap-resamples', type=int, default=10000)
    p.add_argument('--seed', type=int, default=2027)
    p.add_argument('--out-dir', required=True)
    p.set_defaults(function=evaluate)
    args = parser.parse_args()
    args.function(args)


if __name__ == '__main__':
    main()
