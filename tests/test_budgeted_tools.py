import copy
import json
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest

from lime_rec.agentic.budgeted import (PROTOCOL, TOOLS, BudgetedEnvironment, LLMSelector,
    LinearRouter, features, fixed_policy, messages, rank_view, rollout, training_states)
from scripts.agentic.run_budgeted_tools import evaluate, fit, paired


def record(uid='u'):
    return {'user_id': uid, 'history': ['old'], 'tools': {
        t: [{'item_id': f'{t}_{i}', 'score': 1 - i / 20} for i in range(20)] for t in TOOLS}}


def data(split='validation'):
    rows = [record(str(i)) for i in range(6)]
    return {'protocol': PROTOCOL, 'identity': {'dataset': 'fixture', 'top_k': 20},
            'split': split, 'rows': rows,
            'targets': {r['user_id']: f'{TOOLS[i % 3]}_0' for i, r in enumerate(rows)}}


def test_hidden_evidence_and_mutation_isolation():
    row = record()
    env = BudgetedEnvironment(row, 2)
    view = env.view()
    assert view['observations'] == {}
    assert 'sasrec_0' not in json.dumps(view)
    assert 'user_id' not in view and 'target_item_id' not in view and 'candidate_ids' not in view
    env.call('sasrec')
    view = env.view()
    assert set(view['observations']) == {'sasrec'}
    assert 'semantic_0' not in json.dumps(view)
    mutated = copy.deepcopy(row)
    mutated['tools']['semantic'][0]['score'] = 999
    other = BudgetedEnvironment(mutated, 2)
    other.call('sasrec')
    assert other.view() == view
    np.testing.assert_array_equal(features(other.view()), features(view))
    view['observations']['sasrec'][0]['score'] = 999
    assert env.view()['observations']['sasrec'][0]['score'] == 1


def test_budget_and_duplicate_guards():
    env = BudgetedEnvironment(record(), 2)
    with pytest.raises(ValueError):
        env.call('unknown')
    env.call('sasrec')
    with pytest.raises(ValueError):
        env.call('sasrec')
    env.call('semantic')
    with pytest.raises(ValueError, match='budget'):
        env.call('itemcf')
    assert env.view()['remaining_tool_budget'] == 0


def test_fusion_uses_only_revealed_candidates_and_history_penalty():
    row = record()
    row['history'] = ['old']
    row['tools']['sasrec'] = [{'item_id': 'old', 'score': 1.0}, {'item_id': 'new', 'score': .95}]
    result = rollout(row, 1, fixed_policy(('sasrec',)))
    assert result['ranking'] == ['new', 'old']
    assert result['candidate_ids'] == ['new', 'old']
    assert len(result['tool_calls']) == 1


def test_invalid_llm_action_is_failure_without_fallback():
    class Client:
        def generate(self, *args, **kwargs):
            return {'text': '{"tool":"hidden"}', 'usage': {'prompt_tokens': 4}}
    policy = LLMSelector(Client())
    result = rollout(record(), 2, policy)
    assert result['format_failure'] and result['ranking'] == []
    assert result['tool_calls'] == [] and len(policy.events) == 1


def test_router_validation_only_and_public_inputs():
    with pytest.raises(ValueError, match='validation'):
        LinearRouter.fit(data('test'))
    router = LinearRouter.fit(data())
    for budget in (1, 2):
        r = rollout(record(), budget, router)
        assert not r['format_failure']
        assert len(set(r['tool_calls'])) == budget
    state = copy.deepcopy(router.state)
    state['protocol']['version'] = 'different'
    with pytest.raises(ValueError):
        LinearRouter(state)


def test_fit_sft_and_inference_messages_match(tmp_path):
    source = tmp_path / 'validation.json'
    source.write_text(json.dumps(data()))
    out = tmp_path / 'fit'
    fit(SimpleNamespace(data=str(source), ridge=1.0, out_dir=str(out)))
    sft = [json.loads(l) for l in (out / 'selector_sft.jsonl').read_text().splitlines()]
    assert sft
    for row in sft:
        view = json.loads(row['prompt'])
        assert messages(view) == [{'role': 'system', 'content': row['system']},
                                  {'role': 'user', 'content': row['prompt']}]
        assert json.loads(row['completion'])['tool'] in view['available_tools']
        assert 'target' not in view and 'tools' not in view
    with pytest.raises(FileExistsError):
        fit(SimpleNamespace(data=str(source), ridge=1.0, out_dir=str(out)))


def test_training_states_and_targets_never_enter_prompt():
    states = list(training_states(record(), 'semantic_0'))
    assert len(states) == 5
    view, rewards = states[0]
    assert rewards['semantic'] == 1.0 and rewards['sasrec'] == 0
    assert 'semantic_0' not in messages(view)[1]['content']


def test_evaluate_cli_keeps_all_users_and_marks_completion(tmp_path):
    valid, test = tmp_path / 'valid.json', tmp_path / 'test.json'
    valid.write_text(json.dumps(data()))
    heldout = data('test')
    heldout['targets']['0'] = 'unreachable'
    test.write_text(json.dumps(heldout))
    fitted = tmp_path / 'fit'
    fit(SimpleNamespace(data=str(valid), ridge=1., out_dir=str(fitted)))
    args = SimpleNamespace(data=str(test), router=str(fitted / 'router.json'),
        llm=False, llm_artifact=None, max_tokens=128, bootstrap_resamples=20, seed=2027,
        out_dir=str(tmp_path / 'evaluation'))
    evaluate(args)
    summary = json.loads((tmp_path / 'evaluation/summary.json').read_text())
    assert len(summary) == 12
    assert all(v['users'] == 6 for v in summary.values())
    assert summary['all_tools_reference']['mean_tool_calls'] == 3
    assert summary['router_b2']['mean_tool_calls'] == 2
    assert summary['all_tools_reference']['R@10'] <= 5/6
    assert (tmp_path / 'evaluation/COMPLETED.json').exists()
    args.out_dir = str(tmp_path / 'bad')
    heldout['identity']['top_k'] = 30
    test.write_text(json.dumps(heldout))
    with pytest.raises(ValueError, match='identities'):
        evaluate(args)
    assert not (tmp_path / 'bad').exists()


def test_paired_intervals_preserve_user_pairing():
    a = [{'user_id': str(i), 'hit10': 1, 'ndcg10': 1} for i in range(4)]
    b = [{'user_id': str(i), 'hit10': 0, 'ndcg10': 0} for i in range(4)]
    assert paired(a, b, 0, 20)['hit10']['ci95'] == [1., 1.]
    with pytest.raises(ValueError):
        paired(a, b[::-1], 0, 20)


def test_llm_evaluation_and_paired_output(tmp_path, monkeypatch):
    from lime_rec.agentic.llm_client import OpenAICompatibleClient
    class Client:
        model_name = 'fixture-selector'
        def generate(self, payload, **kwargs):
            view = json.loads(payload[1]['content'])
            assert 'targets' not in view and 'candidate_ids' not in view
            return {'text': json.dumps({'tool': view['available_tools'][0]}),
                    'usage': {'prompt_tokens': 7, 'completion_tokens': 3}}
    monkeypatch.setattr(OpenAICompatibleClient, 'from_environment', lambda: Client())
    valid, test = tmp_path / 'valid.json', tmp_path / 'test.json'
    valid.write_text(json.dumps(data()))
    test.write_text(json.dumps(data('test')))
    fitted = tmp_path / 'fit'
    fit(SimpleNamespace(data=str(valid), ridge=1., out_dir=str(fitted)))
    out = tmp_path / 'evaluation'
    evaluate(SimpleNamespace(data=str(test), router=str(fitted / 'router.json'),
        llm=True, llm_artifact=None, max_tokens=128, bootstrap_resamples=20, seed=2027,
        out_dir=str(out)))
    summary = json.loads((out / 'summary.json').read_text())
    assert len(summary) == 14
    assert summary['llm_b2']['mean_llm_calls'] == 2
    assert summary['llm_b2']['mean_input_tokens'] == 14
    comparisons = json.loads((out / 'paired_comparisons.json').read_text())
    assert len(comparisons) == 11
    assert comparisons['fixed_sasrec_b1_minus_llm_b1']['hit10']['difference'] == 0


def test_prepare_rechecks_live_scores_and_separates_labels(tmp_path, monkeypatch):
    from lime_rec.evaluation import ExpertEvaluator
    from lime_rec.agentic.candidates import build_candidate_pool
    from lime_rec.agentic.tools import score_hash
    from scripts.agentic.run_budgeted_tools import prepare
    items = [f'i{i:02}' for i in range(24)]
    dataset = SimpleNamespace(name='fixture', item_ids=items,
        history_by_user={'u': [items[0]]}, valid_by_user={'u': items[1]}, test_by_user={'u': items[2]})
    scores = np.array([np.linspace(0, 1, 24), np.linspace(1, 0, 24),
                       np.roll(np.linspace(0, 1, 24), 5)], dtype=np.float32)
    evaluator = SimpleNamespace(dataset=dataset,
        sasrec=SimpleNamespace(item_index={item: i for i, item in enumerate(items)}),
        normalized_scores=lambda user, history: scores)
    monkeypatch.setattr(ExpertEvaluator, 'from_paths', lambda *a, **k: evaluator)
    artifacts = {}
    for name in ('sasrec', 'itemcf', 'semantic'):
        artifact = tmp_path / name
        artifact.write_text('fixture')
        artifacts[name] = str(artifact)
    interactions = tmp_path / 'interactions'
    interactions.write_text('fixture')
    config = tmp_path / 'config.json'
    config.write_text(json.dumps({'name': 'fixture', 'interactions_path': str(interactions)}))
    pool = build_candidate_pool('u', items, dict(zip(TOOLS, scores)), 20)
    restricted = scores[:, [items.index(i) for i in pool.item_ids]]
    row = {'user_id': 'u', 'split': 'validation', 'history': [items[0]],
           'candidate_ids': list(pool.item_ids), 'candidate_hash': pool.candidate_hash,
           'expert_scores': restricted.tolist(), 'expert_score_hashes': [score_hash(v) for v in restricted]}
    cache = tmp_path / 'cache.jsonl'
    cache.write_text(json.dumps(row) + '\n')
    Path(str(cache) + '.metadata.json').write_text(json.dumps({'split': 'validation',
        'candidate_per_expert': 20, 'dataset': 'fixture', 'artifacts': artifacts, 'num_users': 1}))
    args = SimpleNamespace(out=str(tmp_path / 'prepared.json'), config=str(config),
        candidate_cache=str(cache), split='validation', top_k=20, device='cpu',
        max_users=None, sample_seed=2027)
    prepare(args)
    prepared = json.loads((tmp_path / 'prepared.json').read_text())
    assert prepared['targets'] == {'u': items[1]}
    assert 'target' not in prepared['rows'][0]
    assert set(prepared['rows'][0]) == {'user_id', 'history', 'tools'}
    assert prepared['live_cache_equivalence_checked']
    # Self-consistent tampering of cached values still fails live equivalence.
    row['expert_scores'][0][0] += .2
    row['expert_score_hashes'] = [score_hash(v) for v in row['expert_scores']]
    cache.write_text(json.dumps(row) + '\n')
    args.out = str(tmp_path / 'rejected.json')
    with pytest.raises(ValueError, match='live expert scores'):
        prepare(args)
    assert not (tmp_path / 'rejected.json').exists()
