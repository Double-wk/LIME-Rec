"""Recompute Table 4's two-sided CIs from frozen paired user predictions.

Run from lime-rec: python3 scripts/agentic/table4_confidence_intervals.py
The original one-sided recovery-audit artifact is retained unchanged.
"""
from pathlib import Path
import hashlib
import json
import zlib

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'output_final/results/recovery_battery/table_agent_audit.json'
OUT = ROOT / 'output_final/results/recovery_battery/table4_two_sided_ci.json'


def load(relative):
    path = ROOT / relative
    records = [json.loads(line) for line in path.read_text().splitlines() if line]
    rows = {row['user_id']: row for row in records}
    assert len(rows) == len(records), f'duplicate users: {path}'
    return rows, {'path': relative, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def hits(rows, users):
    return np.array([float(rows[u]['target_rank'] is not None and
                           rows[u]['target_rank'] < 10) for u in users])


def interval(witness, controller, users, scope, label):
    assert all(witness[u]['target_item_id'] == controller[u]['target_item_id'] for u in users)
    w, c = hits(witness, users), hits(controller, users)
    diff = w - c
    rng = np.random.default_rng([2027, zlib.crc32(f'{scope}|{label}'.encode())])
    boot = diff[rng.integers(0, len(users), size=(10000, len(users)))].mean(axis=1)
    return {'n_users': len(users), 'deterministic_r10': float(w.mean()),
            'controller_r10': float(c.mean()), 'delta': float(diff.mean()),
            'ci95': np.percentile(boot, [2.5, 97.5]).tolist(),
            'lcb95_check': float(np.percentile(boot, 5)),
            'rng_scope': scope, 'rng_label': label}


def main():
    audit = json.loads(SOURCE.read_text())
    witness, witness_source = load(audit['witness']['path'])
    users = sorted(witness)
    assert len(users) == 1000
    result = {'method': 'paired user-level percentile bootstrap, two-sided 95%',
              'resamples': 10000, 'seed': 2027, 'metric': 'R@10',
              'sign': 'deterministic minus controller',
              'rng': 'default_rng([seed, crc32(scope + "|" + label)])',
              'witness_source': witness_source,
              'original_audit_source': {'path': str(SOURCE.relative_to(ROOT)),
                  'sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest()},
              'panel_a': {}, 'panel_b': {}}
    for label, original in audit['conditions'].items():
        if label == 'Qwen3-4B forced-observe-all':
            continue  # This condition is absent from Table 4.
        controller, source = load(original['path'])
        assert sorted(controller) == users, label
        cell = interval(witness, controller, users, 'overall', label)
        for key, old_key in [('deterministic_r10', 'U_R'), ('controller_r10', 'U_A'),
                             ('delta', 'delta'), ('lcb95_check', 'lcb95')]:
            assert np.isclose(cell[key], original[old_key], atol=1e-12, rtol=0), (label, key)
        cell['controller_source'] = source
        result['panel_a'][label] = cell
        if label == 'Granite-3.3-8B adaptive':
            valid = [u for u in users if not controller[u]['format_failure'] and controller[u]['ranking']]
            assert len(valid) == 777
            cell_b = interval(witness, controller, valid, 'valid-output', label)
            cell_b['controller_source'] = source
            result['panel_b'][label] = cell_b
    assert len(result['panel_a']) == 9
    OUT.write_text(json.dumps(result, indent=2) + '\n')
    for panel in ('panel_a', 'panel_b'):
        for label, cell in result[panel].items():
            low, high = cell['ci95']
            print(f'{panel} {label}: delta={cell["delta"]:+.3f}, CI=[{low:+.3f}, {high:+.3f}]')


if __name__ == '__main__':
    main()
