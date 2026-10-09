"""Paired fixed-instance sensitivity analyses; no training-seed population claim."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import beta


def read_predictions(path):
    rows = {}
    with Path(path).open() as handle:
        for line in handle:
            row = json.loads(line)
            uid = str(row['user_id'])
            if uid in rows:
                raise ValueError(f'duplicate user {uid}: {path}')
            ranking = row['ranking']
            if len(set(ranking)) != len(ranking):
                raise ValueError(f'duplicate ranked items: {uid}')
            rows[uid] = row
    if not rows:
        raise ValueError(f'empty predictions: {path}')
    return rows


def align(target, witness):
    if set(target) != set(witness):
        raise ValueError('paired user sets differ')
    users = sorted(target)
    for uid in users:
        if target[uid]['target_item_id'] != witness[uid]['target_item_id']:
            raise ValueError(f'paired target differs: {uid}')
    return users


def hits(rows, users):
    return np.array([rows[u]['target_item_id'] in rows[u]['ranking'][:10] for u in users], dtype=float)


def paired_audit(a, r, margin=0.05, resamples=10000, seed=2027, alpha=0.05):
    """Bootstrap the whole relative-margin statistic, including target uncertainty.

    Also report a conservative finite-sample bound derived from exact marginal
    binomial intervals for the three favorable/adverse paired cells. This is
    deliberately not a mislabeled exact McNemar non-inferiority test.
    """
    a, r = np.asarray(a, dtype=float), np.asarray(r, dtype=float)
    if a.ndim != 1 or a.shape != r.shape or not len(a):
        raise ValueError('require nonempty paired vectors')
    if not np.isin(a, [0, 1]).all() or not np.isin(r, [0, 1]).all():
        raise ValueError('R@10 must be binary per user')
    if not 0 <= margin < 1 or not 0 < alpha < 1 or resamples < 1:
        raise ValueError('invalid margin, alpha or resamples')
    n = len(a)
    cells = np.array([np.sum((a == 0) & (r == 1)), np.sum((a == 1) & (r == 0)),
                      np.sum((a == 1) & (r == 1)), np.sum((a == 0) & (r == 0))])
    # Multinomial resampling is exactly user-bootstrap resampling for binary pairs.
    draws = np.random.default_rng(seed).multinomial(n, cells / n, size=resamples)
    diff = (draws[:, 0] - draws[:, 1]) / n
    joint = (draws[:, 0] - (1 - margin) * draws[:, 1] + margin * draws[:, 2]) / n
    lcb = float(np.quantile(diff, alpha))
    joint_lcb = float(np.quantile(joint, alpha))
    # Simultaneous cell bounds via union bound; valid even with zero discordance.
    tail = alpha / 3
    lower = lambda k: 0.0 if k == 0 else float(beta.ppf(tail, k, n - k + 1))
    upper = lambda k: 1.0 if k == n else float(beta.ppf(1 - tail, k + 1, n - k))
    exact_lcb = lower(cells[0]) - (1 - margin) * upper(cells[1]) + margin * lower(cells[2])
    joint_mean = float(np.mean(r - (1 - margin) * a))
    # Hoeffding valid one-sided p bound for H0: E[R-(1-eta)A] <= 0.
    p_bound = float(np.exp(-2 * n * max(0, joint_mean)**2 / (2 - margin)**2))
    return {'n_users': n, 'U_A': float(a.mean()), 'U_R': float(r.mean()),
            'delta': float((r-a).mean()), 'lcb95': lcb,
            'r_min_relative': max(0.0, -lcb) / float(a.mean()) if a.mean() else None,
            'relative_margin': margin, 'absolute_margin_at_observed_target': float(margin*a.mean()),
            'joint_margin_mean': joint_mean, 'joint_margin_lcb': joint_lcb,
            'bootstrap_verdict': 'superiority' if lcb > 0 else 'recovery' if joint_lcb > 0 else 'unresolved',
            'exact_cell_bound_lcb': float(exact_lcb), 'exact_cell_bound_supports_recovery': bool(exact_lcb > 0),
            'hoeffding_p_bound': p_bound, 'paired_cells': dict(zip(
                ('witness_only_hit', 'target_only_hit', 'both_hit', 'neither_hit'), map(int, cells))),
            'assumption': 'iid evaluation users, conditional on fixed trained instances',
            'bootstrap': {'resamples': resamples, 'seed': seed, 'alpha': alpha},
            'robustness_method': 'simultaneous Clopper-Pearson cell bounds, conservative; not Tango/McNemar NI'}


def holm(pvalues, alpha=0.05):
    """Holm-adjusted p-values; include every planned test, not just successes."""
    if not pvalues:
        raise ValueError('empty family')
    if any(not 0 <= p <= 1 for p in pvalues.values()):
        raise ValueError('p-values outside [0,1]')
    running, output = 0.0, {}
    for i, (key, p) in enumerate(sorted(pvalues.items(), key=lambda kv: kv[1])):
        running = max(running, (len(pvalues)-i)*p)
        output[key] = {'adjusted_p_bound': min(1.0, running), 'reject': running <= alpha}
    return output


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()
