"""Exploratory paired error profiles using train-derived strata."""
import argparse
import json
from pathlib import Path

import numpy as np

from lime_rec.data import load_dataset
from .paired_statistics import align, hits, paired_audit, read_predictions, sha


def main():
    p = argparse.ArgumentParser()
    for key in ('config', 'target', 'witness', 'out'):
        p.add_argument('--' + key, required=True)
    args = p.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    ds = load_dataset(cfg['name'], cfg['interactions_path'], cfg.get('metadata_path'),
                      cfg.get('min_user_interactions', 5), cfg.get('min_item_interactions', 5),
                      split_manifest=cfg.get('split_manifest'))
    a, r = read_predictions(args.target), read_predictions(args.witness)
    users = align(a, r)
    if not set(users) <= set(ds.test_by_user):
        raise ValueError('prediction users outside configured test split')
    ah, rh = hits(a, users), hits(r, users)
    cuts = np.quantile(list(ds.item_popularity.values()), [1/3, 2/3])
    strata = {'all': [], 'history_1_4': [], 'history_5_7': [],
              'history_8_20': [], 'history_over20': [], 'repeat': [], 'novel': [],
              'popularity_low': [], 'popularity_mid': [], 'popularity_high': []}
    for i, u in enumerate(users):
        target = ds.test_by_user[u]
        if a[u]['target_item_id'] != target:
            raise ValueError('prediction target differs from configured split')
        history = ds.history_by_user[u] + [ds.valid_by_user[u]]
        h = len(history)
        bucket = 'history_1_4' if h <= 4 else 'history_5_7' if h <= 7 else 'history_8_20' if h <= 20 else 'history_over20'
        pop = ds.item_popularity.get(target, 0)
        popularity = 'popularity_low' if pop <= cuts[0] else 'popularity_mid' if pop <= cuts[1] else 'popularity_high'
        for key in ('all', bucket, 'repeat' if target in history else 'novel', popularity):
            strata[key].append(i)
    report = {'role': 'exploratory subgroup analysis; no corrected subgroup claims',
              'population': 'paired prediction users only', 'popularity_cutoffs': cuts.tolist(),
              'history_rule': 'full train history plus validation item, before any serving cap',
              'repeat_target_rate': len(strata['repeat']) / len(users),
              'sha256': {k: sha(getattr(args, k)) for k in ('config', 'target', 'witness')},
              'profiles': {k: paired_audit(ah[ix], rh[ix]) if ix else {'n_users': 0}
                           for k, ix in strata.items()}}
    dest = Path(args.out); dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open('x') as f:
        json.dump(report, f, indent=2); f.write('\n')


if __name__ == '__main__':
    main()
