"""Audit an explicitly enumerated family; fail on missing comparisons."""
import argparse
import json
from pathlib import Path

from .paired_statistics import align, hits, holm, paired_audit, read_predictions, sha


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', required=True, help='JSON: comparisons [{id,target,witness}]')
    p.add_argument('--out', required=True)
    p.add_argument('--resamples', type=int, default=10000)
    args = p.parse_args()
    manifest = json.loads(Path(args.manifest).read_text())
    records = manifest['comparisons']
    if len({x['id'] for x in records}) != len(records):
        raise ValueError('duplicate comparison ids')
    output = {'analysis_role': 'retrospective sensitivity on existing test predictions',
              'family': manifest, 'comparisons': {}}
    for x in records:
        a, r = read_predictions(x['target']), read_predictions(x['witness'])
        users = align(a, r)
        row = paired_audit(hits(a, users), hits(r, users), margin=manifest.get('margin', 0.05),
                           resamples=args.resamples)
        row['sha256'] = {k: sha(x[k]) for k in ('target', 'witness')}
        output['comparisons'][x['id']] = row
    output['holm_family'] = holm({k: v['hoeffding_p_bound'] for k,v in output['comparisons'].items()})
    output['holm_note'] = 'Conservative finite-sample p bounds; not bootstrap p-values or a prospective preregistration.'
    dest = Path(args.out); dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open('x') as f:
        json.dump(output,f,indent=2); f.write('\n')


if __name__ == '__main__':
    main()
