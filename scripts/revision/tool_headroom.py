"""Diagnostic hindsight oracle; labels never go to a deployable policy."""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from lime_rec.agentic.budgeted import TOOLS, fixed_policy, rank_view, rollout, utility
from scripts.agentic.run_budgeted_tools import load_data
from .paired_statistics import paired_audit, read_predictions, sha


def analyze(data, budget):
    sequences = list(itertools.combinations(TOOLS, budget))
    values, ndcg, coverage = [], [], []
    for record in data['rows']:
        target = data['targets'][record['user_id']]
        outcomes = [rollout(record, budget, fixed_policy(s)) for s in sequences]
        values.append([float(target in o['ranking'][:10]) for o in outcomes])
        ndcg.append([utility(o['ranking'], target) for o in outcomes])
        coverage.append([float(target in o['candidate_ids']) for o in outcomes])
    values, ndcg, coverage = map(np.asarray, (values,ndcg,coverage))
    return sequences, values, ndcg, coverage


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data', required=True, help='prepared test.json from budgeted tools')
    p.add_argument('--budget', type=int, choices=[1,2], required=True)
    p.add_argument('--fixed-tools', required=True, help='validation-frozen reference, e.g. sasrec,itemcf')
    p.add_argument('--controller-predictions', action='append', default=[])
    p.add_argument('--out', required=True)
    args=p.parse_args();data=load_data(args.data)
    if data['split'] != 'test': raise ValueError('headroom is a test-only diagnostic')
    sequences, values, ndcg, coverage=analyze(data,args.budget)
    tools=tuple(args.fixed_tools.split(','))
    if len(set(tools))!=args.budget or set(tools)-set(TOOLS): raise ValueError('invalid fixed tools')
    reference=next(i for i,s in enumerate(sequences) if set(s)==set(tools))
    oracle=values.max(1);fixed=values[:,reference]
    out={'role':'hindsight bound, not a deployable or validation-selected policy',
         'data_sha256':sha(args.data),'identity':data['identity'],'budget':args.budget,
         'fixed_tools':list(tools),'fixed_reference_source':'must be frozen on validation before test',
         'n_users':len(values),'oracle_R@10':float(oracle.mean()),
         'oracle_N@10':float(ndcg.max(1).mean()),
         'oracle_candidate_coverage':float(coverage.max(1).mean()),
         'fixed_R@10':float(fixed.mean()),'fixed_N@10':float(ndcg[:,reference].mean()),
         'headroom_absolute_R@10':float((oracle-fixed).mean()),
         'oracle_minus_fixed':paired_audit(fixed,oracle,margin=0), 'controllers':{},
         'tool_sets':{','.join(s):{'R@10':float(values[:,i].mean()),'N@10':float(ndcg[:,i].mean()),
                                 'candidate_coverage':float(coverage[:,i].mean())} for i,s in enumerate(sequences)}}
    for path in args.controller_predictions:
        rows=read_predictions(path)
        if set(rows)!=set(data['targets']): raise ValueError('controller user mismatch')
        hits=[]; valid=[]
        for record in data['rows']:
            uid=record['user_id'];r=rows[uid];target=data['targets'][uid]
            if r.get('target_item_id',target)!=target: raise ValueError('controller target mismatch')
            if r.get('budget')!=args.budget: raise ValueError('controller budget mismatch')
            hits.append(target in r['ranking'][:10]);valid.append(not r['format_failure'])
        hits=np.asarray(hits,dtype=float);valid=np.asarray(valid)
        row={'sha256':sha(path),'oracle_minus_controller':paired_audit(hits,oracle,margin=0),
             'fixed_minus_controller':paired_audit(hits,fixed),'format_failure_rate':float(1-valid.mean()),
             'valid_subset_users':int(valid.sum())}
        if valid.any():row['valid_subset_fixed_minus_controller']=paired_audit(hits[valid],fixed[valid])
        out['controllers'][path]=row
    dest=Path(args.out);dest.parent.mkdir(parents=True,exist_ok=True)
    with dest.open('x') as f:json.dump(out,f,indent=2);f.write('\n')


if __name__=='__main__':main()
