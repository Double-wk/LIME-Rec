"""Validation-only global witness selection; exploratory test surface is explicit."""
import argparse
from contextlib import ExitStack
import json
from pathlib import Path

import numpy as np
from scipy.special import logsumexp
from scipy.stats import rankdata

from .paired_statistics import sha


def normalize(raw, history, masked):
    eligible=(raw > -1e8).all(axis=1)
    if masked:eligible &= ~history
    if not eligible.any(axis=1).all():raise ValueError('empty eligible catalog')
    lo=np.where(eligible[:,None,:],raw,np.inf).min(2,keepdims=True)
    hi=np.where(eligible[:,None,:],raw,-np.inf).max(2,keepdims=True)
    return np.where(eligible[:,None,:],(raw-lo)/(hi-lo+1e-8),0),eligible


def simplex(step):
    if step < 1:raise ValueError('simplex divisions must be positive')
    return [(i/step,j/step,(step-i-j)/step) for i in range(step+1) for j in range(step-i+1)]


def evaluate(scores,eligible,targets,scale):
    scores=np.where(eligible,scores,-np.inf)
    if not eligible[np.arange(len(targets)),targets].all():
        raise ValueError('masked protocol excludes a target: report repeat-target rate before comparison')
    # Stable full-catalog item-index tie rule; item_ids are canonical sorted IDs.
    target_scores=scores[np.arange(len(targets)),targets]
    ranks=(scores>target_scores[:,None]).sum(1)
    ranks+=((scores==target_scores[:,None]) & (np.arange(scores.shape[1])[None,:]<targets[:,None])).sum(1)
    return {'hits':ranks<10,'ndcg':np.where(ranks<10,1/np.log2(ranks+2),0),
            'ce':logsumexp(scale*scores,axis=1)-scale*target_scores}


def configs(divisions,penalties):
    rows={}
    for name,w in [('sasrec',(1,0,0)),('itemcf',(0,1,0)),('semantic',(0,0,1)),
                   ('sasrec_itemcf',(.8,.2,0)),('sasrec_semantic',(.6,0,.4)),
                   ('fusion_nocal',(.60,.15,.25)),('equal_weight',(1/3,)*3)]:
        rows[name]={'kind':'weighted','weights':w,'penalty':0.0}
    for name in ('borda','rrf','combsum'):rows[name]={'kind':name,'penalty':0.0}
    for w in simplex(divisions):
        for penalty in penalties:
            name=f'grid_{w[0]:.3f}_{w[1]:.3f}_{w[2]:.3f}_p{penalty:g}'
            rows[name]={'kind':'weighted','weights':w,'penalty':penalty}
    return rows


def fused(config,scores,history,ranks):
    kind=config['kind']
    if kind=='weighted':value=np.einsum('e,bei->bi',config['weights'],scores)
    elif kind=='combsum':value=scores.sum(1)
    elif kind=='borda':value=-ranks.sum(1)
    else:value=(1/(60+ranks)).sum(1)
    return value-config['penalty']*history


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--cache',required=True);p.add_argument('--out',required=True)
    p.add_argument('--masked',action='store_true')
    p.add_argument('--divisions',type=int,default=10)
    p.add_argument('--penalties',default='0,0.05,0.10,0.20')
    p.add_argument('--scale',type=float,default=20)
    p.add_argument('--exploratory-test-surface',action='store_true')
    args=p.parse_args();root=Path(args.cache);meta=json.loads((root/'manifest.json').read_text())
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    candidates=configs(args.divisions,[float(x) for x in args.penalties.split(',')])
    summaries={};frozen=None
    for split in ('val','test'):
        names=list(candidates) if split=='val' or args.exploratory_test_surface else list(frozen['selected'])
        accum={name:{'hits':0,'ndcg':0.,'ce':0.,'users':0} for name in names}
        repeated,total=0,0
        with ExitStack() as stack:
            predictions={name:stack.enter_context((out/f'{name}_predictions.jsonl').open('x'))
                         for name in frozen['selected']} if split=='test' else {}
            for shard in meta['splits'][split]:
                path=root/shard['path']
                if sha(path)!=shard['sha256']:raise ValueError('score shard hash mismatch')
                with np.load(path) as data:
                    raw=data['scores'];history=data['history_mask'];targets=data['targets'];users=data['user_ids']
                    repeated+=int(history[np.arange(len(targets)),targets].sum());total+=len(targets)
                    scores,eligible=normalize(raw,history,args.masked)
                    ranks=np.stack([rankdata(-np.where(eligible,scores[:,e,:],-np.inf),axis=1,method='average') for e in range(3)],axis=1)
                    for name in names:
                        value=fused(candidates[name],scores,history,ranks)
                        metrics=evaluate(value,eligible,targets,args.scale)
                        acc=accum[name];acc['hits']+=int(metrics['hits'].sum());acc['ndcg']+=float(metrics['ndcg'].sum())
                        acc['ce']+=float(metrics['ce'].sum());acc['users']+=len(targets)
                        if split=='test' and name in frozen['selected']:
                            value=np.where(eligible,value,-np.inf)
                            order=np.argsort(-value,axis=1,kind='stable')[:,:20]
                            for i,uid in enumerate(users):
                                row={'user_id':str(uid),'target_item_id':meta['item_ids'][int(targets[i])],
                                     'ranking':[meta['item_ids'][int(j)] for j in order[i] if eligible[i,j]],'variant':name}
                                predictions[name].write(json.dumps(row)+'\n')
        summaries[split]={k:{'R@10':v['hits']/v['users'],'N@10':v['ndcg']/v['users'],
                             'CE':v['ce']/v['users'],'n_users':v['users']} for k,v in accum.items()}
        summaries[split+'_repeat_target_rate']=repeated/total
        if split=='val':
            grid=[k for k in candidates if k.startswith('grid_')]
            # Lowest complexity breaks validation ties: fewer components, lower penalty, then ID.
            best=min(grid,key=lambda k:(-summaries['val'][k]['R@10'],sum(w>0 for w in candidates[k]['weights']),
                                        abs(candidates[k]['penalty']),k))
            named=[k for k in candidates if not k.startswith('grid_')]
            subset_best={}
            for name,allow in [('sasrec_cal',[0]),('sasrec_itemcf_cal',[0,1]),('sasrec_semantic_cal',[0,2])]:
                options=[k for k in grid if all(w==0 or i in allow for i,w in enumerate(candidates[k]['weights']))]
                subset_best[name]=min(options,key=lambda k:(-summaries['val'][k]['R@10'],abs(candidates[k]['penalty']),k))
            selected=list(dict.fromkeys(named+[best]+list(subset_best.values())))
            frozen={'selection_split':'validation','objective':'R@10','selected':selected,'best':best,
                    'subset_best':subset_best,'configs':{k:candidates[k] for k in selected},
                    'cache_manifest_sha256':sha(root/'manifest.json'),'masked':args.masked}
            with (out/'frozen_validation_selection.json').open('x') as f:json.dump(frozen,f,indent=2);f.write('\n')
    report={'dataset':meta['dataset'],'protocol':{'history_cap':meta['history_cap'],'masked':args.masked},
            'selected':frozen,'search_log':candidates,'summaries':summaries,
            'test_surface_role':'exploratory only; never used for selection' if args.exploratory_test_surface else 'not computed',
            'fraction_at_selected_point_utility':float(np.mean([v['R@10']>=summaries['test'][frozen['best']]['R@10']
                for k,v in summaries['test'].items() if k.startswith('grid_')])) if args.exploratory_test_surface else None,
            'region_note':'fraction at least selected-witness point utility, NOT statistical recovery against any target'}
    with (out/'summary.json').open('x') as f:json.dump(report,f,indent=2);f.write('\n')


if __name__=='__main__':main()
