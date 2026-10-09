"""Three-seed SASRec ensemble under the same exported score protocol."""
import argparse
from contextlib import ExitStack
import json
from pathlib import Path

import numpy as np

from .paired_statistics import sha
from .witness_ladder import normalize, evaluate


def main():
    p=argparse.ArgumentParser();p.add_argument('--cache',action='append',required=True)
    p.add_argument('--masked',action='store_true');p.add_argument('--out',required=True)
    args=p.parse_args()
    if len(args.cache)!=3 or len(set(args.cache))!=3:raise ValueError('three distinct seed caches required')
    roots=list(map(Path,args.cache));metas=[json.loads((r/'manifest.json').read_text()) for r in roots]
    if len({m['artifacts']['sasrec-model']['sha256'] for m in metas})!=3:raise ValueError('three distinct checkpoints required')
    for m in metas[1:]:
        for k in ('dataset','history_cap','item_ids','split_strategy'):
            if m[k]!=metas[0][k]:raise ValueError(f'ensemble mismatch: {k}')
        if m['artifacts']['interactions']['sha256']!=metas[0]['artifacts']['interactions']['sha256']:
            raise ValueError('ensemble interaction mismatch')
        for key in ('config', 'split_manifest', 'metadata_path'):
            if m['artifacts'].get(key)!=metas[0]['artifacts'].get(key):
                raise ValueError(f'ensemble data provenance mismatch: {key}')
        if len(m['splits']['test'])!=len(metas[0]['splits']['test']):raise ValueError('shards differ')
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False);hits,ndcg,total=0,0.,0
    with (out/'predictions.jsonl').open('x') as f:
        for parts in zip(*(m['splits']['test'] for m in metas)):
            with ExitStack() as stack:
                batches=[]
                for root,part in zip(roots,parts):
                    path=root/part['path']
                    if sha(path)!=part['sha256']:raise ValueError('ensemble shard hash mismatch')
                    batches.append(stack.enter_context(np.load(path)))
                first=batches[0];channels=[];eligibility=[]
                for b in batches:
                    for k in ('user_ids','targets','history_mask'):
                        if not np.array_equal(first[k],b[k]):raise ValueError(f'paired ensemble mismatch: {k}')
                    score,valid=normalize(b['scores'],b['history_mask'],args.masked)
                    channels.append(score[:,0,:]);eligibility.append(valid)
                if not all(np.array_equal(eligibility[0],v) for v in eligibility):raise ValueError('catalog eligibility mismatch')
                fused=np.mean(channels,axis=0);valid=eligibility[0]
                metrics=evaluate(fused,valid,first['targets'],20)
                hits+=int(metrics['hits'].sum());ndcg+=float(metrics['ndcg'].sum());total+=len(first['targets'])
                order=np.argsort(-np.where(valid,fused,-np.inf),axis=1,kind='stable')[:,:20]
                for i,u in enumerate(first['user_ids']):
                    f.write(json.dumps({'user_id':str(u),'target_item_id':metas[0]['item_ids'][int(first['targets'][i])],
                        'ranking':[metas[0]['item_ids'][int(j)] for j in order[i] if valid[i,j]]})+'\n')
    report={'R@10':hits/total,'N@10':ndcg/total,'n_users':total,'masked':args.masked,
            'history_cap':metas[0]['history_cap'],'method':'mean per-seed normalized SASRec score, no calibration',
            'independence_note':'one ensemble, not three independent ensemble runs',
            'cache_manifest_sha256':[sha(r/'manifest.json') for r in roots]}
    (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
