"""Export batch shards for isolated matched-history witness experiments."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from lime_rec.evaluation import ExpertEvaluator
from lime_rec.protocols import truncate_history
from scripts.evaluation.run_repeat_aware_gate import collect_examples
from .paired_statistics import sha


def main():
    p=argparse.ArgumentParser()
    for key in ('config','sasrec-model','itemcf-model','semantic-emb','out'):p.add_argument('--'+key,required=True)
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    p.add_argument('--history-cap',type=int,default=20)
    p.add_argument('--batch-size',type=int,default=128)
    args=p.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    if args.history_cap < 1 or args.batch_size < 1:raise ValueError('positive history-cap and batch-size required')
    ev=ExpertEvaluator.from_paths(args.config,args.sasrec_model,args.itemcf_model,args.semantic_emb,
                                device=args.device,mask_history=False)
    if ev.sasrec.maxlen!=args.history_cap:
        raise ValueError('SASRec checkpoint history cap differs; retrain rather than relabel')
    config=json.loads(Path(args.config).read_text())
    artifacts={k:{'path':getattr(args,k.replace('-','_')),'sha256':sha(getattr(args,k.replace('-','_')))}
               for k in ('config','sasrec-model','itemcf-model','semantic-emb')}
    artifacts['interactions']={'path':config['interactions_path'],'sha256':sha(config['interactions_path'])}
    for key in ('split_manifest', 'metadata_path'):
        if config.get(key):
            artifacts[key]={'path':config[key], 'sha256':sha(config[key])}
    manifest={'dataset':ev.dataset.name,'item_ids':ev.dataset.item_ids,'history_cap':args.history_cap,
              'artifacts':artifacts,'split_strategy':config.get('split_strategy','leave_two_out'),
              'schema':'anonymous-raw-expert-scores-v1','splits':{}}
    manifest['resources']={'device':args.device, 'batch_size':args.batch_size,
        'sasrec_parameters':sum(p.numel() for p in ev.sasrec.model.parameters()),
        'training_cost':'not measured by exporter',
        'timing_scope':'score export wall time including compression/I/O, not serving latency'}
    started=time.perf_counter()
    with torch.inference_mode():
        for split in ('val','test'):
            examples=collect_examples(ev,split);paths=[]
            for start in range(0,len(examples),args.batch_size):
                batch=examples[start:start+args.batch_size]
                histories=[truncate_history(row[1],args.history_cap) for row in batch]
                scores=np.stack(ev.score_batch([row[0] for row in batch],histories),axis=1)
                masks=np.zeros((len(batch),ev.dataset.num_items),dtype=bool)
                for i,h in enumerate(histories):
                    masks[i,[ev.sasrec.item_index[x] for x in set(h) if x in ev.sasrec.item_index]]=True
                name=f'{split}_{start:06d}.npz';path=out/name
                np.savez_compressed(path,scores=scores,history_mask=masks,
                                    user_ids=np.array([row[0] for row in batch]),
                                    targets=np.array([row[2] for row in batch]))
                paths.append({'path':name,'sha256':sha(path),'n_users':len(batch)})
            if not paths:raise ValueError(f'empty {split}')
            manifest['splits'][split]=paths
    manifest['resources']['export_seconds']=time.perf_counter()-started
    with (out/'manifest.json').open('x') as f:json.dump(manifest,f,indent=2);f.write('\n')


if __name__=='__main__':main()
