"""Train-only k-core global temporal split; no held-out events enter training."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

from lime_rec.data import _read_interactions, _iterative_kcore
from .paired_statistics import sha


def split_rows(rows, train_end, validation_end, min_user=5, min_item=5):
    if train_end >= validation_end:raise ValueError('ordered cutoffs required')
    train=_iterative_kcore([r for r in rows if r.timestamp<=train_end],min_user,min_item)
    users={r.user_id for r in train};items={r.item_id for r in train}
    history=defaultdict(list)
    for r in sorted(train,key=lambda x:(x.timestamp,x.item_id)):history[r.user_id].append(r.item_id)
    valid,test={},{}
    for r in sorted(rows,key=lambda x:(x.timestamp,x.item_id)):
        if r.user_id not in users or r.item_id not in items:continue
        if train_end<r.timestamp<=validation_end:valid.setdefault(r.user_id,r.item_id)
        if r.timestamp>validation_end:test.setdefault(r.user_id,r.item_id)
    eligible=set(valid)&set(test)
    if not eligible:raise ValueError('no users with both temporal targets')
    return {'history_by_user':dict(history),'valid_by_user':{u:valid[u] for u in sorted(eligible)},
            'test_by_user':{u:test[u] for u in sorted(eligible)},'item_ids':sorted(items),
            'training_users':len(users),'evaluation_users':len(eligible),'train_interactions':len(train),
            'train_end':train_end,'validation_end':validation_end,
            'context_rule':'training history for val; training history plus chosen validation target for test',
            'candidate_rule':'training catalog only; warm users/items; report exclusion counts',
            'excluded_postcutoff_events':sum(r.timestamp>train_end and (r.user_id not in users or r.item_id not in items) for r in rows)}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',required=True);p.add_argument('--out',required=True)
    p.add_argument('--raw-interactions',required=True,help='source with actual Unix timestamps, not converted GRAM sequence positions')
    p.add_argument('--train-end',type=int,required=True);p.add_argument('--validation-end',type=int,required=True)
    args=p.parse_args();config=json.loads(Path(args.config).read_text())
    config['interactions_path']=args.raw_interactions
    rows=list(_read_interactions(Path(args.raw_interactions)))
    if not rows or min(r.timestamp for r in rows)<100000000 or args.train_end<100000000:
        raise ValueError('BLOCKED_MISSING_REAL_TIMESTAMPS: sequence-position timestamps are not global time')
    result=split_rows(rows,args.train_end,args.validation_end,
                      config.get('min_user_interactions',5),config.get('min_item_interactions',5))
    result['dataset']=config['name'];result['source_sha256']=sha(config['interactions_path'])
    result['metadata_path']=config.get('metadata_path')
    dest=Path(args.out);dest.parent.mkdir(parents=True,exist_ok=True)
    with dest.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    config['split_strategy']='global_temporal'
    config['split_manifest']=str(dest.resolve())
    config['interactions_path']=str(Path(config['interactions_path']).resolve())
    if config.get('metadata_path'):config['metadata_path']=str(Path(config['metadata_path']).resolve())
    with dest.with_suffix('.config.json').open('x') as f:json.dump(config,f,indent=2);f.write('\n')


if __name__=='__main__':main()
