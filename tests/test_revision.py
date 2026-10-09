import numpy as np
import pytest

from scripts.revision.paired_statistics import align, holm, paired_audit, read_predictions
from scripts.revision.tool_headroom import analyze
from scripts.revision.witness_ladder import configs, evaluate, normalize, simplex


def test_pairing_rejects_target_mismatch_and_duplicates(tmp_path):
    a={'u':{'target_item_id':'x'}}
    with pytest.raises(ValueError):align(a,{'u':{'target_item_id':'y'}})
    with pytest.raises(ValueError):align(a,{})
    p=tmp_path/'x.jsonl';p.write_text('{"user_id":"u","ranking":[]}\n'*2)
    with pytest.raises(ValueError):read_predictions(p)


def test_zero_discordance_not_exact_certainty():
    a=np.r_[np.ones(100),np.zeros(900)]
    row=paired_audit(a,a,margin=0,resamples=200)
    assert row['lcb95']==0
    assert row['bootstrap_verdict']=='unresolved'
    assert row['exact_cell_bound_lcb']<0
    assert row['hoeffding_p_bound']==1


def test_joint_margin_includes_target_sampling():
    a=np.r_[np.ones(200),np.zeros(800)]
    row=paired_audit(a,a,margin=.05,resamples=2000)
    assert 0<row['joint_margin_lcb']<.01
    assert row['delta']==0
    assert row['bootstrap_verdict']=='recovery'
    assert row['absolute_margin_at_observed_target']==pytest.approx(.01)


def test_holm_controls_entire_family():
    result=holm({'a':.001,'b':.03,'c':.5})
    assert result['a']['adjusted_p_bound']==pytest.approx(.003)
    assert result['a']['reject']
    assert result['b']['adjusted_p_bound']==pytest.approx(.06)
    assert not result['b']['reject']


def test_masked_extreme_does_not_change_normalization():
    raw=np.array([[[999.,2.,4.],[888.,1.,3.],[777.,6.,8.]]])
    history=np.array([[True,False,False]])
    scores,eligible=normalize(raw,history,True)
    assert not eligible[0,0]
    np.testing.assert_allclose(scores[0,:,1],[0,0,0])
    np.testing.assert_allclose(scores[0,:,2],[1,1,1])
    scores2,_=normalize(raw*2,history,True)
    np.testing.assert_allclose(scores,scores2,atol=1e-7)


def test_stable_tie_and_masked_target_fail_closed():
    s=np.zeros((1,12));valid=np.ones((1,12),dtype=bool)
    assert not evaluate(s,valid,np.array([10]),20)['hits'][0]
    valid[0,10]=False
    with pytest.raises(ValueError):evaluate(s,valid,np.array([10]),20)


def test_all_simplex_candidates_and_no_named_calibration():
    w=simplex(10)
    assert len(w)==66
    assert all(sum(x)==pytest.approx(1) for x in w)
    c=configs(2,[0,.1])
    assert c['sasrec']['penalty']==0
    assert c['fusion_nocal']['penalty']==0


def test_oracle_discovers_per_user_headroom_with_same_tools():
    def rows(item):return [{'item_id':item,'score':1.}]
    data={'rows':[{'user_id':'u1','history':[], 'tools':{'sasrec':rows('a'),'itemcf':rows('b'),'semantic':rows('c')}},
                  {'user_id':'u2','history':[], 'tools':{'sasrec':rows('a'),'itemcf':rows('b'),'semantic':rows('c')}}],
          'targets':{'u1':'a','u2':'b'}}
    sets,hits,ndcg,coverage=analyze(data,1)
    assert hits.max(1).mean()==1
    assert hits[:,0].mean()==.5
    assert np.all(ndcg.max(1)>=ndcg[:,0])
    assert np.all(coverage.max(1)>=hits.max(1))


def test_temporal_filter_uses_training_only_and_ties_stay_on_one_side():
    from lime_rec.data import Interaction
    from scripts.revision.temporal_split import split_rows
    rows=[Interaction('u','a',1),Interaction('u','b',2),Interaction('u','a',3),
          Interaction('u','b',4),Interaction('u','a',5),Interaction('u','new',6),
          Interaction('future','a',2),Interaction('future','a',6),Interaction('future','a',7)]
    result=split_rows(rows,3,4,min_user=2,min_item=1)
    assert result['history_by_user']=={'u':['a','b','a']}
    assert result['valid_by_user']=={'u':'b'}
    assert result['test_by_user']=={'u':'a'}
    assert 'new' not in result['item_ids']
    assert 'future' not in result['history_by_user']


def test_temporal_manifest_loader_rejects_changed_input(tmp_path):
    import hashlib,json
    from lime_rec.data import load_dataset
    raw=tmp_path/'events.jsonl';raw.write_text('not used by explicit split\n')
    manifest=tmp_path/'split.json'
    manifest.write_text(json.dumps({'dataset':'d','source_sha256':hashlib.sha256(raw.read_bytes()).hexdigest(),
        'history_by_user':{'u':['a','b']},'valid_by_user':{'u':'b'},'test_by_user':{'u':'a'},'item_ids':['a','b']}))
    ds=load_dataset('d',str(raw),split_manifest=str(manifest))
    assert ds.history_by_user=={'u':['a','b']}
    raw.write_text('changed\n')
    with pytest.raises(ValueError):load_dataset('d',str(raw),split_manifest=str(manifest))


def test_schema_excludes_called_tools_and_limits_visible_candidates():
    from scripts.revision.strict_controller import action_schema
    view={'available_tools':['sasrec','itemcf'],'called_tools':['sasrec'],'remaining_tool_budget':1}
    assert action_schema(view)['properties']['tool']['enum']==['itemcf']
    view['candidate_ids']=[str(i) for i in range(10)]
    schema=action_schema(view)
    assert schema['anyOf'][0]['properties']['tool']['enum']==['itemcf']
    view['remaining_tool_budget']=0
    assert action_schema(view)['properties']['ranking']['items']['enum']==view['candidate_ids']


def test_selector_retries_are_bounded_and_do_not_reveal_labels():
    from scripts.revision.strict_controller import RetryingSelector
    class Client:
        def __init__(self):self.calls=0
        def generate(self,messages,**kwargs):
            self.calls+=1
            assert 'target_item_id' not in messages[-1]['content']
            return {'text':'bad' if self.calls==1 else '{"tool":"sasrec"}','usage':{}}
    client=Client();policy=RetryingSelector(client,retries=1)
    view={'history':[],'observations':{},'available_tools':['sasrec'],'called_tools':[],'remaining_tool_budget':1}
    assert policy(view)=='sasrec'
    assert client.calls==2
    assert len(policy.events)==2


def test_schema_client_sends_constraint_and_preserves_usage(monkeypatch):
    import json
    from scripts.revision import strict_controller
    captured = {}
    class Response:
        def raise_for_status(self): pass
        def json(self):
            return {'choices':[{'message':{'content':'{"tool":"sasrec"}'}}],
                    'usage':{'prompt_tokens':12, 'completion_tokens':5}}
    def post(url, **kwargs):
        captured.update(kwargs['json'])
        return Response()
    monkeypatch.setattr(strict_controller.requests, 'post', post)
    client = strict_controller.SchemaClient('http://local/v1', 'test', 'test-model')
    view = {'available_tools':['sasrec'], 'called_tools':[], 'remaining_tool_budget':1}
    result = client.generate([{'role':'user','content':json.dumps(view)}], temperature=0, max_tokens=128)
    assert captured['response_format']['json_schema']['strict']
    assert captured['response_format']['json_schema']['schema']['properties']['tool']['enum']==['sasrec']
    assert result['usage']['prompt_tokens']==12


def test_witness_selection_independent_of_test_labels(tmp_path, monkeypatch):
    import json, sys
    from scripts.revision.witness_ladder import main
    from scripts.revision.paired_statistics import sha
    selected = []
    for case, target in enumerate((0, 11)):
        cache = tmp_path / f'cache{case}'; cache.mkdir()
        meta = {'dataset':'synthetic', 'item_ids':[str(i) for i in range(12)],
                'history_cap':20, 'splits':{}}
        for split in ('val','test'):
            path = cache / f'{split}.npz'
            raw = np.array([[np.arange(12), np.arange(12)[::-1], np.arange(12)]], dtype=float)
            np.savez(path, scores=raw, history_mask=np.zeros((1,12),dtype=bool),
                     targets=np.array([11 if split=='val' else target]), user_ids=np.array(['u']))
            meta['splits'][split]=[{'path':path.name, 'sha256':sha(path)}]
        (cache/'manifest.json').write_text(json.dumps(meta))
        out = tmp_path / f'out{case}'
        monkeypatch.setattr(sys, 'argv', ['witness_ladder','--cache',str(cache), '--out',str(out),
                                         '--divisions','2', '--penalties','0,0.1'])
        main()
        frozen = json.loads((out/'frozen_validation_selection.json').read_text())
        selected.append(frozen['selected'])
        assert json.loads((out/'summary.json').read_text())['test_surface_role']=='not computed'
        assert all((out/f'{name}_predictions.jsonl').exists() for name in frozen['selected'])
    assert selected[0]==selected[1]
