"""OOF blend assessment, full-data refits, and strict submission validation."""
import argparse
import json
import shutil
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from .data import ROOT, ARTIFACTS, SEED, locate_data, save_json
from .modeling import CONFIGS, load_training, select_columns, fold_columns, fit_model, predict, feature_importance, make_splits, run_experiment, update_results


def read_oof(name,strategy='stratified'):
    return pd.read_csv(ARTIFACTS/'experiments'/f'{name}__{strategy}'/'oof.csv').set_index('signal_id')


def score_prediction(prediction,s,strategy='stratified'):
    y = s.eskalatsiya.to_numpy()
    scores = [float(roc_auc_score(y[va],prediction[va])) for _,va in make_splits(s,strategy)]
    valid = np.isfinite(prediction)
    return {'auc_mean':float(np.mean(scores)),'auc_std':float(np.std(scores)),'auc_oof':float(roc_auc_score(y[valid],prediction[valid])),'fold_auc':scores}


def assess_blends(names):
    _,_,_,s = load_training()
    preds = {name:read_oof(name).loc[s.index,'prediction'].to_numpy() for name in names}
    rows=[]
    candidates={}
    for name in names:
        candidates[name]={name:1.0}
    anchor=names[0]
    for other in names[1:]:
        for w in [.25,.5,.75]:
            candidates[f'{anchor}:{w:g}+{other}:{1-w:g}']={anchor:w,other:1-w}
    for label,weights in candidates.items():
        p=sum(weight*preds[name] for name,weight in weights.items())
        rows.append({'blend':label,'weights':json.dumps(weights),**score_prediction(p,s)})
    result=pd.DataFrame(rows).sort_values('auc_oof',ascending=False)
    result.to_csv(ARTIFACTS/'blend_results.csv',index=False)
    pd.DataFrame(preds).corr().to_csv(ARTIFACTS/'oof_correlations.csv')
    print(result.to_string(index=False),flush=True)
    return result


def validate_submission(submission,test,sample):
    assert submission.columns.tolist()==['signal_id','ehtimollik']
    assert len(submission)==len(test)==len(sample)
    assert submission.signal_id.is_unique and submission.signal_id.notna().all()
    assert set(submission.signal_id)==set(test.signal_id)==set(sample.signal_id)
    assert submission.signal_id.tolist()==sample.signal_id.tolist()
    p=submission.ehtimollik.to_numpy()
    assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
    assert p.std()>1e-4 and np.unique(p).size>100
    return {'rows':len(submission),'columns':list(submission.columns),'unique_ids':int(submission.signal_id.nunique()),'matches_sample_order':True,'finite':True,'range_valid':True,'prediction_summary':submission.ehtimollik.describe().to_dict()}


def finalize(selection_path=None,team_id='SUBMIT_READY'):
    selection_path=Path(selection_path) if selection_path else ROOT/'solution'/'selection.json'
    selection=json.loads(selection_path.read_text())
    strategy=selection.get('validation_strategy','nested')
    assert abs(sum(selection['weights'].values())-1)<1e-9
    x,z,groups,s=load_training()
    y=s.eskalatsiya.to_numpy()
    oof=np.zeros(len(x))
    test_prediction=np.zeros(len(z))
    importance=pd.Series(0.0,index=x.columns)
    model_dir=ARTIFACTS/'models'
    model_dir.mkdir(exist_ok=True)
    details=[]
    temporal_oof=np.zeros(len(x))
    exploratory_oof=np.zeros(len(x))
    have_temporal=True
    for name,weight in selection['weights'].items():
        config=selection['configs'][name]
        result=run_experiment(name,config=config,strategy=strategy,loaded=(x,z,groups,s))
        run_experiment(name,config=config,strategy='stratified',loaded=(x,z,groups,s))
        run_experiment(name,config=config,strategy='temporal',loaded=(x,z,groups,s))
        oof+=weight*read_oof(name,strategy).loc[x.index,'prediction'].to_numpy()
        exploratory_oof+=weight*read_oof(name,'stratified').loc[x.index,'prediction'].to_numpy()
        try:
            temporal_oof+=weight*read_oof(name,'temporal').loc[x.index,'prediction'].to_numpy()
        except FileNotFoundError:
            have_temporal=False
        columns=select_columns(x,groups,config)
        keep=fold_columns(x[columns],y,config)
        rounds=int(np.median([f['best_iteration'] for f in result['folds']])) or None
        model,best=fit_model(config,x[keep],y,iterations=rounds)
        test_prediction+=weight*predict(model,z[keep],config)
        imp=feature_importance(model,keep)
        if imp.sum()>0:
            importance=importance.add(weight*imp/imp.sum(),fill_value=0)
        joblib.dump({'model':model,'columns':keep,'config':config},model_dir/f'{name}.joblib')
        details.append({'name':name,'weight':weight,'iterations':best,'n_features':len(keep)})
        print(f'Full-data refit: {name}, weight={weight}, iterations={best}, features={len(keep)}',flush=True)
    sample_file=list(locate_data().glob('sample_submission*.csv'))[0]
    sample=pd.read_csv(sample_file,usecols=['signal_id'])
    test=pd.read_csv(locate_data()/'test_signals.csv')
    pred=pd.Series(test_prediction,index=z.index)
    submission=sample.assign(ehtimollik=sample.signal_id.map(pred))
    checks=validate_submission(submission,test,sample)
    output=ROOT/f'team_{team_id}.csv'
    submission.to_csv(output,index=False)
    # Read the actual serialized file back and revalidate.
    validate_submission(pd.read_csv(output),test,sample)
    save_json(checks,ARTIFACTS/'submission_checks.json')
    metrics=score_prediction(oof,s)
    rng=np.random.default_rng(SEED)
    positive=np.flatnonzero(y==1); negative=np.flatnonzero(y==0)
    boot=[]
    for _ in range(500):
        rows=np.r_[rng.choice(positive,len(positive),replace=True),rng.choice(negative,len(negative),replace=True)]
        boot.append(roc_auc_score(y[rows],oof[rows]))
    summary={**metrics,'validation_strategy':strategy,'exploratory':score_prediction(exploratory_oof,s),'description':selection['description'],'selection_note':selection['selection_note'],'components':details,'oof_auc_bootstrap_interval_95':np.quantile(boot,[.025,.975]).tolist(),'bootstrap_caveat':'Conditional on selected models and reused OOF predictions; excludes model-selection uncertainty.','submission_filename':output.name,'temporal':score_prediction(temporal_oof,s,'temporal') if have_temporal else None}
    save_json(summary,ARTIFACTS/'final_summary.json')
    pd.DataFrame({'signal_id':x.index,'target':y,'prediction':oof}).to_csv(ARTIFACTS/'final_oof.csv',index=False)
    importance.sort_values(ascending=False).rename('importance').to_csv(ARTIFACTS/'final_importance.csv',index_label='feature')
    public=ROOT/'eda_site'/'data'
    public.mkdir(parents=True,exist_ok=True)
    for source,dest in [('final_summary.json','final_summary.json'),('experiment_results.csv','experiment_results.csv'),('final_importance.csv','importance.csv'),('blend_results.csv','blend_results.csv')]:
        if (ARTIFACTS/source).exists():
            shutil.copyfile(ARTIFACTS/source,public/dest)
    print(json.dumps(summary,indent=2),flush=True)
    print(f'Validated submission saved to {output}',flush=True)
    return submission,summary


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--blend-models',nargs='+')
    parser.add_argument('--team-id',default='SUBMIT_READY')
    args=parser.parse_args()
    if args.blend_models:
        assess_blends(args.blend_models)
    else:
        finalize(team_id=args.team_id)
