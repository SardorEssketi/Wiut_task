"""Diagnostics for the frozen recipe; no model-selection decisions are made here."""
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from .data import ROOT, ARTIFACTS, SEED, save_json
from .modeling import load_training, make_splits, select_columns, fold_columns, fit_model, predict, feature_importance
from .finalize import read_oof


def final_diagnostics():
    x,z,groups,s=load_training()
    selection=json.loads((ROOT/'solution'/'selection.json').read_text())
    y=s.eskalatsiya.to_numpy()
    tr,va=make_splits(s)[0]
    models=[]
    importances=pd.Series(0.0,index=x.columns)
    for name,weight in selection['weights'].items():
        config=selection['configs'][name]
        columns=select_columns(x,groups,config)
        keep=fold_columns(x.iloc[tr][columns],y[tr],config)
        rounds=json.loads((ARTIFACTS/'experiments'/f'{name}__nested'/'result.json').read_text())['folds'][0]['best_iteration']
        model,_=fit_model(config,x.iloc[tr][keep],y[tr],iterations=rounds)
        models.append((model,keep,config,weight))
        imp=feature_importance(model,keep)
        if imp.sum()>0:
            importances=importances.add(weight*imp/imp.sum(),fill_value=0)
    def ensemble(frame):
        return sum(w*predict(m,frame[cols],cfg) for m,cols,cfg,w in models)
    valid=x.iloc[va]
    baseline=float(roc_auc_score(y[va],ensemble(valid)))
    rng=np.random.default_rng(SEED)
    rows=[]
    candidates={c:[c] for c in importances.nlargest(20).index}
    used=set(c for _,cols,_,_ in models for c in cols)
    for group in sorted(set(groups[c] for c in used)):
        candidates['GROUP:'+group]=[c for c in used if groups[c]==group]
    for name,columns in candidates.items():
        deltas=[]
        for repeat in range(3):
            changed=valid.copy()
            permutation=rng.permutation(len(valid))
            changed.loc[:,columns]=valid.iloc[permutation][columns].to_numpy()
            deltas.append(baseline-roc_auc_score(y[va],ensemble(changed)))
        rows.append({'feature_or_group':name,'n_columns':len(columns),'mean_auc_drop':float(np.mean(deltas)),'std_auc_drop':float(np.std(deltas))})
    result=pd.DataFrame(rows).sort_values('mean_auc_drop',ascending=False)
    result.to_csv(ARTIFACTS/'permutation_importance.csv',index=False)
    result.to_csv(ROOT/'eda_site'/'data'/'permutation_importance.csv',index=False)
    p=sum(w*read_oof(name,'nested').loc[x.index,'prediction'].to_numpy() for name,w in selection['weights'].items())
    single=read_oof('lgb_contrasts','nested').loc[x.index,'prediction'].to_numpy()
    positive=np.flatnonzero(y==1); negative=np.flatnonzero(y==0)
    differences=[]
    for _ in range(500):
        sampled=np.r_[rng.choice(positive,len(positive),replace=True),rng.choice(negative,len(negative),replace=True)]
        differences.append(roc_auc_score(y[sampled],p[sampled])-roc_auc_score(y[sampled],single[sampled]))
    save_json({'heldout_fold':0,'heldout_ensemble_auc':baseline,'permutation_repeats':3,'blend_minus_lgb_oof_auc':float(roc_auc_score(y,p)-roc_auc_score(y,single)),'paired_bootstrap_delta_interval_95':np.quantile(differences,[.025,.975]).tolist(),'caveat':'Diagnostic only. Correlation can suppress marginal importance. Conditional bootstrap does not include model-selection uncertainty.'},ARTIFACTS/'final_diagnostics.json')
    print(result.head(15).to_string(index=False),flush=True)


if __name__=='__main__':
    final_diagnostics()
