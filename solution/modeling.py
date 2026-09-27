"""Fixed-split experiments with fold-local preprocessing and saved OOF predictions."""
import argparse
import json
import os
import time
from pathlib import Path
os.environ.setdefault('OMP_NUM_THREADS', '8')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '8')
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, QuantileTransformer, SplineTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from .data import ARTIFACTS, SEED, locate_data, save_json
from .features import prepare_features
from .enhanced import prepare_enhanced

THREADS = 8
GROUP_SETS = {
    'baseline': ['baseline'],
    'recent': ['baseline','recent'],
    'direction': ['baseline','recent','direction'],
    'types': ['baseline','recent','direction','types','diversity'],
    'change': ['baseline','recent','direction','types','diversity','change'],
    'sequence': ['baseline','recent','direction','types','diversity','change','sequence'],
    'full': ['baseline','recent','direction','types','diversity','change','sequence','calendar'],
    'contrasts': ['baseline','recent','direction','types','diversity','contrasts'],
    'enhanced': ['baseline','recent','direction','types','diversity','change','sequence','contrasts','microtime'],
    'compact': ['baseline','direction','types','diversity','contrasts'],
}
CONFIGS = {
    'logistic': {'family':'logistic','features':'sequence','C':0.1},
    'extra_trees': {'family':'extra','features':'compact','n_estimators':250,'min_samples_leaf':10,'max_features':0.5},
    'hist_gradient': {'family':'hist','features':'sequence','max_iter':500,'max_leaf_nodes':15,'l2_regularization':10.0,'learning_rate':0.05},
    **{f'lgb_{stage}': {'family':'lgb','features':stage,'num_leaves':15,'learning_rate':0.035,'reg_lambda':10.0,'min_child_samples':60,'colsample_bytree':0.85,'subsample':0.85} for stage in GROUP_SETS},
    'lgb_leaves31': {'family':'lgb','features':'sequence','num_leaves':31,'learning_rate':0.03,'reg_lambda':20.0,'min_child_samples':70,'colsample_bytree':0.8,'subsample':0.85},
    'lgb_leaves7': {'family':'lgb','features':'sequence','num_leaves':7,'learning_rate':0.035,'reg_lambda':15.0,'min_child_samples':80,'colsample_bytree':0.9,'subsample':0.9},
    'cat_depth5': {'family':'cat','features':'sequence','depth':5,'learning_rate':0.04,'l2_leaf_reg':8.0,'random_strength':0.5},
    'cat_depth6': {'family':'cat','features':'sequence','depth':6,'learning_rate':0.035,'l2_leaf_reg':12.0,'random_strength':0.5},
    'cat_depth4': {'family':'cat','features':'sequence','depth':4,'learning_rate':0.045,'l2_leaf_reg':8.0,'random_strength':0.5},
    'cat_native': {'family':'cat','features':'full','native_categories':True,'depth':5,'learning_rate':0.04,'l2_leaf_reg':10.0,'random_strength':0.5},
    'xgb_depth3': {'family':'xgb','features':'sequence','max_depth':3,'learning_rate':0.035,'min_child_weight':15,'reg_lambda':20.0,'colsample_bytree':0.85,'subsample':0.85},
    'xgb_depth5': {'family':'xgb','features':'sequence','max_depth':5,'learning_rate':0.03,'min_child_weight':20,'reg_lambda':25.0,'colsample_bytree':0.8,'subsample':0.85},
    'lgb_contrasts': {'family':'lgb','features':'contrasts','num_leaves':15,'learning_rate':0.035,'reg_lambda':15.0,'min_child_samples':80,'colsample_bytree':0.85,'subsample':0.85},
    'lgb_enhanced': {'family':'lgb','features':'enhanced','num_leaves':15,'learning_rate':0.035,'reg_lambda':15.0,'min_child_samples':80,'colsample_bytree':0.85,'subsample':0.85},
    'lgb_compact': {'family':'lgb','features':'compact','num_leaves':7,'learning_rate':0.035,'reg_lambda':20.0,'min_child_samples':100,'colsample_bytree':0.9,'subsample':0.9},
    'cat_compact': {'family':'cat','features':'compact','depth':4,'learning_rate':0.04,'l2_leaf_reg':15.0,'random_strength':0.5},
    'cat_contrasts': {'family':'cat','features':'contrasts','depth':5,'learning_rate':0.04,'l2_leaf_reg':12.0,'random_strength':0.5},
    'xgb_compact': {'family':'xgb','features':'compact','max_depth':2,'learning_rate':0.035,'min_child_weight':30,'reg_lambda':30.0,'colsample_bytree':0.9,'subsample':0.9},
    'logistic_compact': {'family':'logistic','features':'compact','C':0.1},
    'lgb_contrasts7': {'family':'lgb','features':'contrasts','num_leaves':7,'learning_rate':0.03,'reg_lambda':25.0,'min_child_samples':120,'colsample_bytree':0.9,'subsample':0.9},
    'lgb_contrasts31': {'family':'lgb','features':'contrasts','num_leaves':31,'learning_rate':0.025,'reg_lambda':25.0,'min_child_samples':100,'colsample_bytree':0.85,'subsample':0.85},
    'cat_compact6': {'family':'cat','features':'compact','depth':6,'learning_rate':0.03,'l2_leaf_reg':20.0,'random_strength':1.0},
    'cat_compact3': {'family':'cat','features':'compact','depth':3,'learning_rate':0.04,'l2_leaf_reg':15.0,'random_strength':0.5},
    'lgb_select60': {'family':'lgb','features':'contrasts','select_top':60,'num_leaves':7,'learning_rate':0.03,'reg_lambda':25.0,'min_child_samples':100,'colsample_bytree':0.9,'subsample':0.9},
    'spline_compact': {'family':'spline','features':'compact','C':0.05},
    'spline_compact_reg': {'family':'spline','features':'compact','C':0.005},
}


def load_training():
    x,z,groups = prepare_enhanced()
    s = pd.read_csv(locate_data()/'train_signals.csv', parse_dates=['signal_sanasi']).set_index('signal_id').loc[x.index]
    return x,z,groups,s


def select_columns(x, groups, config):
    columns = [c for c in x if groups[c] in GROUP_SETS[config['features']]]
    if config['features'] == 'compact':
        columns = [c for c in columns if c.startswith(('all_','type_','amount_','abs_','own_','daily_','history_'))]
    if 'columns' in config:
        columns = [c for c in columns if c in config['columns']]
    return columns


def clean_columns(training):
    """Fit constant/duplicate removal using this fold's training rows only."""
    nonconstant = training.columns[training.nunique(dropna=False)>1]
    fingerprints, keep = {}, []
    for col in nonconstant:
        digest = pd.util.hash_pandas_object(training[col], index=False).values.tobytes()
        if digest not in fingerprints:
            fingerprints[digest] = col
            keep.append(col)
    return keep


def native_frame(x, config):
    x = x.copy()
    categorical = []
    if config.get('native_categories'):
        categorical = [c for c in x if (c.startswith('last_tx_') and c.endswith(('_direction','_type'))) or c in ['alert_month','alert_dow']]
        for col in categorical:
            x[col] = x[col].fillna(-999).astype(int).astype(str)
    return x, categorical


def fold_columns(x,y,config):
    keep = clean_columns(x)
    if config.get('select_top'):
        from lightgbm import LGBMClassifier
        selector = LGBMClassifier(n_estimators=150,num_leaves=7,learning_rate=.04,min_child_samples=100,reg_lambda=20,n_jobs=THREADS,verbosity=-1,random_state=SEED,importance_type='gain')
        selector.fit(x[keep],y)
        keep = pd.Series(selector.feature_importances_,index=keep).nlargest(config['select_top']).index.tolist()
    return keep


def fit_model(config, x_train, y_train, x_valid=None, y_valid=None, iterations=None):
    family = config['family']
    params = {k:v for k,v in config.items() if k not in ['family','features','native_categories','columns','seed','select_top']}
    seed = config.get('seed',SEED)
    valid = x_valid is not None
    if family == 'lgb':
        import lightgbm as lgb
        model = lgb.LGBMClassifier(n_estimators=iterations or 2500,objective='binary',metric='auc',random_state=seed,n_jobs=THREADS,verbosity=-1,subsample_freq=1,**params)
        fit_args = {'eval_set':[(x_valid,y_valid)],'eval_metric':'auc','callbacks':[lgb.early_stopping(150,verbose=False)]} if valid else {}
        model.fit(x_train,y_train,**fit_args)
        best = model.best_iteration_ or (iterations or 2500)
    elif family == 'cat':
        from catboost import CatBoostClassifier
        x_train, categorical = native_frame(x_train,config)
        if valid:
            x_valid,_ = native_frame(x_valid,config)
        model = CatBoostClassifier(iterations=iterations or 2200,loss_function='Logloss',eval_metric='AUC',random_seed=seed,thread_count=THREADS,verbose=False,allow_writing_files=False,**params)
        model.fit(x_train,y_train,cat_features=categorical,eval_set=(x_valid,y_valid) if valid else None,early_stopping_rounds=150 if valid else None)
        best = model.tree_count_
    elif family == 'xgb':
        from xgboost import XGBClassifier
        model = XGBClassifier(n_estimators=iterations or 2500,objective='binary:logistic',eval_metric='auc',tree_method='hist',n_jobs=THREADS,random_state=seed,early_stopping_rounds=150 if valid else None,**params)
        model.fit(x_train,y_train,eval_set=[(x_valid,y_valid)] if valid else None,verbose=False)
        best = model.best_iteration+1 if valid else (iterations or 2500)
    elif family == 'spline':
        model = make_pipeline(SimpleImputer(strategy='median',add_indicator=True),QuantileTransformer(n_quantiles=200,random_state=seed,subsample=None),SplineTransformer(n_knots=5,degree=2,include_bias=False),StandardScaler(),LogisticRegression(C=params['C'],max_iter=3000,random_state=seed))
        model.fit(x_train,y_train)
        best = 0
    elif family == 'logistic':
        model = make_pipeline(SimpleImputer(strategy='median',add_indicator=True),StandardScaler(),LogisticRegression(C=params['C'],max_iter=4000,random_state=seed))
        model.fit(x_train,y_train)
        best = 0
    elif family == 'extra':
        model = make_pipeline(SimpleImputer(strategy='median',add_indicator=True),ExtraTreesClassifier(random_state=seed,n_jobs=THREADS,**params))
        model.fit(x_train,y_train)
        best = params['n_estimators']
    elif family == 'hist':
        model = HistGradientBoostingClassifier(random_state=seed,early_stopping=True,scoring='roc_auc',validation_fraction=0.15,n_iter_no_change=40,**params)
        model.fit(x_train,y_train)
        best = model.n_iter_
    else:
        raise ValueError(family)
    return model,best


def predict(model,x,config):
    if config['family'] == 'cat':
        x,_ = native_frame(x,config)
    return model.predict_proba(x)[:,1]


def feature_importance(model, columns):
    if hasattr(model,'feature_importances_'):
        return pd.Series(model.feature_importances_,index=columns)
    return pd.Series(0.0,index=columns)


def make_splits(s, strategy='stratified'):
    if strategy in ['stratified','nested']:
        return list(StratifiedKFold(5,shuffle=True,random_state=SEED).split(s,s.eskalatsiya))
    dates = s.signal_sanasi
    unique_dates = np.sort(dates.unique())
    bounds = [unique_dates[int(len(unique_dates)*q)] for q in [.5,.65,.8]] + [unique_dates[-1]+np.timedelta64(1,'D')]
    splits = []
    for lo,hi in zip(bounds[:-1],bounds[1:]):
        tr = np.flatnonzero((dates<lo).to_numpy())
        va = np.flatnonzero(((dates>=lo)&(dates<hi)).to_numpy())
        assert dates.iloc[tr].max()<dates.iloc[va].min()
        splits.append((tr,va))
    return splits


def run_experiment(name, config=None, strategy='stratified', force=False, loaded=None):
    config = dict(CONFIGS[name] if config is None else config)
    out = ARTIFACTS/'experiments'/f'{name}__{strategy}'
    out.mkdir(parents=True,exist_ok=True)
    if (out/'result.json').exists() and not force:
        result = json.loads((out/'result.json').read_text())
        if result['config'] == config:
            print(f"Cached {name} {strategy}: {result['auc_mean']:.6f}",flush=True)
            return result
    x,z,groups,s = loaded if loaded is not None else load_training()
    columns = select_columns(x,groups,config)
    x = x[columns]
    y = s.eskalatsiya.to_numpy()
    oof = np.full(len(x),np.nan)
    test_predictions = np.zeros(len(z))
    splits = make_splits(s,strategy)
    metrics, importances = [], []
    start = time.time()
    for fold,(tr,va) in enumerate(splits):
        keep = fold_columns(x.iloc[tr],y[tr],config)
        if strategy == 'nested':
            inner_train,inner_valid = train_test_split(tr,test_size=.2,stratify=y[tr],random_state=SEED+fold)
            inner_keep = fold_columns(x.iloc[inner_train],y[inner_train],config)
            _,best = fit_model(config,x.iloc[inner_train][inner_keep],y[inner_train],x.iloc[inner_valid][inner_keep],y[inner_valid])
            model,best = fit_model(config,x.iloc[tr][keep],y[tr],iterations=int(best) or None)
        else:
            model,best = fit_model(config,x.iloc[tr][keep],y[tr],x.iloc[va][keep],y[va])
        p = predict(model,x.iloc[va][keep],config)
        oof[va] = p
        test_predictions += predict(model,z[keep],config)/len(splits)
        auc = float(roc_auc_score(y[va],p))
        metrics.append({'fold':fold,'auc':auc,'best_iteration':int(best),'n_features':len(keep),'n_train':len(tr),'n_valid':len(va),'train_date_max':str(s.signal_sanasi.iloc[tr].max()),'valid_date_min':str(s.signal_sanasi.iloc[va].min()),'valid_date_max':str(s.signal_sanasi.iloc[va].max())})
        importances.append(feature_importance(model,keep))
        print(f'{name} {strategy} fold {fold+1}/{len(splits)}: AUC={auc:.6f}, iterations={best}, features={len(keep)}, elapsed={time.time()-start:.1f}s',flush=True)
    valid = np.isfinite(oof)
    result = {'name':name,'strategy':strategy,'config':config,'auc_mean':float(np.mean([m['auc'] for m in metrics])),'auc_std':float(np.std([m['auc'] for m in metrics])),'auc_oof':float(roc_auc_score(y[valid],oof[valid])),'folds':metrics,'seconds':time.time()-start,'feature_version':'historical_v1'}
    pd.DataFrame({'signal_id':x.index,'target':y,'prediction':oof}).to_csv(out/'oof.csv',index=False)
    pd.DataFrame({'signal_id':z.index,'prediction':test_predictions}).to_csv(out/'test_cv.csv',index=False)
    pd.concat(importances,axis=1).fillna(0).mean(axis=1).sort_values(ascending=False).rename('importance').to_csv(out/'importance.csv',index_label='feature')
    save_json(result,out/'result.json')
    update_results()
    print(f"FINISHED {name} {strategy}: {result['auc_mean']:.6f} +/- {result['auc_std']:.6f}; pooled OOF {result['auc_oof']:.6f}",flush=True)
    return result


def update_results():
    rows = []
    for f in (ARTIFACTS/'experiments').glob('*/result.json'):
        r = json.loads(f.read_text())
        rows.append({'experiment':r['name'],'features':r['config']['features'],'model':r['config']['family'],'validation':r['strategy'],'auc_mean':r['auc_mean'],'auc_std':r['auc_std'],'auc_oof':r['auc_oof'],'seconds':r['seconds'],'fold_auc':'; '.join(f"{m['auc']:.6f}" for m in r['folds']),'iterations':'; '.join(str(m['best_iteration']) for m in r['folds']),'parameters':json.dumps(r['config'])})
    df = pd.DataFrame(rows).sort_values(['validation','auc_mean'],ascending=[True,False])
    df.to_csv(ARTIFACTS/'experiment_results.csv',index=False)
    return df


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--models',nargs='+',default=['lgb_baseline','lgb_recent','lgb_direction','lgb_types','lgb_change','lgb_sequence','lgb_full','logistic','hist_gradient','extra_trees','cat_depth5','xgb_depth3'])
    p.add_argument('--strategy',default='stratified',choices=['stratified','temporal','nested'])
    p.add_argument('--force',action='store_true')
    args = p.parse_args()
    loaded = load_training()
    for name in args.models:
        run_experiment(name,strategy=args.strategy,force=args.force,loaded=loaded)


if __name__ == '__main__':
    main()
