import json
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, ks_2samp
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier
from .data import ARTIFACTS, ROOT, SEED, locate_data, save_json
from .enhanced import prepare_enhanced


def validation_diagnostics():
    x,z,groups = prepare_enhanced()
    p = locate_data()
    train = pd.read_csv(p/'train_signals.csv',parse_dates=['signal_sanasi']).set_index('signal_id').loc[x.index]
    test = pd.read_csv(p/'test_signals.csv',parse_dates=['signal_sanasi']).set_index('signal_id').loc[z.index]
    contingency = pd.crosstab(train.signal_sanasi.dt.to_period('M'),train.eskalatsiya)
    chi2,pvalue,dof,_ = chi2_contingency(contingency)
    ks = ks_2samp(train.signal_sanasi.astype('int64'),test.signal_sanasi.astype('int64'))
    combined = pd.concat([x,z],ignore_index=True)
    source = np.r_[np.zeros(len(x)),np.ones(len(z))]
    preds = np.zeros(len(combined))
    aucs = []
    for tr,va in StratifiedKFold(3,shuffle=True,random_state=SEED).split(combined,source):
        model = LGBMClassifier(n_estimators=150,num_leaves=7,min_child_samples=100,learning_rate=.03,reg_lambda=20,verbosity=-1,n_jobs=4,random_state=SEED)
        model.fit(combined.iloc[tr],source[tr])
        preds[va] = model.predict_proba(combined.iloc[va])[:,1]
        aucs.append(float(roc_auc_score(source[va],preds[va])))
    result = {'monthly_target_chi2':float(chi2),'monthly_target_pvalue':float(pvalue),'monthly_target_dof':int(dof),'train_test_date_ks':float(ks.statistic),'train_test_date_ks_pvalue':float(ks.pvalue),'adversarial_source_auc':float(roc_auc_score(source,preds)),'adversarial_fold_auc':aucs,'duplicate_full_feature_rows_train':int(x.duplicated().sum()),'duplicate_full_feature_rows_across_splits':int(combined.duplicated().sum()-x.duplicated().sum()-z.duplicated().sum()),'interpretation':'Test dates are interspersed with training dates. Use 5-fold stratification for primary model selection, and expanding temporal splits as a robustness check. No customer identifier exists, so repeated-customer dependence cannot be ruled out.'}
    save_json(result,ARTIFACTS/'validation_diagnostics.json')
    save_json(result,ROOT/'eda_site'/'data'/'validation_diagnostics.json')
    print(json.dumps(result,indent=2),flush=True)
    return result


if __name__ == '__main__':
    validation_diagnostics()
