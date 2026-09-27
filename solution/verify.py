"""Meaningful integrity checks: historical cutoff, row coverage, and public app pages."""
from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd
from .data import ROOT, ARTIFACTS, load_data, historical_transactions, save_json
from .features import build_features
from .enhanced import enrich


def integrity_checks():
    train, test, tx_train, tx_test, sample = load_data()
    # A small real subset has realistic categories and timestamp precision.
    s = train.iloc[:20].copy()
    t = tx_train.loc[tx_train.signal_id.isin(s.signal_id)].copy()
    base,groups = build_features(s,t)
    extra = t.iloc[[0]].copy()
    extra['tranzaksiya_vaqti'] = s.set_index('signal_id').loc[extra.signal_id,'signal_sanasi'].to_numpy()+np.timedelta64(1,'h')
    extra['miqdor_indeksi'] = 1e6
    augmented,_ = build_features(s,pd.concat([t,extra],ignore_index=True))
    pd.testing.assert_frame_equal(base,augmented)
    enhanced_base,_ = enrich(s,t,base)
    enhanced_augmented,_ = enrich(s,pd.concat([t,extra],ignore_index=True),augmented)
    pd.testing.assert_frame_equal(enhanced_base,enhanced_augmented)
    # A valid transaction at the exact cutoff must contribute to its alert.
    extra['tranzaksiya_vaqti'] = s.set_index('signal_id').loc[extra.signal_id,'signal_sanasi'].to_numpy()
    included,_ = build_features(s,pd.concat([t,extra],ignore_index=True))
    sid = extra.signal_id.iloc[0]
    assert included.loc[sid,'all_count'] == base.loc[sid,'all_count']+1
    assert included.loc[sid,'all_max'] == 1e6
    assert base.index.tolist()==s.signal_id.tolist()
    assert 'signal_id' not in base and 'eskalatsiya' not in base
    # All cutoff-filtered rows must meet the declared rule.
    h = historical_transactions(s,t)
    assert (h.tranzaksiya_vaqti<=h.signal_sanasi).all()
    audit = json.loads((ARTIFACTS/'data_audit.json').read_text())
    data_dir = __import__('solution.data',fromlist=['locate_data']).locate_data()
    for name,digest in audit['source_sha256'].items():
        assert hashlib.sha256((data_dir/name).read_bytes()).hexdigest()==digest
    result = {'future_transaction_invariance':True,'enhanced_future_transaction_invariance':True,'features_checked':len(enhanced_base.columns),'exact_cutoff_inclusion':True,'signal_row_order_preserved':True,'no_identifier_or_target_predictors':True,'source_files_unchanged':True}
    save_json(result,ARTIFACTS/'integrity_checks.json')
    print(result,flush=True)
    return result


def test_site():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT/'eda_site'/'app.py')).run(timeout=60)
    assert not at.exception,at.exception
    pages = ['Overview','Target & time','Transaction behavior','Outcome differences','Validation & models','Conclusions']
    for page in pages:
        at.sidebar.radio[0].set_value(page).run(timeout=60)
        assert not at.exception,(page,at.exception)
    save_json({'pages_passed':pages,'framework':'streamlit.testing.v1.AppTest','app_sha256':hashlib.sha256((ROOT/'eda_site'/'app.py').read_bytes()).hexdigest()},ARTIFACTS/'site_checks.json')
    print('All six Streamlit pages passed AppTest.',flush=True)


if __name__ == '__main__':
    integrity_checks()
    test_site()
