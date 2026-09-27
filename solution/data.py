from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / 'artifacts'
SEED = 20260927


def locate_data(data_dir=None):
    if data_dir is not None:
        return Path(data_dir)
    matches = list(ROOT.rglob('train_signals.csv'))
    if len(matches) != 1:
        raise ValueError(f'Expected one data directory, found {matches}')
    return matches[0].parent


def load_data(data_dir=None):
    p = locate_data(data_dir)
    train = pd.read_csv(p / 'train_signals.csv', parse_dates=['signal_sanasi'])
    test = pd.read_csv(p / 'test_signals.csv', parse_dates=['signal_sanasi'])
    tx_train = pd.read_parquet(p / 'train_transactions.parquet')
    tx_test = pd.read_parquet(p / 'test_transactions.parquet')
    sample_paths = list(p.glob('sample_submission*.csv'))
    if len(sample_paths) != 1:
        raise ValueError('Expected exactly one sample submission')
    # The sample probabilities are intentionally never read.
    sample = pd.read_csv(sample_paths[0], usecols=['signal_id'])
    for s, t in [(train, tx_train), (test, tx_test)]:
        assert s.signal_id.notna().all() and s.signal_id.is_unique
        assert s.signal_sanasi.notna().all()
        assert t.signal_id.notna().all()
        assert set(t.signal_id) <= set(s.signal_id)
        t['tranzaksiya_vaqti'] = pd.to_datetime(t.tranzaksiya_vaqti, errors='raise')
        assert t.tranzaksiya_vaqti.notna().all()
    assert not set(train.signal_id) & set(test.signal_id)
    assert set(train.eskalatsiya.unique()) == {0, 1}
    assert sample.signal_id.is_unique and set(sample.signal_id) == set(test.signal_id)
    return train, test, tx_train, tx_test, sample


def historical_transactions(signals, transactions):
    """Conservative cutoff: date-only alerts are interpreted as midnight.

    No later timestamp, including later hours of the same date, is used.
    Excluded-transaction counts are audit information, never model features.
    """
    t = transactions.merge(signals[['signal_id', 'signal_sanasi']], on='signal_id', how='left', validate='many_to_one')
    assert len(t) == len(transactions) and t.signal_sanasi.notna().all()
    t['age_days'] = (t.signal_sanasi - t.tranzaksiya_vaqti).dt.total_seconds() / 86400
    t = t.loc[t.age_days >= 0].copy()
    t['day'] = t.tranzaksiya_vaqti.dt.floor('D')
    t['hour'] = t.tranzaksiya_vaqti.dt.hour
    t['dow'] = t.tranzaksiya_vaqti.dt.dayofweek
    for col in ['kirim_chiqim', 'tranzaksiya_turi']:
        t[col] = t[col].astype('category')
    return t


def save_json(value, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, default=str, allow_nan=False), encoding='utf-8')


def inspect_data(data_dir=None):
    ARTIFACTS.mkdir(exist_ok=True)
    train, test, tx_train, tx_test, sample = load_data(data_dir)
    report = {'target_counts': train.eskalatsiya.value_counts().to_dict(), 'target_rate': float(train.eskalatsiya.mean()), 'splits': {}}
    for name, s, t in [('train', train, tx_train), ('test', test, tx_test)]:
        counts = t.groupby('signal_id').size().reindex(s.signal_id, fill_value=0)
        dates = s.set_index('signal_id').signal_sanasi
        age = (t.signal_id.map(dates) - t.tranzaksiya_vaqti).dt.total_seconds() / 86400
        valid = age >= 0
        hist_counts = t.loc[valid].groupby('signal_id').size().reindex(s.signal_id, fill_value=0)
        amt = t.miqdor_indeksi
        report['splits'][name] = {
            'signal_shape': list(s.shape), 'transaction_shape': list(t.shape),
            'unique_signals': int(s.signal_id.nunique()), 'unique_transaction_signals': int(t.signal_id.nunique()),
            'duplicate_signals': int(s.signal_id.duplicated().sum()), 'duplicate_transaction_rows': int(t.duplicated().sum()),
            'signal_missing': s.isna().sum().to_dict(), 'transaction_missing': t.isna().sum().to_dict(),
            'signal_dtypes': s.dtypes.astype(str).to_dict(), 'transaction_dtypes': t.dtypes.astype(str).to_dict(),
            'alert_date_min': str(s.signal_sanasi.min()), 'alert_date_max': str(s.signal_sanasi.max()),
            'transaction_date_min': str(t.tranzaksiya_vaqti.min()), 'transaction_date_max': str(t.tranzaksiya_vaqti.max()),
            'transactions_per_signal': counts.describe(percentiles=[.01,.1,.25,.5,.75,.9,.95,.99]).to_dict(),
            'historical_transactions_per_signal': hist_counts.describe(percentiles=[.01,.1,.25,.5,.75,.9,.95,.99]).to_dict(),
            'signals_without_transactions': int((counts == 0).sum()), 'signals_without_historical_transactions': int((hist_counts == 0).sum()),
            'transactions_before_timestamp': int((age > 0).sum()), 'transactions_at_timestamp': int((age == 0).sum()), 'transactions_after_timestamp': int((age < 0).sum()),
            'transactions_on_alert_calendar_date': int((t.tranzaksiya_vaqti.dt.normalize() == t.signal_id.map(dates).dt.normalize()).sum()),
            'non_midnight_transactions': int((t.tranzaksiya_vaqti != t.tranzaksiya_vaqti.dt.normalize()).sum()),
            'age_days_quantiles': age.quantile([0,.01,.1,.5,.9,.99,1]).to_dict(),
            'amount_summary': amt.describe(percentiles=[.001,.01,.1,.25,.5,.75,.9,.99,.999]).to_dict(),
            'nonfinite_amount_count': int((~np.isfinite(amt)).sum()),
            'category_counts': {c: t[c].value_counts(dropna=False).to_dict() for c in ['kirim_chiqim','tranzaksiya_turi']},
            'historical_category_proportions': {c: t.loc[valid,c].value_counts(normalize=True).to_dict() for c in ['kirim_chiqim','tranzaksiya_turi']},
        }
    monthly = train.assign(month=train.signal_sanasi.dt.to_period('M').astype(str)).groupby('month').eskalatsiya.agg(['size','mean','sum'])
    monthly.to_csv(ARTIFACTS / 'monthly_target.csv')
    report['monthly_target'] = monthly.reset_index().to_dict('records')
    report['leakage_policy'] = 'Only transactions timestamped at or before alert midnight. Excluded rows and their counts never enter features. signal_id is only a join key. No target encodings or external data.'
    report['source_sha256'] = {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in locate_data(data_dir).iterdir() if f.suffix in ['.csv','.parquet']}
    save_json(report, ARTIFACTS / 'data_audit.json')
    print(json.dumps(report, indent=2, default=str), flush=True)
    return report


if __name__ == '__main__':
    inspect_data()
