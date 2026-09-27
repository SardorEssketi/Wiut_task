"""Target-free, per-alert historical feature engineering (identical for both splits)."""
import gc
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from .data import ARTIFACTS, historical_transactions, load_data, save_json, locate_data

VERSION = 'historical_v1'
WINDOWS = [1, 2, 3, 7, 14, 30, 60, 90]
DIRECTIONS = ['kirim', 'chiqim']
TYPES = ['karta', 'bank_otkazmasi', 'naqd', 'xalqaro']


def build_features(signals, transactions):
    t = historical_transactions(signals, transactions)
    t = t.sort_values(['signal_id', 'tranzaksiya_vaqti'], kind='stable')
    t['amount'] = t.miqdor_indeksi.astype(float)
    t['abs_amount'] = t.amount.abs()
    t['positive'] = (t.amount > 0).astype(float)
    t['negative'] = (t.amount < 0).astype(float)
    t['weekend'] = (t.dow >= 5).astype(float)
    t['night'] = ((t.hour < 6) | (t.hour >= 22)).astype(float)
    t['business'] = ((t.hour >= 9) & (t.hour < 18) & (t.dow < 5)).astype(float)
    idx = pd.Index(signals.signal_id, name='signal_id')
    cols, groups = {}, {}

    def add(name, values, group, zero=False):
        if isinstance(values, pd.Series):
            values = values.reindex(idx)
            if zero:
                values = values.fillna(0)
            values = values.to_numpy()
        cols[name] = np.asarray(values, dtype=np.float32)
        groups[name] = group

    def aggregate(frame, prefix, group, rich=False):
        g = frame.groupby('signal_id', observed=True)
        a = g.amount.agg(['count', 'mean', 'median', 'std', 'min', 'max', 'sum'])
        for stat in a:
            add(prefix + '_' + stat, a[stat], group, zero=stat in ['count','sum'])
        for stat in ['mean','max']:
            add(prefix + '_abs_' + stat, g.abs_amount.agg(stat), group)
        add(prefix + '_positive_share', g.positive.mean(), group)
        add(prefix + '_negative_share', g.negative.mean(), group)
        add(prefix + '_active_days', g.day.nunique(), group, zero=True)
        add(prefix + '_recency', g.age_days.min(), group)
        if rich:
            q = g.amount.quantile([.1,.25,.75,.9,.95,.99]).unstack()
            for quantile in q:
                add(prefix + f'_q{int(quantile*100):02d}', q[quantile], group)
            add(prefix + '_skew', g.amount.skew(), group)
            add(prefix + '_range', a['max'] - a['min'], group)
            add(prefix + '_iqr', q[.75]-q[.25], group)
        return g

    full = aggregate(t, 'all', 'baseline', rich=True)
    add('history_oldest_days', full.age_days.max(), 'baseline')
    add('history_span_days', full.age_days.max()-full.age_days.min(), 'baseline')
    add('first_tx_epoch_days', (full.tranzaksiya_vaqti.min()-pd.Timestamp('2025-01-01')).dt.total_seconds()/86400, 'calendar')
    add('last_tx_epoch_days', (full.tranzaksiya_vaqti.max()-pd.Timestamp('2025-01-01')).dt.total_seconds()/86400, 'calendar')
    add('has_history', (full.size()>0).astype(float), 'baseline', zero=True)
    daily = t.groupby(['signal_id','day'], observed=True).size().rename('daily_count').reset_index()
    dg = daily.groupby('signal_id').daily_count
    for stat in ['mean','median','std','max']:
        add('daily_count_'+stat, dg.agg(stat), 'baseline')
    for col in ['weekend','night','business']:
        add(col+'_share', full[col].mean(), 'calendar')
    for day in range(7):
        add(f'dow_{day}_share', t.loc[t.dow == day].groupby('signal_id').size() / full.size(), 'calendar', zero=True)

    # Central moments are stable on standardized values, with a variance floor.
    mean = full.amount.transform('mean')
    t['centered2'] = (t.amount-mean)**2
    t['centered4'] = t.centered2**2
    g = t.groupby('signal_id', observed=True)
    add('all_kurtosis', g.centered4.mean()/(g.centered2.mean()**2+1e-8)-3, 'baseline')
    hist_std = full.amount.transform('std').fillna(0)
    t['own_z_extreme'] = ((t.amount-mean).abs() > 2*(hist_std+1e-6)).astype(float)
    add('own_z_extreme_share', t.groupby('signal_id').own_z_extreme.mean(), 'baseline')
    for threshold in [1,2,3,5]:
        add(f'abs_above_{threshold}_share', t.loc[t.abs_amount>threshold].groupby('signal_id').size()/full.size(), 'baseline', zero=True)

    def categorical(frame, prefix):
        n = frame.groupby('signal_id', observed=True).size()
        for field, levels, group in [('kirim_chiqim', DIRECTIONS, 'direction'), ('tranzaksiya_turi', TYPES, 'types')]:
            shares = []
            for level in levels:
                sub = frame.loc[frame[field] == level]
                sg = sub.groupby('signal_id', observed=True)
                count = sg.size().reindex(n.index, fill_value=0)
                p = count/n
                shares.append(p)
                add(f'{prefix}_{level}_count', count, group, zero=True)
                add(f'{prefix}_{level}_share', p, group, zero=True)
                for stat in ['mean','std','max']:
                    add(f'{prefix}_{level}_amount_{stat}', sg.amount.agg(stat), group)
                add(f'{prefix}_{level}_recency', sg.age_days.min(), group)
            probabilities = pd.concat(shares, axis=1)
            add(f'{prefix}_{field}_entropy', -(probabilities*np.log(probabilities.clip(lower=1e-12))).sum(axis=1), 'diversity')
            add(f'{prefix}_{field}_concentration', (probabilities**2).sum(axis=1), 'diversity')
            add(f'{prefix}_{field}_dominant_share', probabilities.max(axis=1), 'diversity')
            add(f'{prefix}_{field}_diversity', (probabilities>0).sum(axis=1), 'diversity')
        for direction in DIRECTIONS:
            for kind in TYPES:
                sg = frame.loc[(frame.kirim_chiqim == direction) & (frame.tranzaksiya_turi == kind)].groupby('signal_id', observed=True)
                add(f'{prefix}_{direction}_{kind}_count', sg.size(), 'types', zero=True)
                add(f'{prefix}_{direction}_{kind}_mean', sg.amount.mean(), 'types')
        incoming = cols[f'{prefix}_kirim_count']
        outgoing = cols[f'{prefix}_chiqim_count']
        add(prefix+'_direction_ratio', (incoming+1)/(outgoing+1), 'direction')
        add(prefix+'_direction_difference', incoming-outgoing, 'direction')

    categorical(t, 'all')
    for w in WINDOWS:
        frame = t.loc[t.age_days <= w]
        aggregate(frame, f'w{w}', 'recent', rich=w in [3,7,30])
        if w in [1,3,7,14,30,90]:
            categorical(frame, f'w{w}')
        add(f'w{w}_count_share', cols[f'w{w}_count']/(cols['all_count']+1), 'recent')
        add(f'w{w}_daily_intensity', cols[f'w{w}_count']/w, 'recent')
    for hours in [6,12]:
        frame = t.loc[t.age_days <= hours/24]
        aggregate(frame, f'h{hours}', 'recent')

    # Recent and older intervals are disjoint; only historical information used.
    for recent, older in [(3,30),(7,37),(14,60),(30,90)]:
        prefix = f'old{recent}_{older}'
        frame = t.loc[(t.age_days>recent) & (t.age_days<=older)]
        aggregate(frame, prefix, 'change')
        for field, levels in [('kirim_chiqim',DIRECTIONS),('tranzaksiya_turi',TYPES)]:
            og = frame.groupby('signal_id', observed=True).size()
            for level in levels:
                share = frame.loc[frame[field] == level].groupby('signal_id', observed=True).size().reindex(og.index, fill_value=0)/og
                add(prefix+'_'+level+'_share', share, 'change')
                add(f'change{recent}_{older}_{level}_share', cols[f'w{recent}_{level}_share']-cols[prefix+'_'+level+'_share'], 'change')
        add(f'change{recent}_{older}_count_rate_ratio', (cols[f'w{recent}_count']/recent+0.1)/(cols[prefix+'_count']/(older-recent)+0.1), 'change')
        for stat in ['mean','max','std','abs_mean']:
            add(f'change{recent}_{older}_{stat}_diff', cols[f'w{recent}_{stat}']-cols[prefix+'_'+stat], 'change')
        add(f'change{recent}_{older}_mean_z', (cols[f'w{recent}_mean']-cols[prefix+'_mean'])/(cols[prefix+'_std']+0.1), 'change')
    add('most_active_day_share', cols['daily_count_max']/(cols['all_count']+1), 'recent')

    # Ties preserve source order; tied timestamp fraction is audited separately.
    t['gap_hours'] = t.groupby('signal_id', observed=True).tranzaksiya_vaqti.diff().dt.total_seconds()/3600
    t['reverse_rank'] = t.groupby('signal_id', observed=True).cumcount(ascending=False)
    g = t.groupby('signal_id', observed=True)
    for stat in ['mean','median','std','min','max']:
        add('gap_hours_'+stat, g.gap_hours.agg(stat), 'sequence')
    add('gap_burstiness', (cols['gap_hours_std']-cols['gap_hours_mean'])/(cols['gap_hours_std']+cols['gap_hours_mean']+1e-6), 'sequence')
    for field in ['kirim_chiqim','tranzaksiya_turi']:
        prev = g[field].shift()
        t[field+'_switch'] = np.where(prev.notna(), (t[field]!=prev).astype(float), np.nan)
        add(field+'_switch_share', t.groupby('signal_id', observed=True)[field+'_switch'].mean(), 'sequence')
    for n in [3,5,10,20]:
        recent = t.loc[t.reverse_rank<n]
        sg = recent.groupby('signal_id', observed=True)
        for stat in ['mean','std','min','max']:
            add(f'last{n}_amount_{stat}', sg.amount.agg(stat), 'sequence')
        add(f'last{n}_span_days', sg.age_days.max()-sg.age_days.min(), 'sequence')
        add(f'last{n}_incoming_share', recent.assign(incoming=(recent.kirim_chiqim=='kirim').astype(float)).groupby('signal_id').incoming.mean(), 'sequence')
    for rank in range(5):
        last = t.loc[t.reverse_rank == rank].set_index('signal_id')
        for col in ['amount','age_days','gap_hours']:
            add(f'last_tx_{rank}_{col}', last[col], 'sequence')
        add(f'last_tx_{rank}_direction', last.kirim_chiqim.astype(object).map({v:i for i,v in enumerate(DIRECTIONS)}), 'sequence')
        add(f'last_tx_{rank}_type', last.tranzaksiya_turi.astype(object).map({v:i for i,v in enumerate(TYPES)}), 'sequence')

    s = signals.set_index('signal_id').reindex(idx)
    for name, values in [('alert_month',s.signal_sanasi.dt.month),('alert_dow',s.signal_sanasi.dt.dayofweek),('alert_day',s.signal_sanasi.dt.day),('alert_epoch_days',(s.signal_sanasi-pd.Timestamp('2025-01-01')).dt.days)]:
        add(name, values, 'calendar')
    result = pd.DataFrame(cols, index=idx).replace([np.inf,-np.inf],np.nan)
    assert len(result) == len(signals) and result.index.is_unique
    return result, groups


def prepare_features(force=False):
    ARTIFACTS.mkdir(exist_ok=True)
    path = ARTIFACTS / 'feature_metadata.json'
    import json
    source = locate_data()
    fingerprint = hashlib.sha256(Path(__file__).read_bytes() + (Path(__file__).parent/'data.py').read_bytes())
    for name in ['train_signals.csv','test_signals.csv','train_transactions.parquet','test_transactions.parquet']:
        fingerprint.update((source/name).read_bytes())
    digest = fingerprint.hexdigest()
    if not force and path.exists() and json.loads(path.read_text())['fingerprint'] == digest:
        return pd.read_parquet(ARTIFACTS/'train_features.parquet'), pd.read_parquet(ARTIFACTS/'test_features.parquet'), json.loads(path.read_text())['groups']
    train, test, tx_train, tx_test, _ = load_data()
    print('Building training features', flush=True)
    x, groups = build_features(train, tx_train)
    del tx_train
    gc.collect()
    print('Building test features', flush=True)
    z, test_groups = build_features(test, tx_test)
    assert list(x.columns) == list(z.columns) and groups == test_groups
    x.to_parquet(ARTIFACTS/'train_features.parquet')
    z.to_parquet(ARTIFACTS/'test_features.parquet')
    save_json({'version':VERSION, 'fingerprint':digest, 'groups':groups, 'train_shape':list(x.shape),'test_shape':list(z.shape)},path)
    print(f'Features complete: {x.shape}, {z.shape}', flush=True)
    return x,z,groups


if __name__ == '__main__':
    prepare_features()
