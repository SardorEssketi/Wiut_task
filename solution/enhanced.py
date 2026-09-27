"""Target-free refinements: within-alert type contrasts and distribution shape."""
import hashlib
import json
import gc
from pathlib import Path
import numpy as np
import pandas as pd
from .data import ARTIFACTS, historical_transactions, load_data, save_json
from .features import prepare_features, TYPES, DIRECTIONS


def enrich(signals,transactions,base):
    h = historical_transactions(signals,transactions)
    idx = base.index
    columns = {}
    def put(name,s):
        columns[name] = s.reindex(idx).astype(np.float32) if isinstance(s,pd.Series) else np.asarray(s,dtype=np.float32)
    for prefix in ['all','w90','w30','w7']:
        for i,a in enumerate(TYPES):
            for b in TYPES[i+1:]:
                put(f'{prefix}_{a}_minus_{b}_mean',base[f'{prefix}_{a}_amount_mean']-base[f'{prefix}_{b}_amount_mean'])
                put(f'{prefix}_{a}_minus_{b}_mean_z',(base[f'{prefix}_{a}_amount_mean']-base[f'{prefix}_{b}_amount_mean'])/(base[f'{prefix}_std']+0.1))
                put(f'{prefix}_{a}_to_{b}_std',(base[f'{prefix}_{a}_amount_std']+0.1)/(base[f'{prefix}_{b}_amount_std']+0.1))
            put(f'{prefix}_{a}_direction_amount_difference',base[f'{prefix}_kirim_{a}_mean']-base[f'{prefix}_chiqim_{a}_mean'])
            put(f'{prefix}_{a}_incoming_fraction',(base[f'{prefix}_kirim_{a}_count']+0.5)/(base[f'{prefix}_{a}_count']+1))
        put(f'{prefix}_incoming_minus_outgoing_mean',base[f'{prefix}_kirim_amount_mean']-base[f'{prefix}_chiqim_amount_mean'])
    for kind in TYPES:
        g = h.loc[h.tranzaksiya_turi==kind].groupby('signal_id',observed=True).miqdor_indeksi
        for q,s in g.quantile([.01,.1,.25,.5,.75,.9,.99]).unstack().items():
            put(f'type_{kind}_q{int(q*100)}',s)
        put(f'type_{kind}_min',g.min())
        put(f'type_{kind}_skew',g.skew())
        for direction in DIRECTIONS:
            dg = h.loc[(h.tranzaksiya_turi==kind)&(h.kirim_chiqim==direction)].groupby('signal_id',observed=True).miqdor_indeksi
            for stat in ['min','max','std','median']:
                put(f'type_{kind}_{direction}_{stat}',dg.agg(stat))
        for recent in [7,30,90]:
            put(f'type_{kind}_recent{recent}_mean_change',base[f'w{recent}_{kind}_amount_mean']-base[f'all_{kind}_amount_mean'])
    # Amount tails and transformed distribution shape are ordinary descriptive features.
    for threshold in [-2.8,-2,-1,0,1,2,3]:
        freq = h.loc[h.miqdor_indeksi>threshold].groupby('signal_id',observed=True).size().reindex(idx,fill_value=0)
        put(f'amount_above_{str(threshold).replace(".","p").replace("-","neg")}_share',freq/(base.all_count+1))
    # Subday windows are justified by timestamp precision and visible pre-alert clustering.
    for minutes in [1,5,15,60,180]:
        f = h.loc[h.age_days<=minutes/1440]
        g = f.groupby('signal_id',observed=True)
        put(f'minutes{minutes}_count',g.size().reindex(idx,fill_value=0))
        put(f'minutes{minutes}_share',g.size().reindex(idx,fill_value=0)/(base.all_count+1))
        for stat in ['mean','std','min','max']:
            put(f'minutes{minutes}_amount_{stat}',g.miqdor_indeksi.agg(stat))
    added = pd.DataFrame(columns,index=idx).replace([np.inf,-np.inf],np.nan)
    return pd.concat([base,added],axis=1),{c:('microtime' if c.startswith('minutes') else 'contrasts') for c in added}


def prepare_enhanced(force=False):
    x,z,groups = prepare_features()
    digest = hashlib.sha256(Path(__file__).read_bytes()+(ARTIFACTS/'feature_metadata.json').read_bytes()).hexdigest()
    meta = ARTIFACTS/'enhanced_metadata.json'
    if not force and meta.exists() and json.loads(meta.read_text())['fingerprint']==digest:
        return pd.read_parquet(ARTIFACTS/'train_enhanced.parquet'),pd.read_parquet(ARTIFACTS/'test_enhanced.parquet'),json.loads(meta.read_text())['groups']
    train,test,tx_train,tx_test,_ = load_data()
    print('Building type contrasts and microtime features',flush=True)
    x,new_groups = enrich(train,tx_train,x)
    del tx_train
    gc.collect()
    z,_ = enrich(test,tx_test,z)
    groups.update(new_groups)
    assert list(x.columns)==list(z.columns)
    x.to_parquet(ARTIFACTS/'train_enhanced.parquet')
    z.to_parquet(ARTIFACTS/'test_enhanced.parquet')
    save_json({'fingerprint':digest,'groups':groups,'train_shape':list(x.shape)},meta)
    print(f'Enhanced features: {x.shape}',flush=True)
    return x,z,groups


if __name__ == '__main__':
    prepare_enhanced()
