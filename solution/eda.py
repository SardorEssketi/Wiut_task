"""Aggregate-only EDA outputs suitable for a public website."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
from .data import ROOT, ARTIFACTS, load_data, historical_transactions, save_json
from .features import prepare_features

PUBLIC = ROOT/'eda_site'/'data'
PLOTS = ROOT/'eda_site'/'plots'
EXPLORE = ['all_count','all_mean','all_std','all_max','all_abs_mean','all_q95','daily_count_max','all_kirim_share','all_xalqaro_share','w1_count','w3_count','w7_count','w7_count_share','change7_37_count_rate_ratio','change7_37_mean_diff','most_active_day_share','gap_burstiness','last5_amount_mean']


def save_plot(name, title, xlabel, ylabel):
    plt.title(title,loc='left',fontweight='bold',pad=15)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(PLOTS/f'{name}.png',dpi=150,bbox_inches='tight')
    plt.close()


def generate_eda():
    PUBLIC.mkdir(parents=True,exist_ok=True)
    PLOTS.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'figure.figsize':(9,4.5),'axes.spines.top':False,'axes.spines.right':False,'axes.prop_cycle':plt.cycler(color=['#087f8c','#ee8c42','#5863a7']),'font.size':10})
    x,z,groups = prepare_features()
    train,test,tx_train,tx_test,_ = load_data()
    y = train.set_index('signal_id').eskalatsiya.reindex(x.index)
    audit = json.loads((ARTIFACTS/'data_audit.json').read_text())
    save_json(audit,PUBLIC/'overview.json')
    count_rows, type_rows, amount_rows, timing_rows, volume_rows = [],[],[],[],[]
    bins = np.array([-np.inf,-10,-5,-3,-2,-1,0,1,2,3,5,10,np.inf])
    age_bins = [-.000001,.25,.5,1,2,3,7,14,30,60,90,180,365,10000]
    for split,s,t in [('train',train,tx_train),('test',test,tx_test)]:
        h = historical_transactions(s,t)
        h['target_group'] = h.signal_id.map(y).map({0:'Dismissed',1:'Escalated'}).fillna('Unlabeled test')
        for label,frame in [('All',h),*list(h.groupby('target_group',observed=True))]:
            for field in ['kirim_chiqim','tranzaksiya_turi']:
                for category,count in frame[field].value_counts().items():
                    type_rows.append({'split':split,'target_group':label,'field':field,'category':category,'count':int(count),'fraction':float(count/len(frame))})
            hist,_ = np.histogram(frame.miqdor_indeksi.dropna(),bins)
            for i,count in enumerate(hist):
                amount_rows.append({'split':split,'target_group':label,'bin':f'{bins[i]:g} to {bins[i+1]:g}','order':i,'count':int(count),'fraction':float(count/max(hist.sum(),1))})
            agehist,_ = np.histogram(frame.age_days,age_bins)
            for i,count in enumerate(agehist):
                timing_rows.append({'split':split,'target_group':label,'bin':f'{max(0,age_bins[i]):g} to {age_bins[i+1]:g} days','order':i,'count':int(count),'fraction':float(count/max(agehist.sum(),1))})
        volume = h.groupby(h.tranzaksiya_vaqti.dt.to_period('M').astype(str),observed=True).size()
        for month,count in volume.items():
            volume_rows.append({'split':split,'month':month,'count':int(count),'per_signal':float(count/len(s))})
    pd.DataFrame(type_rows).to_csv(PUBLIC/'categories.csv',index=False)
    pd.DataFrame(amount_rows).to_csv(PUBLIC/'amount_histogram.csv',index=False)
    pd.DataFrame(timing_rows).to_csv(PUBLIC/'timing_histogram.csv',index=False)
    pd.DataFrame(volume_rows).to_csv(PUBLIC/'monthly_volume.csv',index=False)
    monthly = train.assign(month=train.signal_sanasi.dt.to_period('M').astype(str)).groupby('month').eskalatsiya.agg(['size','mean','sum']).reset_index()
    test_month = test.assign(month=test.signal_sanasi.dt.to_period('M').astype(str)).groupby('month').size().rename('test_signals')
    monthly = monthly.join(test_month,on='month').rename(columns={'size':'train_signals','mean':'target_rate','sum':'escalated'})
    monthly['target_se'] = np.sqrt(monthly.target_rate*(1-monthly.target_rate)/monthly.train_signals)
    monthly.to_csv(PUBLIC/'monthly_alerts.csv',index=False)

    feature_bins, feature_summaries, distributions, drift = [],[],[],[]
    for col in x:
        a,b = x[col].dropna(),z[col].dropna()
        if len(a)>0 and len(b)>0:
            ks = ks_2samp(a,b)
            drift.append({'feature':col,'group':groups[col],'ks_statistic':float(ks.statistic),'ks_pvalue':float(ks.pvalue),'train_missing':float(x[col].isna().mean()),'test_missing':float(z[col].isna().mean()),'train_mean':float(a.mean()),'test_mean':float(b.mean()),'train_unique':int(a.nunique()),'train_dominant_fraction':float(x[col].value_counts(dropna=False,normalize=True).iloc[0])})
        if col not in EXPLORE:
            continue
        for label,mask in [('Dismissed',y==0),('Escalated',y==1)]:
            vals = x.loc[mask,col]
            feature_summaries.append({'feature':col,'group':label,'mean':float(vals.mean()),'median':float(vals.median()),'q25':float(vals.quantile(.25)),'q75':float(vals.quantile(.75))})
        _, edges = pd.qcut(x[col].dropna(),10,duplicates='drop',retbins=True)
        if len(edges)<3:
            continue
        edges[0],edges[-1] = -np.inf,np.inf
        bucket = pd.cut(x[col],edges)
        grouped = pd.DataFrame({'bin':bucket,'target':y,'value':x[col]}).groupby('bin',observed=True).agg(n=('target','size'),target_rate=('target','mean'),mean_value=('value','mean'))
        for order,(interval,row) in enumerate(grouped.iterrows()):
            rate = float(row.target_rate)
            feature_bins.append({'feature':col,'bin':str(interval),'order':order,'n':int(row.n),'target_rate':rate,'mean_value':float(row.mean_value),'se':float(np.sqrt(rate*(1-rate)/row.n))})
        for split,vals in [('train',x[col]),('test',z[col])]:
            freq = pd.cut(vals,edges).value_counts(sort=False,normalize=True)
            for order,(interval,fraction) in enumerate(freq.items()):
                distributions.append({'feature':col,'split':split,'bin':str(interval),'order':order,'fraction':float(fraction)})
    fb = pd.DataFrame(feature_bins)
    fb.to_csv(PUBLIC/'binned_target.csv',index=False)
    pd.DataFrame(feature_summaries).to_csv(PUBLIC/'feature_summary.csv',index=False)
    pd.DataFrame(distributions).to_csv(PUBLIC/'feature_distributions.csv',index=False)
    drift = pd.DataFrame(drift).sort_values('ks_statistic',ascending=False)
    drift.to_csv(ARTIFACTS/'feature_drift.csv',index=False)
    drift.to_csv(PUBLIC/'feature_drift.csv',index=False)
    # Pairwise redundancy report is diagnostic only; model transformations remain fold-local.
    corr = x.corr().abs()
    pairs = np.argwhere(np.triu(corr.to_numpy()>0.995,k=1))
    pd.DataFrame([{'feature_a':x.columns[i],'feature_b':x.columns[j],'abs_correlation':float(corr.iloc[i,j])} for i,j in pairs]).to_csv(ARTIFACTS/'high_correlations.csv',index=False)
    save_json({'constant_features':x.columns[x.nunique(dropna=False)<=1].tolist(),'near_constant_features':drift.loc[drift.train_dominant_fraction>.995,'feature'].tolist(),'high_correlation_pairs':int(len(pairs))},ARTIFACTS/'feature_quality.json')

    plt.figure(); plt.bar(['Dismissed','Escalated'],[(y==0).sum(),(y==1).sum()]); save_plot('target','Escalation target balance','Outcome','Alerts')
    plt.figure()
    for label,mask in [('Dismissed',y==0),('Escalated',y==1)]:
        plt.hist(x.loc[mask,'all_count'],bins=50,density=True,histtype='step',linewidth=2,label=label)
    plt.legend(); save_plot('activity','Historical transactions per alert','Transaction count','Density')
    plt.figure()
    for split,g in pd.DataFrame(volume_rows).groupby('split'):
        plt.plot(g.month,g.per_signal,label=split)
    plt.xticks(rotation=60); plt.legend(); save_plot('volume','Historical transaction volume by month','Transaction month','Transactions / split alert count')
    for field,name in [('kirim_chiqim','direction'),('tranzaksiya_turi','types')]:
        data = pd.DataFrame(type_rows)
        table = data.loc[(data.split=='train')&(data.target_group!='All')&(data.field==field)].pivot(index='category',columns='target_group',values='fraction')
        table.plot.bar(rot=0); save_plot(name,'Historical transaction composition by outcome','Category','Fraction of transactions')
    plt.figure()
    for label,g in pd.DataFrame(amount_rows).query("split == 'train' and target_group != 'All'").groupby('target_group'):
        plt.plot(g.bin,g.fraction,marker='o',label=label)
    plt.xticks(rotation=45); plt.legend(); save_plot('amount','Standardized amount index by outcome','Index interval (not currency)','Fraction of transactions')
    plt.figure()
    for label,g in pd.DataFrame(timing_rows).query("split == 'train' and target_group != 'All'").groupby('target_group'):
        plt.plot(g.bin,g.fraction,marker='o',label=label)
    plt.xticks(rotation=60); plt.legend(); save_plot('recency','When historical transactions occurred','Time before alert','Fraction of transactions')
    plt.figure(); plt.errorbar(monthly.month,monthly.target_rate,yerr=1.96*monthly.target_se,marker='o'); plt.xticks(rotation=60); save_plot('target_time','Escalation rate over alert time (approx. 95% intervals)','Alert month','Escalation rate')
    for col,name in [('w7_count','recent_target'),('change7_37_count_rate_ratio','burst_target'),('all_std','volatility_target')]:
        g = fb[fb.feature==col]
        plt.figure(); plt.errorbar(g.mean_value,g.target_rate,yerr=1.96*g.se,marker='o'); save_plot(name,f'Escalation rate vs {col}','Mean feature value in bin','Escalation rate')
    plt.figure()
    for split,g in pd.DataFrame(distributions).query("feature == 'all_count'").groupby('split'):
        plt.plot(g.order,g.fraction,marker='o',label=split)
    plt.legend(); save_plot('train_test','Train and test activity distributions','Training-defined quantile bin','Fraction of alerts')
    print(f'EDA exported to {PUBLIC} and {PLOTS}',flush=True)


if __name__ == '__main__':
    generate_eda()
