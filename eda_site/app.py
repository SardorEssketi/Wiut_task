"""Public-ready report. Uses aggregate artifacts only; no raw data required."""
from pathlib import Path
import json
import pandas as pd
import plotly.express as px
import streamlit as st

BASE = Path(__file__).resolve().parent
DATA = BASE/'data'
st.set_page_config(page_title='Alert intelligence | EDA & validation', page_icon='◈', layout='wide')
st.markdown('''<style>
.block-container {max-width:1200px;padding-top:2.5rem;}
h1,h2,h3 {letter-spacing:-.025em;}
[data-testid="stMetric"] {background:#eef6f6;padding:18px;border-radius:10px;}
</style>''',unsafe_allow_html=True)


@st.cache_data
def table(name):
    return pd.read_csv(DATA/name)


@st.cache_data
def document(name):
    return json.loads((DATA/name).read_text(encoding='utf-8'))


def chart(fig):
    fig.update_layout(template='plotly_white',font_family='Arial',legend_title_text='',margin=dict(l=15,r=15,t=40,b=20),colorway=['#087f8c','#ee8c42','#5863a7'])
    st.plotly_chart(fig,width='stretch')


if not (DATA/'overview.json').exists():
    st.error('Report data are missing. Run `python -m solution.data` and `python -m solution.eda` from the project root.')
    st.stop()

overview = document('overview.json')
train,test = overview['splits']['train'],overview['splits']['test']
page = st.sidebar.radio('Explore the report',['Overview','Target & time','Transaction behavior','Outcome differences','Validation & models','Conclusions'])
st.sidebar.caption('Synthetic financial monitoring data · historical features only · no hidden test labels')
st.title('Alert intelligence')
st.caption('Understanding historical transaction behavior and predicting escalation')

if page == 'Overview':
    st.markdown('Financial monitoring generates alerts from customer transaction activity. The task is to estimate each alert’s probability of escalation (`eskalatsiya = 1`). Dismissed alerts have target 0.')
    a,b,c,d = st.columns(4)
    a.metric('Training alerts',f"{train['unique_signals']:,}")
    b.metric('Test alerts',f"{test['unique_signals']:,}")
    c.metric('Training transactions',f"{train['transaction_shape'][0]:,}")
    d.metric('Escalation rate',f"{overview['target_rate']:.2%}")
    st.subheader('What success means')
    st.write('ROC-AUC measures how often a randomly chosen escalated alert receives a higher score than a randomly chosen dismissed alert. AUC 0.5 is chance ranking; 1.0 is perfect ranking. Predictions stay continuous—no classification threshold is required.')
    st.subheader('Two tables, one alert history')
    st.code('Signals: one row per signal_id\n    │\n    └── Transactions: many historical rows per signal_id',language=None)
    st.dataframe(pd.DataFrame([
        {'Table':'Signals','Columns':'signal_id, signal_sanasi, eskalatsiya (training only)','Grain':'One alert'},
        {'Table':'Transactions','Columns':'signal_id, tranzaksiya_vaqti, kirim_chiqim, tranzaksiya_turi, miqdor_indeksi','Grain':'One transaction'},
    ]),hide_index=True,width='stretch')
    st.write(f"Both alert splits span {train['alert_date_min'][:10]} to {train['alert_date_max'][:10]}. Historical coverage extends up to approximately 180 days before an alert.")
    st.info(f"Leakage control: {train['transactions_after_timestamp']:,} training and {test['transactions_after_timestamp']:,} test transactions after alert midnight are excluded. Their counts are also excluded from predictors. Transactions exactly at the cutoff are retained.")
    st.write('The model summarizes activity, amount-index behavior, direction, transaction type, recency, and changes from older history. All training and test features use the same code. The standardized amount index is not a currency amount.')
    with st.expander('Data integrity audit'):
        st.write('No missing source values, duplicate alert IDs, or duplicate transaction rows were found. Every alert retains at least one historical transaction. IDs are join keys, never predictors.')
        st.json({'train':train,'test':test},expanded=False)

elif page == 'Target & time':
    st.subheader('Class balance')
    counts = pd.DataFrame({'Outcome':['Dismissed','Escalated'],'Alerts':[overview['target_counts']['0'],overview['target_counts']['1']]})
    chart(px.bar(counts,x='Outcome',y='Alerts',color='Outcome',text='Alerts'))
    st.write(f"Only {overview['target_rate']:.2%} of alerts are escalated. Stratification preserves this minority class in each validation fold; model selection uses ROC-AUC.")
    monthly = table('monthly_alerts.csv')
    monthly['95% half-width'] = 1.96*monthly.target_se
    chart(px.line(monthly,x='month',y='target_rate',error_y='95% half-width',markers=True,title='Target rate by alert month'))
    st.caption('Error bars are approximate binomial 95% intervals. Monthly fluctuations do not by themselves demonstrate a causal time effect.')
    chart(px.line(monthly,x='month',y=['train_signals','test_signals'],markers=True,title='Train and test alert coverage'))
    st.write('Both splits cover the same calendar period. This supports random stratified CV as the primary estimate; expanding temporal holdouts assess performance when predicting later alerts.')

elif page == 'Transaction behavior':
    st.subheader('Activity, composition, and timing')
    dist = table('feature_distributions.csv')
    feature = st.selectbox('Alert-level distribution',sorted(dist.feature.unique()),index=sorted(dist.feature.unique()).index('all_count'))
    chart(px.bar(dist[dist.feature==feature],x='bin',y='fraction',color='split',barmode='group',title=feature))
    st.caption('Bins are defined from training alerts and applied unchanged to test. Each alert has equal weight in this chart.')
    cats = table('categories.csv')
    field = st.radio('Transaction category',['kirim_chiqim','tranzaksiya_turi'],horizontal=True)
    chart(px.bar(cats[(cats.field==field)&(cats.target_group=='All')],x='category',y='fraction',color='split',barmode='group'))
    st.write('Incoming transactions and card/bank transfers dominate the table. Rare cash and international activity is summarized with both counts and proportions, then evaluated through feature ablations.')
    amount = table('amount_histogram.csv')
    chart(px.bar(amount[amount.target_group=='All'],x='bin',y='fraction',color='split',barmode='group',title='Standardized transaction-size index'))
    st.caption('The index includes negative values and a positive tail. Summing it describes cumulative standardized activity, not money transferred.')
    timing = table('timing_histogram.csv')
    chart(px.bar(timing[timing.target_group=='All'],x='bin',y='fraction',color='split',barmode='group',title='Transaction age before alert'))
    st.caption('Unequal time-bin widths: bars show fractions of transactions, not an hourly density. Only transactions at or before alert midnight are included.')
    volume = table('monthly_volume.csv')
    chart(px.line(volume,x='month',y='per_signal',color='split',title='Transaction volume per split alert count'))
    st.caption('The rise and fall at coverage boundaries can reflect the finite observation period; this chart does not establish economic seasonality.')

elif page == 'Outcome differences':
    st.subheader('Which behaviors accompany escalation?')
    bins = table('binned_target.csv')
    options = sorted(bins.feature.unique())
    feature = st.selectbox('Behavior to inspect',options,index=options.index('w7_count'))
    data = bins[bins.feature==feature].copy()
    data['95% half-width'] = 1.96*data.se
    chart(px.line(data,x='mean_value',y='target_rate',error_y='95% half-width',markers=True,hover_data=['n','bin'],title=f'Escalation rate vs {feature}'))
    st.caption('Each point summarizes a training-defined quantile bin. Associations are descriptive, not causal; the curves were not used as target encodings.')
    summary = table('feature_summary.csv')
    chosen = summary[summary.feature==feature]
    st.dataframe(chosen,hide_index=True,width='stretch')
    if len(chosen)==2:
        a = chosen.set_index('group')['median']
        st.write(f"For {feature}, the median is {a['Escalated']:.3g} in escalated alerts and {a['Dismissed']:.3g} in dismissed alerts. This motivates measuring that behavior; cross-validation determines whether it adds predictive value.")
    cats = table('categories.csv')
    chart(px.bar(cats[(cats.split=='train')&(cats.target_group!='All')&(cats.field=='tranzaksiya_turi')],x='category',y='fraction',color='target_group',barmode='group',title='Transaction types by outcome'))
    st.caption('Composition charts weight transactions equally; alerts with longer histories contribute more transactions. The feature summaries above weight alerts equally.')

elif page == 'Validation & models':
    st.subheader('Validation before model selection')
    st.write('Primary: five shuffled stratified folds, fixed seed 20260927. The frozen final recipe uses a separate 20% inner holdout to choose boosting rounds, then refits on each full outer training fold. Exploratory models used outer-fold early stopping. Secondary: three expanding temporal splits; each validation block is strictly later than its training block. Model-specific preprocessing is fitted inside each training fold. Every serious experiment stores out-of-fold probabilities.')
    if (DATA/'experiment_results.csv').exists():
        results = table('experiment_results.csv')
        shortlist=['lgb_baseline','lgb_types','lgb_contrasts','cat_compact','xgb_compact','logistic_compact','extra_trees','hist_gradient']
        show_all=st.checkbox('Show all exploratory variants',value=False)
        plotted=results if show_all else results[results.experiment.isin(shortlist)]
        chart(px.bar(plotted,x='experiment',y='auc_mean',error_y='auc_std',color='validation',barmode='group',title='Mean fold ROC-AUC (error bars: fold standard deviation)'))
        st.dataframe(results[['experiment','features','model','validation','auc_mean','auc_std','auc_oof','fold_auc']],hide_index=True,width='stretch')
        st.download_button('Download experiment table',results.to_csv(index=False),'experiment_results.csv','text/csv')
    if (DATA/'final_summary.json').exists():
        summary = document('final_summary.json')
        st.success(f"Selected model: {summary['description']}. Mean five-fold AUC: {summary['auc_mean']:.6f}; pooled OOF AUC: {summary['auc_oof']:.6f}.")
        st.write(summary.get('selection_note',''))
    if (DATA/'importance.csv').exists():
        importance = table('importance.csv').head(20).sort_values('importance')
        chart(px.bar(importance,x='importance',y='feature',orientation='h',title='Leading features in the selected model'))
    if (DATA/'permutation_importance.csv').exists():
        perm=table('permutation_importance.csv')
        perm=perm[~perm.feature_or_group.str.startswith('GROUP:')].head(12).sort_values('mean_auc_drop')
        chart(px.bar(perm,x='mean_auc_drop',y='feature_or_group',error_x='std_auc_drop',orientation='h',title='Held-out permutation importance'))
        st.caption('First outer validation fold, three shuffles per feature. Bars show the AUC loss when a feature is permuted. Correlated features can mask one another; these diagnostics did not change the frozen recipe.')
    drift = table('feature_drift.csv')
    with st.expander('Train/test feature drift diagnostics'):
        st.dataframe(drift.head(30),hide_index=True,width='stretch')
        st.caption('KS effect sizes are descriptive. Hundreds of comparisons mean isolated small p-values are expected; features are not removed automatically based on p-values.')
    st.warning('These are local validation results, not a leaderboard score. Reusing folds for model and blend selection can introduce optimism. Fold standard deviation is not a confidence interval for hidden-test AUC.')

else:
    st.subheader('Findings and modeling implications')
    st.write(f"The training set has {train['unique_signals']:,} alerts and an escalation rate of {overview['target_rate']:.2%}. Complete relational coverage allows every alert to receive historical behavior features.")
    st.write('The roughly 180-day histories support recent-window and older-baseline comparisons. Activity counts, category composition, amount distribution, and recency are evaluated jointly because any one summary can miss nonlinear interactions.')
    if (DATA/'findings.json').exists():
        for finding in document('findings.json')['findings']:
            st.markdown(f'- {finding}')
    st.write('The shared train/test date range supports stratified validation, while temporal holdouts test sensitivity to time. The final submission contains one continuous probability per test alert.')
    st.info('This website contains aggregate EDA and local validation results only. No hidden test labels are available. Public hosting is a separate deployment step.')

st.divider()
st.caption('Reproducible analysis · fixed folds · historical cutoff enforced · ROC-AUC selection')
