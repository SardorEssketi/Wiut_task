"""Build the readable, executable competition notebook from modular source."""
import nbformat as nbf
from .data import ROOT


def create_notebook():
    nb=nbf.v4.new_notebook()
    cells=[]
    def md(text): cells.append(nbf.v4.new_markdown_cell(text))
    def code(text): cells.append(nbf.v4.new_code_cell(text))
    md('''# Historical alert escalation — legitimate, reproducible modeling

Predict a continuous escalation probability for each test alert. **Metric: ROC-AUC.**

The final recipe is an equal-weight LightGBM/CatBoost blend. This notebook rebuilds features from the original competition files, runs the useful comparative experiments, audits a frozen recipe with inner early stopping, refits on all training alerts, and validates the final CSV. The full exploratory history is in `EXPERIMENTS.md`.

**Leakage rule:** interpret the date-only alert as midnight; retain transactions at or before that timestamp. Exclude later transactions and their counts. Never use IDs, sample probabilities, or unavailable labels as predictors.''')
    md('## 1. Imports and configuration')
    code('''import os
os.environ['OMP_NUM_THREADS'] = '8'
os.environ['OPENBLAS_NUM_THREADS'] = '8'
from pathlib import Path
import sys, json, gc, contextlib, io
import numpy as np
import pandas as pd
from IPython.display import display, Image, Markdown

ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / 'solution' / 'data.py').exists())
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
SEED = 20260927
np.random.seed(SEED)
REBUILD_FEATURES = True
RETRAIN_CORE_EXPERIMENTS = True
TEAM_ID = 'SUBMIT_READY'  # Change only the filename identifier when your team ID is known.

from solution.data import load_data, inspect_data, ARTIFACTS
from solution.features import prepare_features
from solution.enhanced import prepare_enhanced
from solution.modeling import CONFIGS, run_experiment, update_results, make_splits
from solution.eda import generate_eda
from solution.diagnostics import validation_diagnostics
from solution.finalize import assess_blends, finalize, validate_submission
from solution.report import generate_report
print('Project:', ROOT)
print('Python:', sys.version)''')
    md('## 2. Data loading\nThe sample CSV is used only for IDs and ordering. Its probability values are never loaded.')
    code('''train, test, train_transactions, test_transactions, sample = load_data()
display(pd.DataFrame({'table': ['train signals', 'test signals', 'train transactions', 'test transactions'],
                      'rows': [len(train), len(test), len(train_transactions), len(test_transactions)],
                      'columns': [train.shape[1], test.shape[1], train_transactions.shape[1], test_transactions.shape[1]]}))
display(train.head())
display(train_transactions.head())''')
    md('## 3. Data validation and historical cutoff\nThe complete audit records shapes, dtypes, missing values, duplicates, unique IDs, category frequencies, date ranges, amount tails, transactions per signal, and cutoff eligibility. The raw files are fingerprinted for later integrity checks.')
    code('''del train_transactions, test_transactions
gc.collect()
with contextlib.redirect_stdout(io.StringIO()):
    audit = inspect_data()
display(pd.DataFrame({split: {
    'alerts': d['unique_signals'], 'transactions': d['transaction_shape'][0],
    'duplicate alerts': d['duplicate_signals'], 'duplicate transactions': d['duplicate_transaction_rows'],
    'missing transaction values': sum(d['transaction_missing'].values()),
    'no historical transactions': d['signals_without_historical_transactions'],
    'before alert': d['transactions_before_timestamp'], 'at cutoff': d['transactions_at_timestamp'],
    'after cutoff (excluded)': d['transactions_after_timestamp'],
    'median transactions': d['transactions_per_signal']['50%']
} for split, d in audit['splits'].items()}))
print('Training target rate:', audit['target_rate'])
print('Policy:', audit['leakage_policy'])
display(pd.DataFrame(audit['monthly_target']))''')
    md('''**Decision:** exclude 840 training and 51 test transactions after alert midnight. All alerts retain history. Dates span 2025–2026 in both splits; the synthetic calendar is not interpreted relative to the computer's current date. Exactly-at-cutoff transactions remain eligible. Standardized amount indices can be negative and are never described as currency.''')
    md('## 4. Feature engineering\nFeature engineering is target-free and local to each alert. The same functions process both splits. Recent and older comparison windows are disjoint; all divisions are stabilized. Missing category histories remain missing for amount summaries and zero for counts.')
    code('''base_train, base_test, base_groups = prepare_features(force=REBUILD_FEATURES)
X, X_test, groups = prepare_enhanced(force=REBUILD_FEATURES)
assert X.index.tolist() == train.signal_id.tolist()
assert X_test.index.tolist() == test.signal_id.tolist()
assert X.columns.tolist() == X_test.columns.tolist()
assert 'signal_id' not in X and 'eskalatsiya' not in X
print('Broad features:', base_train.shape, 'Refined features:', X.shape)
display(pd.Series(groups).value_counts().rename('feature count'))
del base_train, base_test
gc.collect()''')
    md('''The 840 initial candidates cover activity, amount distributions, 1/2/3/7/14/30/60/90-day windows, direction/type interactions, changes from older history, burstiness, calendar, and sequence information. The 1,065-feature refinement adds within-alert type contrasts, direction differences within each type, volatility ratios, quantiles, and short timestamp windows. Ablations decide what survives; generating a feature does not imply that it helps.''')
    md('## 5. EDA\nAll public-site outputs are aggregate tables. The plots below distinguish alert-weighted summaries from transaction-weighted compositions.')
    code('''generate_eda()
for name in ['target', 'activity', 'volume', 'direction', 'types', 'amount', 'recency', 'target_time', 'recent_target', 'burst_target', 'volatility_target', 'train_test']:
    display(Image(filename=str(ROOT / 'eda_site' / 'plots' / f'{name}.png'), width=850))''')
    md('''### Interpretations supported by the data

- **Target:** 17.18% of alerts are escalated. Use stratified folds and ROC-AUC rather than accuracy.
- **History volume:** escalated alerts have median 493 transactions versus 455 for dismissed alerts; counts alone remain weak predictors.
- **Calendar volume:** finite history/alert coverage changes monthly totals; this is not evidence of economic seasonality.
- **Composition:** incoming transactions and card/bank transfers dominate. Rare cash/international activity needs both counts and proportions. These plots weight transactions; alert-level features avoid letting long histories dominate the model.
- **Amount behavior:** escalated alerts have slightly lower full-history amount means and volatility. Type-specific contrasts are more useful than a single global amount statistic.
- **Recency:** many transactions cluster shortly before alerts, but median seven-day shares are nearly identical across outcomes (about 8.9%). Do not claim a large burst-related effect based on this dataset.
- **Binned target rates:** intervals are descriptive, and overlapping error bars caution against interpreting small differences. No target-rate bins are used as features.
- **Train/test:** date and activity distributions are similar. This supports, but does not prove, comparable hidden-test behavior.''')
    md('## 6. Validation strategy\nPrimary outer splits: five shuffled stratified folds, seed 20260927. Expanding temporal holdouts train only on dates strictly before validation. There is no customer ID for grouped CV. No exact duplicate complete feature rows were detected.')
    code('''diagnostics = validation_diagnostics()
display(pd.Series(diagnostics))
s = train.set_index('signal_id').loc[X.index]
fold_rows = []
for strategy in ['stratified', 'temporal']:
    for fold, (tr, va) in enumerate(make_splits(s, strategy), start=1):
        fold_rows.append({'strategy': strategy, 'fold': fold, 'training alerts': len(tr), 'validation alerts': len(va),
                          'last training date': str(s.signal_sanasi.iloc[tr].max().date()),
                          'first validation date': str(s.signal_sanasi.iloc[va].min().date())})
display(pd.DataFrame(fold_rows))''')
    md('''Exploratory boosted models use outer validation for early stopping. For the frozen final recipe, a separate 20% stratified **inner** split chooses boosting rounds; each outer training fold is then refitted before predicting its untouched outer fold. This controls stopping optimism but is **not a fully nested re-run of all model/feature selection**. Every model stores OOF probabilities.''')
    md('## 7. Baseline and feature ablations\nThe same LightGBM configuration measures the incremental feature groups. Constant/duplicate removal is fold-local. Poorer ablations are retained in the experiment record.')
    code('''loaded = (X, X_test, groups, s)
ablation_names = ['lgb_baseline', 'lgb_recent', 'lgb_direction', 'lgb_types', 'lgb_change', 'lgb_sequence', 'lgb_full']
for name in ablation_names:
    run_experiment(name, force=RETRAIN_CORE_EXPERIMENTS, loaded=loaded)
display(update_results().query("experiment in @ablation_names and validation == 'stratified'")
        [['experiment', 'features', 'auc_mean', 'auc_std', 'auc_oof']])''')
    md('## 8. Stronger model comparisons\nThe core comparison contrasts refined LightGBM, compact CatBoost, shallow XGBoost, and regularized logistic regression. The full report also records histogram boosting, ExtraTrees, spline models, native CatBoost categories, depth/leaves/regularization tuning, and fold-local feature selection.')
    code('''core_names = ['lgb_contrasts', 'cat_compact', 'xgb_compact', 'logistic_compact']
for name in core_names:
    run_experiment(name, force=RETRAIN_CORE_EXPERIMENTS, loaded=loaded)
display(update_results().query("experiment in @core_names and validation == 'stratified'")
        [['experiment', 'model', 'auc_mean', 'auc_std', 'auc_oof', 'fold_auc']])''')
    md('## 9. Ensemble assessment and stricter validation\nBlend weights are coarse (25/50/75%), not finely optimized. The frozen 50/50 LightGBM/CatBoost recipe lies on a broad performance plateau. Its gain is small; do not claim statistical certainty.')
    code('''blend_table = assess_blends(core_names)
display(blend_table[['blend', 'auc_mean', 'auc_std', 'auc_oof']])
selection = json.loads((ROOT / 'solution' / 'selection.json').read_text())
for strategy in ['nested', 'temporal']:
    for name, config in selection['configs'].items():
        run_experiment(name, config=config, strategy=strategy, force=RETRAIN_CORE_EXPERIMENTS, loaded=loaded)
display(update_results().query("validation != 'stratified'")[['experiment', 'validation', 'auc_mean', 'auc_std', 'auc_oof']])''')
    md('## 10. Final all-data fitting and feature importance\nUse the frozen blend weights and median inner-selected iteration counts. The final models are refitted on all training alerts. Importance is descriptive; correlated predictors can share or substitute importance.')
    code('''submission, final_summary = finalize(team_id=TEAM_ID)
display(pd.Series({k: final_summary[k] for k in ['description', 'validation_strategy', 'auc_mean', 'auc_std', 'auc_oof', 'oof_auc_bootstrap_interval_95']}))
importance = pd.read_csv(ARTIFACTS / 'final_importance.csv')
display(importance.head(25))
ax = importance.head(20).iloc[::-1].plot.barh(x='feature', y='importance', legend=False, figsize=(10, 7), title='Final model feature importance')
display(ax.figure)''')
    md('## 11. Test prediction and submission validation\nPredictions are continuous probabilities. Validate the actual serialized CSV, exact column order, sample ID order, complete coverage, unique IDs, finite values, range, and nonzero variance.')
    code('''output_path = ROOT / f'team_{TEAM_ID}.csv'
saved = pd.read_csv(output_path)
checks = validate_submission(saved, test, sample)
display(pd.Series(checks))
display(saved.head())
print('Final submission:', output_path)''')
    md('## 12. Integrity checks and report export')
    code('''from solution.verify import integrity_checks
integrity_checks()
generate_report()
print('EDA site: python -m streamlit run eda_site/app.py')
print('Raw files unchanged; future-transaction invariance checked.')''')
    md('''## 13. Conclusions

The useful signal is concentrated in type-specific amount distributions and within-alert contrasts. Adding every temporal/sequence feature does not automatically help. A compact CatBoost model complements refined LightGBM slightly.

Read `final_summary` for the **measured** strict five-fold score, exploratory score, and temporal check. The expected strict mean AUC is about **0.6513**; exploratory mean AUC is about **0.6550**. These are local estimates, not a leaderboard result. Fold reuse for model selection, missing customer identifiers, and uncertainty about date-only alert availability remain limitations.

Rename `team_SUBMIT_READY.csv` with your team ID before uploading. The Streamlit site is ready to run locally; public hosting requires your own deployment step.''')
    nb.cells=cells
    nb.metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.12.6'}}
    nbf.write(nb,ROOT/'solution.ipynb')
    print(f'Created notebook with {len(cells)} cells.',flush=True)


if __name__=='__main__':
    create_notebook()
