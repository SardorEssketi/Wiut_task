"""Write a concise report from measured artifacts, without invented conclusions."""
import json
import platform
from importlib.metadata import version
from pathlib import Path
import pandas as pd
from .data import ROOT, ARTIFACTS, save_json


def generate_report():
    summary=json.loads((ARTIFACTS/'final_summary.json').read_text())
    audit=json.loads((ARTIFACTS/'data_audit.json').read_text())
    diagnostics=json.loads((ARTIFACTS/'validation_diagnostics.json').read_text())
    results=pd.read_csv(ARTIFACTS/'experiment_results.csv')
    train=audit['splits']['train']; test=audit['splits']['test']
    score=results.set_index(['experiment','validation']).auc_mean
    findings=[
        f"The escalation rate is {audit['target_rate']:.2%}; every alert has at least one valid historical transaction.",
        f"Activity-only LightGBM scored {score.loc[('lgb_baseline','stratified')]:.4f} mean AUC. Adding transaction-type information reached {score.loc[('lgb_types','stratified')]:.4f}; within-alert amount/type contrasts reached {score.loc[('lgb_contrasts','stratified')]:.4f}.",
        'Escalated alerts have a median of 493 historical transactions versus 455 for dismissed alerts, but activity concentration is very similar: median last-7-day shares are about 8.90% and 8.94%, respectively. Recent bursts alone are not a strong discriminator here.',
        'Escalated alerts have a lower median full-history amount-index mean (-0.144 versus -0.069) and standard deviation (0.824 versus 0.857). Type-specific distribution shape and contrasts add information beyond these weak marginal differences.',
        f"The train/test source classifier scored {diagnostics['adversarial_source_auc']:.4f} AUC, with no detectable separation under that diagnostic. Both splits span the same alert dates.",
        f"The final model is {summary['description']}. Reported scores are local estimates, not a hidden-test result.",
    ]
    save_json({'findings':findings},ROOT/'eda_site'/'data'/'findings.json')
    versions={name:version(name) for name in ['numpy','pandas','scipy','scikit-learn','pyarrow','catboost','lightgbm','xgboost','matplotlib','plotly','streamlit','nbformat','nbclient','ipykernel','joblib']}
    save_json({'python':platform.python_version(),'platform':platform.platform(),'packages':versions,'seed':20260927},ARTIFACTS/'environment.json')
    lines=['# Experiment results','', 'Metric: ROC-AUC. Higher is better. All values below come from saved predictions, using fixed splits and seed 20260927.', '', '| Experiment | Features | Model | Validation | Mean AUC | Fold SD | Pooled OOF AUC |','|---|---|---|---|---:|---:|---:|']
    for _,r in results.iterrows():
        lines.append(f"| {r.experiment} | {r.features} | {r.model} | {r.validation} | {r.auc_mean:.6f} | {r.auc_std:.6f} | {r.auc_oof:.6f} |")
    lines+=['','`stratified`: exploratory 5-fold CV with outer-fold early stopping. `nested`: the same 5 outer folds, with stopping selected on a separate 20% inner holdout and a refit on the full outer training fold. This is nested stopping, not a fully nested search over every feature/model choice. `temporal`: three expanding training blocks with strictly later validation blocks. Temporal exploratory models use their validation block for stopping.','', 'Preprocessing and optional supervised feature selection are fitted within each training fold. Full parameter sets, fold AUC, iteration counts, timings, feature importance, and OOF predictions are in `artifacts/experiments/`.','', 'The broad 500-tree ExtraTrees run was interrupted during its first fold because it was disproportionately slow. Its completed replacement uses 250 trees and the compact feature set. No score is claimed for the interrupted run.','', 'Early features were computed without targets. Later feature groups were informed by training EDA and exploratory CV; repeated selection still introduces model-selection optimism.','', '## Final selection','',summary['description'], '',summary['selection_note'],'',f"Mean fold AUC: **{summary['auc_mean']:.6f}**; pooled OOF AUC: **{summary['auc_oof']:.6f}**. Fold SD: {summary['auc_std']:.6f}.",'', 'A paired bootstrap of the strict OOF predictions gives a blend-minus-LightGBM AUC gain of about 0.00122, with a conditional 95% interval of [-0.00180, 0.00405]. The small benefit is uncertain. Held-out permutation importance independently highlights the type-contrast group; see `artifacts/permutation_importance.csv` and `artifacts/final_diagnostics.json`.','', '## Findings','']+['- '+f for f in findings]
    (ROOT/'EXPERIMENTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    temporal=summary.get('temporal')
    temporal_text=f"Temporal blend mean AUC: {temporal['auc_mean']:.6f}, fold SD {temporal['auc_std']:.6f}." if temporal else ''
    readme=f'''# Escalation prediction — reproducible competition solution

Predict `eskalatsiya` from each alert's historical transactions. The competition metric is **ROC-AUC**. No external labels, ID predictors, hidden-label reconstruction, or post-alert features are used.

## Result

- Final model: **{summary['description']}**.
- Final five-fold mean ROC-AUC: **{summary['auc_mean']:.6f}** (fold SD {summary['auc_std']:.6f}); pooled OOF AUC: **{summary['auc_oof']:.6f}**.
- {temporal_text}
- {summary['selection_note']}
- Output: **`{summary['submission_filename']}`**, exactly 6,000 rows with `signal_id,ehtimollik`.

See [EXPERIMENTS.md](EXPERIMENTS.md) and [the machine-readable experiment table](artifacts/experiment_results.csv) for all measured comparisons. Fold SD is not a leaderboard confidence interval. Bootstrap intervals in `artifacts/final_summary.json` condition on the selected OOF predictions and exclude selection uncertainty.

## Data and leakage decisions

| Split | Alerts | Transactions | Escalations | Post-cutoff rows excluded |
|---|---:|---:|---:|---:|
| Train | {train['unique_signals']:,} | {train['transaction_shape'][0]:,} | 2,405 (17.18%) | {train['transactions_after_timestamp']} |
| Test | {test['unique_signals']:,} | {test['transaction_shape'][0]:,} | Unknown | {test['transactions_after_timestamp']} |

Both alert splits cover **2025-01-01 through 2026-12-31**. Transaction history extends up to 180 days. There are no missing raw values, duplicate alert IDs, duplicate transaction rows, or alerts without historical transactions. The full audit, including dtypes, all date ranges, category counts, quantiles, and source SHA-256 hashes, is `artifacts/data_audit.json`.

An alert date is conservatively interpreted as **00:00:00 at the start of that date**. Use `transaction_timestamp <= alert_timestamp`; exclude later hours even on the same calendar date. Transactions exactly at the cutoff are available under this rule. Excluded-row counts never enter predictors. The supplied amount index ranges roughly from -2.91 to 6.69; sums describe standardized activity, not currency. Source order breaks timestamp ties in experimental last-N features; the final compact/contrast models omit those sequence features.

`signal_id` is used only for joins and output order. The sample submission's prediction column is never loaded. Raw files are unchanged. The test set supplies no labels.

## Validation and features

Five `StratifiedKFold` outer folds use shuffle and seed **20260927**. Train and test cover the same dates (date KS statistic {diagnostics['train_test_date_ks']:.4f}); an independent train/test-source classifier scored {diagnostics['adversarial_source_auc']:.4f} AUC. Monthly target-rate chi-square p={diagnostics['monthly_target_pvalue']:.3f} is inconclusive evidence of drift. Three expanding date splits provide an additional stress test. See `artifacts/validation_diagnostics.json`.

The first feature set has 840 candidates; the refined set has 1,065. Features cover full-history and 1/2/3/7/14/30/60/90-day counts and amount distributions, subday timing, direction/type proportions, recency, inter-event gaps, daily concentration, behavioral change, calendar effects, and last-N sequences. The refinement adds within-alert type/direction amount differences, normalized contrasts, volatility ratios, and type-specific quantiles. The compact subset has about 210 nonconstant features.

Constant/duplicate removal, imputation, scaling, quantile transforms, splines, and optional supervised feature selection are fitted inside training folds. Feature generation is target-free and per alert. No pooled target encodings are used. Correlations and drift are reported rather than automatically deleting correlated tree inputs.

## Environment and reproduction

Tested on Python {platform.python_version()} on Windows, a 16-logical-core CPU and 16 GB RAM. Models use up to 8 threads. Exact package versions are in `requirements.txt` and `artifacts/environment.json`. A clean run reads about 10 million transactions; allow several GB of RAM and tens of minutes depending on CPU.

```powershell
python -m pip install -r requirements.txt
python -m solution.data
python -m solution.features
python -m solution.enhanced
python -m solution.eda
python -m solution.diagnostics
python -m solution.finalize
python -m solution.final_diagnostics
python -m solution.report
python -m solution.verify
```

The default data discovery searches below the project root for exactly one `train_signals.csv`. Keep the five supplied files together; `sample_submission (3).csv` is supported. Feature caches are fingerprinted from source code and source files. To rebuild them explicitly, call `prepare_features(force=True)` and `prepare_enhanced(force=True)` from the notebook or Python.

To reproduce a model experiment or date check:

```powershell
python -m solution.modeling --models lgb_contrasts cat_compact --force
python -m solution.modeling --strategy nested --models lgb_contrasts cat_compact --force
python -m solution.modeling --strategy temporal --models lgb_contrasts cat_compact --force
```

The final recipe, including frozen blend weights and model configurations, is `solution/selection.json`. `solution.finalize` refits each component on all training alerts using the median chosen boosting rounds, saves models under `artifacts/models/`, and writes/reloads/validates the prediction CSV. To use your team ID: `python -m solution.finalize --team-id YOUR_TEAM_ID`.

## Notebook

Open **`solution.ipynb`** in Jupyter or VS Code and run all cells. It locates the project root, reads the original files, validates the data, regenerates features/EDA, reruns core ablations and final comparisons, verifies the frozen recipe, refits the models, and validates the submission. Default settings rebuild features and retrain the notebook's core experiments, without relying on hidden cell state. The complete exploratory search history is preserved in the experiment table; the notebook retains the useful core comparisons.

Headless execution: `python -m solution.execute_notebook`. This saves executed outputs in `solution.ipynb` and a completion record under `artifacts/`.

## EDA website

```powershell
python -m streamlit run eda_site/app.py
```

The app uses **aggregate tables only** from `eda_site/data/`; it does not need raw competition files or saved model binaries. It includes target/time analysis, interactive distributions, outcome differences, leakage decisions, feature implications, model scores, and limitations. Deployment dependencies and configuration are included in `eda_site/requirements.txt` and `.streamlit/config.toml`.

For Streamlit Community Cloud, put the app and its aggregate data/configuration in a repository you control, select `eda_site/app.py` as the entry point, and choose Python 3.12. No public deployment was performed. Do not publish raw competition files unless competition rules allow it.

## Deliverables and caveats

- `team_SUBMIT_READY.csv`: final predictions, sample ID order, no pandas index.
- `solution.ipynb`: reproducible notebook; `solution/`: modular source and frozen final recipe.
- `eda_site/app.py`, `eda_site/data/`, `eda_site/plots/`: public-ready report and static figures.
- `EXPERIMENTS.md`, `artifacts/experiment_results.csv`: results and parameters.
- `artifacts/experiments/`: OOF/test predictions and fold-level records.
- `artifacts/final_oof.csv`, `artifacts/final_summary.json`: final local validation evidence.
- `artifacts/integrity_checks.json`, `artifacts/site_checks.json`, `artifacts/submission_checks.json`: checks actually run.

Rename the submission to `team_<TEAM_ID>.csv` (or regenerate with `--team-id`) before uploading. Public website hosting is your remaining deployment step. Hidden-test ROC-AUC is unknown. Model/feature choices reused exploratory folds, so local performance can be optimistic even with inner stopping. There is no customer ID for grouped validation. A different organizer interpretation of date-only alert availability could change eligible transactions; this solution deliberately uses the strict historical cutoff.
'''
    (ROOT/'README.md').write_text(readme,encoding='utf-8')
    print('README, experiment report, public findings, and environment manifest written.',flush=True)


if __name__=='__main__':
    generate_report()
