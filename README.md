# Escalation prediction — reproducible competition solution

Predict `eskalatsiya` from each alert's historical transactions. The competition metric is **ROC-AUC**. No external labels, ID predictors, hidden-label reconstruction, or post-alert features are used.

## Result

- Final model: **50% LightGBM with type contrasts + 50% compact CatBoost**.
- Final five-fold mean ROC-AUC: **0.651330** (fold SD 0.014790); pooled OOF AUC: **0.650206**.
- Temporal blend mean AUC: 0.662650, fold SD 0.028201.
- A coarse OOF weight comparison selected a 50/50 blend on exploratory folds (mean AUC 0.654975, pooled AUC 0.654104). The weights were then fixed for the inner-stopping audit (mean AUC 0.651330, pooled AUC 0.650206). The blend gain is small and is not claimed to be statistically decisive. Model and feature search itself was exploratory, not fully nested. Final components are refitted on all training alerts using median inner-selected boosting rounds.
- Output: **`team_SUBMIT_READY.csv`**, exactly 6,000 rows with `signal_id,ehtimollik`.

See [EXPERIMENTS.md](EXPERIMENTS.md) and [the machine-readable experiment table](artifacts/experiment_results.csv) for all measured comparisons. Fold SD is not a leaderboard confidence interval. Bootstrap intervals in `artifacts/final_summary.json` condition on the selected OOF predictions and exclude selection uncertainty.

## Data and leakage decisions

| Split | Alerts | Transactions | Escalations | Post-cutoff rows excluded |
|---|---:|---:|---:|---:|
| Train | 14,000 | 6,987,663 | 2,405 (17.18%) | 840 |
| Test | 6,000 | 3,027,575 | Unknown | 51 |

Both alert splits cover **2025-01-01 through 2026-12-31**. Transaction history extends up to 180 days. There are no missing raw values, duplicate alert IDs, duplicate transaction rows, or alerts without historical transactions. The full audit, including dtypes, all date ranges, category counts, quantiles, and source SHA-256 hashes, is `artifacts/data_audit.json`.

An alert date is conservatively interpreted as **00:00:00 at the start of that date**. Use `transaction_timestamp <= alert_timestamp`; exclude later hours even on the same calendar date. Transactions exactly at the cutoff are available under this rule. Excluded-row counts never enter predictors. The supplied amount index ranges roughly from -2.91 to 6.69; sums describe standardized activity, not currency. Source order breaks timestamp ties in experimental last-N features; the final compact/contrast models omit those sequence features.

`signal_id` is used only for joins and output order. The sample submission's prediction column is never loaded. Raw files are unchanged. The test set supplies no labels.

## Validation and features

Five `StratifiedKFold` outer folds use shuffle and seed **20260927**. Train and test cover the same dates (date KS statistic 0.0060); an independent train/test-source classifier scored 0.4978 AUC. Monthly target-rate chi-square p=0.094 is inconclusive evidence of drift. Three expanding date splits provide an additional stress test. See `artifacts/validation_diagnostics.json`.

The first feature set has 840 candidates; the refined set has 1,065. Features cover full-history and 1/2/3/7/14/30/60/90-day counts and amount distributions, subday timing, direction/type proportions, recency, inter-event gaps, daily concentration, behavioral change, calendar effects, and last-N sequences. The refinement adds within-alert type/direction amount differences, normalized contrasts, volatility ratios, and type-specific quantiles. The compact subset has about 210 nonconstant features.

Constant/duplicate removal, imputation, scaling, quantile transforms, splines, and optional supervised feature selection are fitted inside training folds. Feature generation is target-free and per alert. No pooled target encodings are used. Correlations and drift are reported rather than automatically deleting correlated tree inputs.

## Environment and reproduction

Tested on Python 3.12.6 on Windows, a 16-logical-core CPU and 16 GB RAM. Models use up to 8 threads. Exact package versions are in `requirements.txt` and `artifacts/environment.json`. A clean run reads about 10 million transactions; allow several GB of RAM and tens of minutes depending on CPU.

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
