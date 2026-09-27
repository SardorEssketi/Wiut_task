# Experiment results

Metric: ROC-AUC. Higher is better. All values below come from saved predictions, using fixed splits and seed 20260927.

| Experiment | Features | Model | Validation | Mean AUC | Fold SD | Pooled OOF AUC |
|---|---|---|---|---:|---:|---:|
| lgb_contrasts | contrasts | lgb | nested | 0.650748 | 0.015468 | 0.648990 |
| cat_compact | compact | cat | nested | 0.647492 | 0.013283 | 0.646003 |
| xgb_compact | compact | xgb | nested | 0.645541 | 0.017535 | 0.645892 |
| lgb_contrasts | contrasts | lgb | stratified | 0.654022 | 0.014618 | 0.652407 |
| lgb_contrasts7 | contrasts | lgb | stratified | 0.653462 | 0.012605 | 0.652125 |
| lgb_select60 | contrasts | lgb | stratified | 0.652770 | 0.014049 | 0.649082 |
| lgb_enhanced | enhanced | lgb | stratified | 0.652723 | 0.014800 | 0.650721 |
| cat_compact | compact | cat | stratified | 0.651829 | 0.013179 | 0.651110 |
| cat_contrasts | contrasts | cat | stratified | 0.651488 | 0.016472 | 0.650714 |
| xgb_compact | compact | xgb | stratified | 0.651324 | 0.014890 | 0.648970 |
| lgb_compact | compact | lgb | stratified | 0.651165 | 0.013674 | 0.648012 |
| lgb_contrasts31 | contrasts | lgb | stratified | 0.651117 | 0.015372 | 0.648791 |
| cat_compact3 | compact | cat | stratified | 0.650204 | 0.013346 | 0.650094 |
| cat_compact6 | compact | cat | stratified | 0.649760 | 0.015584 | 0.649654 |
| lgb_types | types | lgb | stratified | 0.636088 | 0.012159 | 0.633200 |
| logistic_compact | compact | logistic | stratified | 0.635252 | 0.009090 | 0.635176 |
| lgb_full | full | lgb | stratified | 0.634516 | 0.013550 | 0.634322 |
| extra_trees | compact | extra | stratified | 0.634378 | 0.011203 | 0.634116 |
| xgb_depth3 | sequence | xgb | stratified | 0.634216 | 0.015078 | 0.633349 |
| lgb_sequence | sequence | lgb | stratified | 0.633402 | 0.012285 | 0.631048 |
| lgb_change | change | lgb | stratified | 0.631859 | 0.013680 | 0.630152 |
| cat_depth5 | sequence | cat | stratified | 0.626575 | 0.011772 | 0.626873 |
| cat_native | full | cat | stratified | 0.626440 | 0.009280 | 0.625915 |
| hist_gradient | sequence | hist | stratified | 0.624201 | 0.009067 | 0.624590 |
| spline_compact_reg | compact | spline | stratified | 0.616678 | 0.004766 | 0.616695 |
| lgb_baseline | baseline | lgb | stratified | 0.605589 | 0.015326 | 0.598761 |
| logistic | sequence | logistic | stratified | 0.605417 | 0.006734 | 0.605512 |
| lgb_direction | direction | lgb | stratified | 0.602946 | 0.016237 | 0.592653 |
| lgb_recent | recent | lgb | stratified | 0.600206 | 0.015187 | 0.596450 |
| spline_compact | compact | spline | stratified | 0.599970 | 0.006034 | 0.599967 |
| lgb_contrasts | contrasts | lgb | temporal | 0.659913 | 0.032066 | 0.657065 |
| cat_compact | compact | cat | temporal | 0.659786 | 0.022597 | 0.660009 |
| xgb_compact | compact | xgb | temporal | 0.657308 | 0.024146 | 0.653943 |

`stratified`: exploratory 5-fold CV with outer-fold early stopping. `nested`: the same 5 outer folds, with stopping selected on a separate 20% inner holdout and a refit on the full outer training fold. This is nested stopping, not a fully nested search over every feature/model choice. `temporal`: three expanding training blocks with strictly later validation blocks. Temporal exploratory models use their validation block for stopping.

Preprocessing and optional supervised feature selection are fitted within each training fold. Full parameter sets, fold AUC, iteration counts, timings, feature importance, and OOF predictions are in `artifacts/experiments/`.

The broad 500-tree ExtraTrees run was interrupted during its first fold because it was disproportionately slow. Its completed replacement uses 250 trees and the compact feature set. No score is claimed for the interrupted run.

Early features were computed without targets. Later feature groups were informed by training EDA and exploratory CV; repeated selection still introduces model-selection optimism.

## Final selection

50% LightGBM with type contrasts + 50% compact CatBoost

A coarse OOF weight comparison selected a 50/50 blend on exploratory folds (mean AUC 0.654975, pooled AUC 0.654104). The weights were then fixed for the inner-stopping audit (mean AUC 0.651330, pooled AUC 0.650206). The blend gain is small and is not claimed to be statistically decisive. Model and feature search itself was exploratory, not fully nested. Final components are refitted on all training alerts using median inner-selected boosting rounds.

Mean fold AUC: **0.651330**; pooled OOF AUC: **0.650206**. Fold SD: 0.014790.

A paired bootstrap of the strict OOF predictions gives a blend-minus-LightGBM AUC gain of about 0.00122, with a conditional 95% interval of [-0.00180, 0.00405]. The small benefit is uncertain. Held-out permutation importance independently highlights the type-contrast group; see `artifacts/permutation_importance.csv` and `artifacts/final_diagnostics.json`.

## Findings

- The escalation rate is 17.18%; every alert has at least one valid historical transaction.
- Activity-only LightGBM scored 0.6056 mean AUC. Adding transaction-type information reached 0.6361; within-alert amount/type contrasts reached 0.6540.
- Escalated alerts have a median of 493 historical transactions versus 455 for dismissed alerts, but activity concentration is very similar: median last-7-day shares are about 8.90% and 8.94%, respectively. Recent bursts alone are not a strong discriminator here.
- Escalated alerts have a lower median full-history amount-index mean (-0.144 versus -0.069) and standard deviation (0.824 versus 0.857). Type-specific distribution shape and contrasts add information beyond these weak marginal differences.
- The train/test source classifier scored 0.4978 AUC, with no detectable separation under that diagnostic. Both splits span the same alert dates.
- The final model is 50% LightGBM with type contrasts + 50% compact CatBoost. Reported scores are local estimates, not a hidden-test result.
