# Height objective quality-gate freeze review

Human primary references: 30 total; 19 boundary-available and 11 unmeasurable after exactly four adjudication overrides (measurable=11, uncertain=8). Human confidence is not a feature.

## Objective feature dictionary

| Feature group | Units | Nullability | Code path | Interpretation |
|---|---|---|---|---|
| valid fractions/counts | fraction or count | nullable only when boundary unavailable | build_features | local/aggregation/central and independent side support |
| boundary NaN distance/run/gap | mm or count | yes | near_stats/nearest_nan | support and missing-data proximity |
| gradient magnitude/prominence | mm/mm and ratio | yes | predict/build_features | edge strength relative to robust side MAD |
| profile-baseline contrast | mm | yes | build_features | boundary height departure from independent substrate baseline |
| aggregation sensitivity | mm | yes | prediction_at/build_features | spread over 0.025/0.05/0.10 mm aggregation |
| method disagreement | mm | yes | prediction_at/build_features | pairwise and maximum disagreement with Controls B–D |
| neighbor continuity/MAD | mm | yes at endpoints/missing neighbor | build_features second pass | official ±0.2 mm continuity without replacement |
| candidate-limit/cap diagnostics | mm/count/boolean | yes | build_features | limit proximity and repeated apparent width |

Evaluated 10 shared deterministic gates and two diagnostic model families using three complete-Track leave-one-out folds. No random row split, sample exceptions, or Track-specific thresholds were used.

## Gate review table

| Gate | Role | FA | Recall | Precision | Width MAE | Center MAE | Development accepted | Review category |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `accept_all_finite` | deterministic_candidate | 10 | 1.0 | 0.6551724137931034 | 0.09471505263157898 | 0.05499415789473685 | 1155 | balanced_precision_recall |
| `valid_ge_0p50` | deterministic_candidate | 5 | 0.7368421052631579 | 0.7368421052631579 | 0.08997528571428573 | 0.05719635714285717 | 837 | balanced_precision_recall |
| `valid_ge_0p70` | deterministic_candidate | 1 | 0.15789473684210525 | 0.75 | 0.1200866666666666 | 0.11625999999999999 | 59 | low_false_accept |
| `side_support_ge_0p50` | deterministic_candidate | 8 | 1.0 | 0.7037037037037037 | 0.09471505263157898 | 0.05499415789473685 | 1127 | balanced_precision_recall |
| `agg_width_le_0p10` | deterministic_candidate | 7 | 0.631578947368421 | 0.631578947368421 | 0.10231166666666669 | 0.05137533333333333 | 635 | high_yield_riskier |
| `method_width_le_0p15` | deterministic_candidate | 7 | 0.6842105263157895 | 0.65 | 0.10842876923076925 | 0.0495186923076923 | 762 | high_yield_riskier |
| `nan_distance_ge_0p02` | deterministic_candidate | 5 | 0.631578947368421 | 0.7058823529411765 | 0.12285916666666667 | 0.03641291666666668 | 747 | high_yield_riskier |
| `balanced_rule_v1` | deterministic_candidate | 4 | 0.5263157894736842 | 0.7142857142857143 | 0.09576060000000002 | 0.05630449999999999 | 570 | high_yield_riskier |
| `low_false_accept_rule_v1` | deterministic_candidate | 5 | 0.5789473684210527 | 0.6875 | 0.10973527272727274 | 0.054164181818181834 | 562 | high_yield_riskier |
| `high_yield_rule_v1` | deterministic_candidate | 9 | 1.0 | 0.6785714285714286 | 0.09471505263157898 | 0.05499415789473685 | 1140 | balanced_precision_recall |
| `diagnostic_depth1_tree` | diagnostic_only | 6 | 1.0 | 0.76 | 0.09471505263157898 | 0.05499415789473685 | 1103 | balanced_precision_recall |
| `diagnostic_regularized_logistic` | diagnostic_only | 5 | 0.9473684210526315 | 0.782608695652174 | 0.08981700000000001 | 0.05774383333333334 | 1036 | balanced_precision_recall |

Candidate highlighted for human review: `side_support_ge_0p50`. It has false accepts=8, recall=1.0, precision=0.7037037037037037, conditional width MAE=0.09471505263157898 mm, center MAE=0.05499415789473685 mm. Development yield is reported separately and is not a full-dataset yield estimate. Low-false-accept, balanced, and higher-yield trade-offs remain visible in the candidate table. This recommendation does not freeze a gate.

Reserved data were not loaded, inspected, plotted, or used. Boundary coordinates remain the frozen estimator outputs regardless of gate decisions.

QUALITY GATE REQUIRES REVISION
