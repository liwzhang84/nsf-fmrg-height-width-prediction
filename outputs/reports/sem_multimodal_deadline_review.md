# SEM multimodal deadline review

- Primary-label eligible rows: 1155.
- Leakage-safe SEM-input eligible rows used by SEM comparisons: 678.
- Validation: three complete-Track outer folds; complete-Track inner selection only.
- Thermal input: fixed `thermal_shape_features`; no Thermal reselection.
- SEM center: masked ±0.40 mm in native coordinates before resizing/features.

## Aggregate metrics

| model_id | sample_count | mae | rmse | unweighted_track_mae | improves_all_three_tracks |
|---|---|---|---|---|---|
| fixed_thermal_shape_ridge | 678 | 0.10026941301618346 | 0.1289996489769902 | 0.10023207466388084 | False |
| sem_pca_ridge | 678 | 0.11102178352404175 | 0.13823823475650718 | 0.11076637091977737 | False |
| sem_pca_upstream_ridge | 678 | 0.10717508197263914 | 0.1342881752279683 | 0.10700217535112609 | False |
| sem_summary_ridge | 678 | 0.09852262201698156 | 0.12608760224423288 | 0.09851738113308188 | False |
| sem_summary_upstream_ridge | 678 | 0.10172682218469176 | 0.12953311746805987 | 0.10168205113934568 | False |
| thermal_plus_sem_pca_ridge | 678 | 0.11276299577228115 | 0.14245868222434904 | 0.11253837788117388 | False |
| thermal_plus_sem_summary_ridge | 678 | 0.10135655755106127 | 0.13140036618702766 | 0.10140044855657859 | False |
| training_mean | 678 | 0.10258907456911119 | 0.13168290814266856 | 0.10248412642544842 | False |
| x_only | 678 | 0.10257211910119797 | 0.13139691150789334 | 0.102473177555092 | False |

SEM AND MULTIMODAL SIGNAL NOT ESTABLISHED ACROSS DEVELOPMENT TRACKS
