# SEM multimodal deadline leakage audit

- Only development Tracks 8, 10, and 14 were loaded.
- Mapping engineering confirmation was validated before labels or images entered training.
- Mapping used README/TIFF stage evidence and full ordered tile sequences, never Height labels.
- Central target pixels were excluded in native coordinates before resize or feature extraction.
- No Height-derived feature, human boundary, or Track ID entered model matrices.
- No random position split was used; all outer and inner splits held out complete Tracks.
- Imputation, scaling, PCA, and ridge fitting were training-fold-only.
- Thermal used only the previously fixed descriptive `thermal_shape_features`.
- No final-test predictions were created.
