# Limitations

- Height estimator uncertainty creates local target noise.
- The all-finite policy treats every finite automatic estimate as the frozen target.
- Evidence comes from a limited number of complete Tracks.
- SEM uses a low-confidence full-mosaic endpoint-linear mapping fallback.
- Adjacent SEM tiles were not registered with high-precision overlap alignment.
- Requiring complete context within one native tile reduces SEM coverage.
- Neighboring deposited-track information remains visible outside the central mask.
- The small SEM gain was below the project-defined predeclared confirmation threshold.
- The fixed Thermal representation was not stable across development folds.
- Results do not establish causal prediction.
- The model is not claimed to reconstruct rapid local target fluctuations.
- Targets are frozen automatic estimates rather than manually validated width everywhere.

No post-hoc Track-21 tuning is proposed or permitted by the frozen protocol.
