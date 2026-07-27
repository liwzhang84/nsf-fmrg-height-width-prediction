# Final Track-21 Height-width test review

## Locked pre-test state

- Final model-lock hash: `2fb60cbb83a04ea9819c3bfff9fa39b9765dc9bad74b4a7d0693103af013be1b`
- Development artifacts existed and matched the lock before held-out source inspection.
- Required primary families were frozen; no PCA, CNN, mapping tuning, or representation search was authorized.

## Structural compatibility result

Locked-protocol incompatibility: expected exactly tiles 01–13 with 1024×768 payloads; observed tiles [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]. No held-out-specific alternative mapping is permitted.

The locked development mapping requires a complete sequence of tiles 13 through 01, with tile 13 on the physical 20 mm side. The canonical held-out source contains an additional tile 14. Dataset documentation defines the highest-numbered tile as the physical 20 mm side, so applying the locked rule would either discard the documented endpoint tile or silently create a held-out-specific mapping.

Neither action is authorized. The run stopped before Height targets, Thermal inputs, SEM features, predictions, metrics, or historical human references were loaded or evaluated. No final-test model ranking exists.

FINAL TEST BLOCKED BY LOCKED-PROTOCOL DATA INCOMPATIBILITY
