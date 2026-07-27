# Data availability

Raw challenge data are not redistributed. Users must independently obtain authorized
access; this release does not provide or invent an external download URL or legal terms.

Expected repository-root layout:

```text
thermal/
  Thermal_8.mat
  Thermal_10.mat
  Thermal_14.mat
  Thermal_21.mat
sem/
  SEM_8/PlainImages/*.tif
  SEM_10/PlainImages/*.tif
  SEM_14/PlainImages/*.tif
  SEM_21/PlainImages/*.tif
height/
  [original Track 8, 10, 14, and 21 Height-map files in the source layout]
```

The exact Height filenames and nested paths are resolved by `nsf_fmrg_data.py` and the
packaged Height data-access helpers. Preserve the source dataset layout rather than
renaming files. Expected source types are Thermal `.mat`, SEM `.tif`, and Height-map
ASCII data.

Compact frozen labels, mappings, manifests, model JSON files, metrics, reports, audits,
and the final figure are included. They can be inspected without raw data. Raw-dependent
features and evaluations can be regenerated after data placement. TIFF and MAT payloads
are excluded because they are large source data and are not authorized for redistribution.
