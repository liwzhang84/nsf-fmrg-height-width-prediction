# Generalized SEM mapping development equivalence

- Amendment file hash: `3c0de4b78b5417cd60df27bbb9e059bcf27986aaadd8964d14e51dbf831a847a`
- Development tile counts remain N=13 for Tracks 8, 10, and 14.
- Tile order remains exactly 13→…→01.
- Recomputed mapping rows: **1192**, identical in sample ID, official x, tile, local coordinate, source path, direction, availability, and exclusions.
- Recomputed leakage-safe SEM cohort: **678 sample IDs**, exactly identical.
- Recomputed SEM-summary matrix hash: `4f8473e69b6c05d0049c920676b6cc0891b708bca176c86fd6acc51815033203`
- Recomputed fixed Thermal-shape matrix hash: `c1dba812c2cd7153954ba6424558b10dfa55007e474b16fe04d5f6c5014ae062`
- Original model lock and fitted artifacts were not modified.

The amendment changes only the accepted source schema from fixed N=13 to contiguous variable N. For development N remains 13, so no trained model input changes.
