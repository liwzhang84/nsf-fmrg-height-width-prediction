# Track-21 Tile-14 metadata compatibility review

- Tile IDs are exactly contiguous 1 through 14.
- All files match the canonical `Plain_SEM_21_NN.tif` naming convention.
- All payloads report 1024×768, TIFF mode P, and 6.235 µm/pixel.
- All 14 file hashes are distinct.
- Stage Y is constant at 28.8874 mm.
- Stage X decreases monotonically from Tile 01 (76.5210 mm) to Tile 14 (0.4057 mm), supporting physical order 14→…→01.
- Dataset documentation assigns the highest-numbered canonical tile to the 20 mm side and Tile 01 to the 100 mm side.
- No filename, tag, stage coordinate, dimension, mode, or hash conflict identifies Tile 14 as a duplicate, thumbnail, calibration frame, or sequence-external image.
- This audit read TIFF metadata only. No image pixel array, Height label, Thermal frame, prediction, metric, or human boundary was loaded.

TRACK21 TILE14 IS A VALID CANONICAL SEQUENCE MEMBER
