# Thermal source and loader audit

Development MAT files contain one selected `temperature_data` array each, with 400×400 spatial samples and 929/961/976 raw frames. Values are reported as uncalibrated Thermal intensity because explicit calibrated units are absent. The official mapping uses zero-based raw indices, increasing raw index with increasing official x; all 1,192 manifest rows exactly match `segment_frame_index + extracted_start` and no correction was applied.
