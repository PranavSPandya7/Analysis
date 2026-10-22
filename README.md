# Analysis

Environmental-processing scripts and outputs for Paper 3:


- sound classification and frequency analysis;
- solar clearness index; and
- sky-view factor (SVF).

Raw recordings, base sensor tables, phase keys and GVI inputs remain external. The combined environmental tables are included under `analysis_output/`.

## Folders

```text
analysis_scripts/   Processing notebooks and Python scripts
analysis_output/    Generated CSVs, images, and summaries
```

## External inputs

The current scripts use:

```text
Raw UCM/audio data:
C:\Users\pandya\OneDrive - UCL\Field experiment raw data\Complete Participantwise data\Final Data\ucm

Phase key:
C:\Users\pandya\Documents\Github\docker\Paper3_Github\output\key.csv

Merged 10-second sensor table:
C:\Users\pandya\Documents\Github\docker\Paper3_Github\output\merged_all_11participants.csv

Matched base sensor table:
C:\Users\pandya\Documents\Github\docker\Paper3_Github\output\merged_all_11participants_8min.csv

GVI inputs (read-only):
C:\Users\pandya\Documents\Github\docker\Paper3_Final\output\GVI_output\merged_all_11participants_GVI.csv
C:\Users\pandya\Documents\Github\docker\Paper3_Final\output\GVI_output\merged_all_11participants_8min_GVI.csv
```

Change these paths before running on another computer.

## Main processing files

### Sound

`analysis_scripts/Updated_JOS_Environmental_sound_classification_with_YAMNet_yenshuo.ipynb`

- Uses 10-second windows aligned to `key.csv`.
- Uses UCM `ucm_SND_dBA` for calibrated sound level.
- Runs official 521-class YAMNet on 16-kHz mono audio.
- Calculates frequency descriptors from the original WAV sample rate.

Outputs are saved in `analysis_output/Sound_ucm/`, including per-phase files and combined full-duration and matched-eight-minute CSVs. Short recordings remain shorter; missing windows are not invented.

### Clearness index

`analysis_scripts/compute_clearness_index.py`

Calculates `k_t = G_H / G_0` from the merged irradiance and solar-altitude fields. Outputs are saved in `analysis_output/Clearness Index/`.

### Sky-view factor

`analysis_scripts/SVF Computation/compute_svf_camup.py`

Uses upward-facing camera frames, Xception/ADE20K segmentation, and the Holmer SVF calculation. Phase timing is taken from `key.csv`; the matched output covers the first 480 seconds.

The two published SVF tables (`analysis_output/SVF Output/merged_SVF_10sec_avg.csv` and `merged_SVF_matched_8min_10sec_avg.csv`) contain the ten-second averages for all 11 retained participants and five route phases. The per-frame SVF tables, processed images and participant-level caches are not included in this sharing package. Recomputing SVF requires the external raw CamUp images.

Run `analysis_scripts/SVF Computation/SVF_10sec.py` to rebuild the ten-second averages from the per-frame SVF tables written by `compute_svf_camup.py` (not included here). Frames qualify when their seven retained-class percentages sum to at least 50%. Windows without qualifying frames retain blank measurements, zero eligible frames and status `no_frames_with_coverage_>=50%`.

### Combined tables

Run `analysis_scripts/merge_11participant_statistical_inputs.py` after the required inputs are available. It reads sound, SVF averages, clearness and external GVI tables, then writes both `merged_all_11participants*.csv` files under `analysis_output/`.

Matched exports retain route windows starting at 0 through 470 seconds; route rows at or beyond 480 seconds are excluded. Non-route rows are preserved. Current outputs have 8,038 full rows and 5,023 matched rows, each with 223 columns.

Sound/SVF intervals start at the exact phase key time, while sensor labels use the floored ten-second grid. Their paired intervals can differ by 0–9 seconds; merging does not reaggregate the sensor data.

## Important checks

- Keep raw data read-only.
- Check participant/phase coverage, row counts, timestamps, missing values, and completeness flags after each run.
- External inputs must exist at the configured paths; this repository is not a self-contained raw-data archive.
