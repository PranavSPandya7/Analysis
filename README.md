# Analysis

Environmental-processing scripts and outputs for Paper 3:

- sound classification and frequency analysis;
- solar clearness index; and
- sky-view factor (SVF).

The raw recordings and merged sensor table remain outside this repository.

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

Current SVF outputs are a `P9 / WalkG` pilot, not a complete 11-participant dataset. They are saved in `analysis_output/SVF Output/`.

## Important checks

- Use the shared environment: `C:\Users\pandya\Documents\Github\docker\ExpData\.venv`.
- Keep raw data read-only.
- Check participant/phase coverage, row counts, timestamps, missing values, and completeness flags after each run.
- Sound, clearness index, and SVF outputs are separate products; this repository does not automatically merge them into statistical inputs.
