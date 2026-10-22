"""Build the canonical 11-participant Paper 3 statistical input tables."""

from datetime import datetime, timedelta
from pathlib import Path
import argparse
import warnings

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "analysis_output"
KEY_CSV = PROJECT_ROOT.parent / "Paper3_Github" / "output" / "key.csv"
# GVI tables remain external to this environmental sharing package; read only.
GVI_INPUT_DIR = PROJECT_ROOT.parent / "Paper3_Final" / "output" / "GVI_output"

KEY = ["ParticipantID", "PhaseID", "Datetime"]
ROUTE_PHASES = {"BikeU", "WalkU", "BikeG", "WalkG", "Tram"}

GVI_FIELDS = [
    "gvi_50m", "gvi_tree_50m", "gvi_grass_50m", "gvi_plant_50m",
    "gvi_match_dist_m", "gvi_phase_mean", "tree_phase_mean",
    "grass_phase_mean", "plant_phase_mean", "gvi_image_capture_date",
]
GVI_SOURCE_ALIASES = {
    "gvi_50m": ("gvi_50m", "gvi_point_gvi"),
    "gvi_tree_50m": ("gvi_tree_50m", "gvi_point_tree"),
    "gvi_grass_50m": ("gvi_grass_50m", "gvi_point_grass"),
    "gvi_plant_50m": ("gvi_plant_50m", "gvi_point_plant"),
    "gvi_match_dist_m": ("gvi_match_dist_m",),
    "gvi_phase_mean": ("gvi_phase_mean",),
    "tree_phase_mean": ("tree_phase_mean",),
    "grass_phase_mean": ("grass_phase_mean",),
    "plant_phase_mean": ("plant_phase_mean",),
}
LEGACY_GVI_FIELDS = [
    "gvi_point_gvi", "gvi_point_tree", "gvi_point_grass", "gvi_point_plant",
    "gvi_matched_point_id", "gvi_phase_min", "gvi_phase_max", "gvi_phase_std",
]
CI_FIELDS = ["Clearness_Index_G0", "Clearness_Index", "Clearness_Index_Infer"]

# SVF is computed at approximately one-second image intervals.  The canonical
# statistical tables use the mean of available frames per key.csv-anchored
# ten-second window, exported by CamUp; the merger does not average them again.
# Class shares and Total_* columns are percent of the fisheye circle (0–100).
# Total_Vegetation_% = Tree_% + Plant_% + Grass_%.
# Total_Builtup_% = Building_% + Wall_%; ceiling only contributes to seven-class coverage.
# Gong annular TVF/BVF [0,1]: same Holmer compute_svf as SVF (Tree∪Plant∪Grass;
# Building∪Wall; Ceiling stays separate from built-up percentage and BVF).
SVF_FIELDS = [
    "Sky_%", "Tree_%", "Plant_%", "Grass_%", "Total_Vegetation_%",
    "Building_%", "Wall_%", "Ceiling_%", "Total_Builtup_%",
    "SVF", "TVF", "BVF",
]
SVF_SOURCE_MODES = {"full", "matched_8min"}
SVF_PENDING_PHASES = set()  # Completed Tram rerun is present in canonical inputs.
MATCHED_SVF_SECONDS = 480.0
UCM_LOCAL_TIME_OFFSET = timedelta(hours=2)  # Same conversion as the sound notebook / UCM builder.

# Sound join matches Updated_JOS_Environmental_sound_classification_with_YAMNet_yenshuo
# FINAL_ANALYSIS_COLUMNS (reduced export written to all_participants_sound_analysis*.csv).
# Join keys: participant, phase, start_time_s.  ucm_SND_dBA stays on the base table.
SOUND_JOIN_KEYS = ["participant", "phase", "start_time_s"]
SOUND_CATEGORY_SCORE_FIELDS = [
    "wind_score", "rural_natural_score", "vehicle_score",
    "urban_manmade_neutral_score", "urban_manmade_positive_score",
    "urban_manmade_negative_score", "mechanisms_score",
]
SOUND_CATEGORY_PROP_FIELDS = [
    "wind_proportion", "rural_natural_proportion", "vehicle_proportion",
    "urban_manmade_neutral_proportion", "urban_manmade_positive_proportion",
    "urban_manmade_negative_proportion", "mechanisms_proportion", "unidentify_proportion",
]
SOUND_FREQUENCY_FIELDS = [
    "dominant_frequency_Hz", "spectral_centroid_Hz", "spectral_rolloff85_Hz",
    "freq_proportion_20_to_200Hz", "freq_proportion_200_to_2000Hz", "freq_proportion_2000_to_20000Hz",
    "octave_proportion_31_5_Hz", "octave_proportion_63_0_Hz", "octave_proportion_125_0_Hz",
    "octave_proportion_250_0_Hz", "octave_proportion_500_0_Hz", "octave_proportion_1000_0_Hz",
    "octave_proportion_2000_0_Hz", "octave_proportion_4000_0_Hz", "octave_proportion_8000_0_Hz",
]
SOUND_MERGE_FIELDS = [
    "segment", "end_time_s", "category",
    *SOUND_CATEGORY_SCORE_FIELDS,
    *SOUND_CATEGORY_PROP_FIELDS,
    *SOUND_FREQUENCY_FIELDS,
]

LEGACY_SOUND_FIELDS = [
    'octave_proportion_31.5Hz', 'octave_proportion_63Hz', 'octave_proportion_125Hz', 'octave_proportion_250Hz', 'octave_proportion_500Hz', 'octave_proportion_1000Hz', 'octave_proportion_2000Hz', 'octave_proportion_4000Hz', 'octave_proportion_8000Hz',
    "octave_proportion_31.25Hz", "octave_proportion_62.5Hz",
    "phase_start_offset_s",
    "sound_level_dbfs", "sound_class_id", "sound_category_label",
    "sound_prop_human_activity", "sound_prop_urban_manmade",
    "sound_prop_urban_manmade_positive", "sound_prop_urban_manmade_neutral",
    "sound_prop_urban_manmade_negative", "sound_prop_mechanisms",
    "sound_prop_rural_natural", "sound_prop_unidentify",
    "sound_prop_vehicle", "sound_prop_wind",
    "sound_score_urban_manmade_positive", "sound_score_urban_manmade_neutral",
    "sound_score_urban_manmade_negative", "sound_score_mechanisms",
    "sound_score_rural_natural", "sound_score_vehicle", "sound_score_wind",
    # Dropped from reduced Sound_ucm export (full pipeline only):
    "sound_type",
    "window_duration_s", "is_complete_10s_window",
    "original_channel_count", "channel_strategy",
    "yamnet_dominant_class", "yamnet_dominant_class_index",
    "yamnet_frame_count", "yamnet_threshold",
    "original_sample_rate_Hz", "frequency_analysis_min_Hz",
    "frequency_analysis_max_Hz", "frequency_analysis_full_20khz_coverage",
]

REMOVE_UCM_AIR_QUALITY_FIELDS = [
    "ucm_AQ_CO_m1", "ucm_AQ_CO2", "ucm_AQ_NO2_m1", "ucm_AQ_O3_m1",
    "ucm_AQ_SO2_m4", "ucm_AQ_pm010", "ucm_AQ_pm025", "ucm_AQ_pm100",
]

EXCLUDED_EXPORT_FIELDS = ["questionnaire_background_name"]


def read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, low_memory=False)


def prepare_gvi(gvi: pd.DataFrame) -> pd.DataFrame:
    """Normalize legacy GVI point names to the canonical Paper 3 schema."""
    gvi = gvi.copy()
    for canonical, aliases in GVI_SOURCE_ALIASES.items():
        if canonical in gvi.columns:
            continue
        source = next((alias for alias in aliases if alias in gvi.columns), None)
        if source is not None:
            gvi[canonical] = gvi[source]
    missing = [field for field in GVI_FIELDS if field not in gvi.columns]
    if missing:
        raise ValueError(f"GVI table is missing canonical fields: {missing}")
    return gvi


def _participant_id(value: object) -> str:
    text = str(value).strip().upper()
    if text.startswith("P"):
        text = text[1:]
    return f"P{int(text)}"


def _clock_time(value: object):
    return datetime.strptime(str(value).strip(), "%H:%M:%S").time()


def load_phase_key(key_csv: Path) -> dict[tuple[str, str], tuple[str, str, str]]:
    """Load route phase date/start/end values from the authoritative key.csv."""
    key = pd.read_csv(key_csv, dtype=str, keep_default_na=False)
    required = {"Participant_ID", "Date"}
    missing = sorted(required.difference(key.columns))
    if missing:
        raise ValueError(f"key.csv is missing required columns: {missing}")

    phase_key: dict[tuple[str, str], tuple[str, str, str]] = {}
    for _, row in key.iterrows():
        try:
            participant = _participant_id(row["Participant_ID"])
        except (TypeError, ValueError):
            continue
        date_text = str(row["Date"]).strip()
        for phase in sorted(ROUTE_PHASES):
            start_col = f"{phase}_start"
            end_col = f"{phase}_end"
            if start_col not in key.columns or end_col not in key.columns:
                raise ValueError(f"key.csv is missing {start_col} or {end_col}")
            start_text = str(row[start_col]).strip()
            end_text = str(row[end_col]).strip()
            if date_text and start_text and end_text:
                phase_key[(participant, phase)] = (date_text, start_text, end_text)
    return phase_key


def _phase_duration_seconds(
    key_values: tuple[str, str, str],
    source_mode: str,
) -> float:
    """Phase length in seconds from key.csv clocks (dummy date; overnight-safe)."""
    _date_text, start_text, end_text = key_values
    if source_mode == "matched_8min":
        return float(MATCHED_SVF_SECONDS)
    start_clock = _clock_time(start_text)
    end_clock = _clock_time(end_text)
    dummy = datetime(2000, 1, 1)
    phase_start = datetime.combine(dummy.date(), start_clock)
    phase_end = datetime.combine(dummy.date(), end_clock)
    if phase_end <= phase_start:
        phase_end += timedelta(days=1)
    return float((phase_end - phase_start).total_seconds())


def load_svf_windows(
    svf_root: Path,
    key_csv: Path,
    source_mode: str,
) -> pd.DataFrame:
    """Read precomputed key-anchored ten-second means; never average them again."""
    if source_mode not in SVF_SOURCE_MODES:
        raise ValueError(f"Unknown SVF source mode: {source_mode}")
    filename = ("merged_SVF_matched_8min_10sec_avg.csv"
                if source_mode == "matched_8min" else "merged_SVF_10sec_avg.csv")
    path = svf_root / filename
    if not path.is_file():
        raise FileNotFoundError(
            f"Average SVF input missing: {path}. Generate it with "
            "analysis_scripts/SVF Computation/SVF_10sec.py; do not substitute clock-aligned archive CSVs."
        )
    print(f"SVF ten-second average input: {path}")
    svf = pd.read_csv(path)
    # Drop legacy published columns if present in older 10 s tables.
    svf = svf.drop(
        columns=[c for c in (
            "SVF_Window_Start", "Timestamp", "SVF_Selection_Method", "SVF_Aggregation_Method"
        ) if c in svf.columns],
        errors="ignore",
    )
    if "Total_Vegetation_%" not in svf.columns and "Combined_Vegetation_%" in svf.columns:
        svf = svf.rename(columns={"Combined_Vegetation_%": "Total_Vegetation_%"})
    elif "Combined_Vegetation_%" in svf.columns:
        svf = svf.drop(columns=["Combined_Vegetation_%"])
    for c in ("Tree_%", "Plant_%", "Grass_%", "Building_%", "Wall_%", "Ceiling_%"):
        if c in svf.columns:
            svf[c] = pd.to_numeric(svf[c], errors="raise")
    if "Total_Vegetation_%" not in svf.columns:
        svf["Total_Vegetation_%"] = svf["Tree_%"] + svf["Plant_%"] + svf["Grass_%"]
    svf["Total_Builtup_%"] = svf["Building_%"] + svf["Wall_%"]
    required = {"ParticipantID", "PhaseID", "SVF_Window_Start_s",
                "SVF_Frames_In_Window", *SVF_FIELDS}
    missing = required - set(svf.columns)
    if missing:
        raise ValueError(f"Selected SVF table missing columns: {sorted(missing)}")
    if svf[["ParticipantID", "PhaseID", "SVF_Window_Start_s"]].isna().any().any():
        raise ValueError("Selected SVF table has missing keys")
    counts = pd.to_numeric(svf["SVF_Frames_In_Window"], errors="raise")
    if not np.isfinite(counts).all() or not (counts.ge(0) & counts.eq(counts.round())).all():
        raise ValueError("Average SVF frame counts must be nonnegative integers")
    svf["SVF_Window_Start_s"] = pd.to_numeric(svf["SVF_Window_Start_s"], errors="raise")
    if not np.isfinite(svf["SVF_Window_Start_s"]).all():
        raise ValueError("SVF_Window_Start_s must be finite")
    if not svf["SVF_Window_Start_s"].ge(0).all():
        raise ValueError("SVF_Window_Start_s must be nonnegative")
    if not svf["SVF_Window_Start_s"].mod(10).eq(0).all():
        raise ValueError("SVF_Window_Start_s must be multiples of 10")
    phase_key = load_phase_key(key_csv)
    for (pid, phase), group in svf.groupby(["ParticipantID", "PhaseID"], sort=False):
        key = (_participant_id(pid), str(phase))
        if key not in phase_key:
            raise ValueError(f"Selected SVF group absent from key.csv: {key}")
        duration = _phase_duration_seconds(phase_key[key], source_mode)
        elapsed = group["SVF_Window_Start_s"]
        if not elapsed.lt(duration).all():
            raise ValueError(f"Selected SVF windows outside phase bounds: {key}")
    svf = svf.rename(columns={"SVF_Window_Start_s": "__svf_start_time_s"})
    keys = ["ParticipantID", "PhaseID", "__svf_start_time_s"]
    if svf.duplicated(keys).any():
        raise ValueError("Selected SVF has multiple frames in one 10-second window")
    fields = list(SVF_FIELDS)
    svf[fields] = svf[fields].apply(pd.to_numeric, errors="raise")
    pending = svf["PhaseID"].isin(SVF_PENDING_PHASES)
    empty = counts.eq(0)
    if not svf.loc[empty, SVF_FIELDS].isna().all().all():
        raise ValueError("Zero-frame SVF windows must have blank measurements")
    if not np.isfinite(svf.loc[~pending & ~empty, SVF_FIELDS].to_numpy()).all():
        raise ValueError("Selected non-pending SVF measurements must be finite")
    svf.loc[pending, fields] = np.nan
    print(f"SVF pending phases blanked: {sorted(SVF_PENDING_PHASES)}; rows: {int(pending.sum())}")
    # Keep the exported window means; no second averaging and no image metadata join.
    return svf[keys + fields + ["SVF_Frames_In_Window"]]


def assert_unique_keys(data: pd.DataFrame, name: str, keys: list[str] = KEY) -> None:
    missing = [column for column in keys if column not in data.columns]
    if missing:
        raise ValueError(f"{name} is missing key columns: {missing}")
    duplicates = int(data.duplicated(keys).sum())
    if duplicates:
        raise ValueError(f"{name} has {duplicates} duplicate merge keys")


def replace_from_source(
    base: pd.DataFrame,
    source: pd.DataFrame,
    fields: list[str],
    source_name: str,
) -> pd.DataFrame:
    present_fields = [field for field in fields if field in source.columns]
    if len(present_fields) != len(set(present_fields)):
        raise ValueError(f"Duplicate fields requested from {source_name}")
    source_part = source[KEY + present_fields].copy()
    base = base.drop(columns=[field for field in present_fields if field in base.columns])
    merged = base.merge(source_part, on=KEY, how="left", validate="one_to_one")
    if len(merged) != len(base):
        raise AssertionError(f"{source_name} changed row count during merge")
    return merged


def prepare_sound(sound: pd.DataFrame) -> pd.DataFrame:
    """Load reduced Sound_ucm FINAL_ANALYSIS export for join on phase elapsed seconds."""
    missing_keys = [column for column in SOUND_JOIN_KEYS if column not in sound.columns]
    if missing_keys:
        raise ValueError(f"Sound table is missing join keys: {missing_keys}")

    class_fields = [
        column for column in sound.columns if column.startswith("yamnet_class_")
    ]
    if class_fields:
        raise ValueError(
            "Sound table still contains yamnet_class_* columns; export "
            "all_participants_sound_analysis*.csv using FINAL_ANALYSIS_COLUMNS only."
        )

    unknown = [
        column for column in sound.columns
        if column not in SOUND_JOIN_KEYS
        and column not in {"ucm_SND_dBA", "phase_start_offset_s"}
        and column not in SOUND_MERGE_FIELDS
    ]
    if unknown:
        raise ValueError(f"Unexpected sound columns (not in FINAL_ANALYSIS schema): {unknown}")

    missing_payload = [field for field in SOUND_MERGE_FIELDS if field not in sound.columns]
    if missing_payload:
        raise ValueError(f"Sound table is missing FINAL_ANALYSIS fields: {missing_payload}")

    sound = sound.rename(
        columns={"participant": "ParticipantID", "phase": "PhaseID"}
    ).copy()
    sound["start_time_s"] = pd.to_numeric(sound["start_time_s"], errors="coerce")
    if sound["start_time_s"].isna().any():
        raise ValueError("Sound table contains non-numeric start_time_s values")

    payload_fields = [field for field in SOUND_MERGE_FIELDS if field in sound.columns]
    numeric_fields = [field for field in payload_fields if field != "category"]
    if numeric_fields:
        sound[numeric_fields] = sound[numeric_fields].apply(pd.to_numeric, errors="raise")

    present_props = [field for field in SOUND_CATEGORY_PROP_FIELDS if field in sound.columns]
    if present_props:
        prop_values = sound[present_props]
        bad_props = ((prop_values < 0.0) | (prop_values > 1.0)).any(axis=1)
        if bad_props.any():
            raise ValueError(
                f"Invalid category prevalence values for {int(bad_props.sum())} sound rows"
            )

    key = ["ParticipantID", "PhaseID", "start_time_s"]
    if int(sound.duplicated(key).sum()):
        raise ValueError("Sound table has duplicate participant-phase-start keys")

    sound["__sound_matched"] = True
    return sound[key + payload_fields + ["__sound_matched"]]


def phase_grid_elapsed(base: pd.DataFrame, key_csv: Path) -> pd.Series:
    """Map sensor clock-grid labels to window sequence anchored by the phase key."""
    datetimes = pd.to_datetime(base["Datetime"], errors="raise")
    if datetimes.isna().any():
        raise ValueError("Base table contains missing Datetime values")
    phase_key = load_phase_key(key_csv)
    elapsed = pd.Series(np.nan, index=base.index)
    for (pid, phase), group in base.loc[base["PhaseID"].isin(ROUTE_PHASES)].groupby(["ParticipantID", "PhaseID"]):
        date_text, start_text, _ = phase_key[(_participant_id(pid), phase)]
        first = datetimes.loc[group.index].min()
        anchor = pd.Timestamp(datetime.strptime(f"{first.year}-{date_text} {start_text}", "%Y-%d-%b %H:%M:%S"))
        grid_start = (anchor + UCM_LOCAL_TIME_OFFSET).floor("10s")
        seconds = (datetimes.loc[group.index] - grid_start).dt.total_seconds()
        if not (seconds.ge(0) & seconds.mod(10).eq(0)).all():
            raise ValueError(f"Base rows do not follow the key-anchored ten-second grid: {pid}/{phase}")
        elapsed.loc[group.index] = seconds
    return elapsed


def add_sound(base: pd.DataFrame, sound: pd.DataFrame) -> tuple[pd.DataFrame, int, int]:
    if "ucm_SND_dBA" not in base.columns:
        raise ValueError("Base table is missing calibrated UCM sound level ucm_SND_dBA")
    route = base["PhaseID"].isin(ROUTE_PHASES)
    working = base.copy()
    working["__sound_start_time_s"] = np.nan

    working.loc[route, "__sound_start_time_s"] = phase_grid_elapsed(working, KEY_CSV).loc[route]

    # Remove stale/legacy sound columns before inserting the finalized stream.
    stale_sound_fields = list(dict.fromkeys(
        SOUND_MERGE_FIELDS + LEGACY_SOUND_FIELDS
        + [field for field in working.columns if field.startswith("yamnet_class_")]
    ))
    working = working.drop(
        columns=[field for field in stale_sound_fields if field in working.columns]
    )

    sound_part = sound.rename(columns={"start_time_s": "__sound_start_time_s"})
    sound_key = ["ParticipantID", "PhaseID", "__sound_start_time_s"]

    merged = working.merge(
        sound_part,
        on=sound_key,
        how="left",
        validate="many_to_one",
    )

    matched = int(merged["__sound_matched"].notna().sum())
    route_rows = int(route.sum())
    merged = merged.drop(columns=["__sound_start_time_s", "__sound_matched"])
    return merged, matched, route_rows


def add_svf(
    base: pd.DataFrame,
    svf_root: Path,
    key_csv: Path,
    source_mode: str,
) -> tuple[pd.DataFrame, int, int]:
    """Join key-anchored SVF windows to participant/phase-relative base rows."""
    svf = load_svf_windows(svf_root, key_csv, source_mode)
    working = base.copy()
    working["__svf_start_time_s"] = np.nan

    route = working["PhaseID"].isin(ROUTE_PHASES)
    working.loc[route, "__svf_start_time_s"] = phase_grid_elapsed(working, key_csv).loc[route]

    dynamic_svf_fields = [c for c in svf.columns if c not in ("ParticipantID", "PhaseID", "__svf_start_time_s")]
    replaced_fields = dynamic_svf_fields + ["Combined_Vegetation_%", "SVF_Aggregation_Method", "SVF_Selection_Method"]
    working = working.drop(columns=[field for field in replaced_fields if field in working.columns])
    merged = working.merge(
        svf,
        on=["ParticipantID", "PhaseID", "__svf_start_time_s"],
        how="left",
        validate="many_to_one",
    )
    matched = int(merged["SVF"].notna().sum())
    route_rows = int(route.sum())
    route_mask = merged["PhaseID"].isin(ROUTE_PHASES)
    missing = route_mask & merged["SVF"].isna()
    merged["SVF_Status"] = "outside PhaseID"
    merged.loc[route_mask & ~missing, "SVF_Status"] = "available"
    merged.loc[missing, "SVF_Status"] = "not_applicable"
    merged.loc[missing & merged["SVF_Frames_In_Window"].eq(0),
               "SVF_Status"] = "no_frames_with_coverage_>=50%"
    if source_mode == "matched_8min":
        merged.loc[missing & merged["__svf_start_time_s"].ge(MATCHED_SVF_SECONDS),
                   "SVF_Status"] = "not_applicable"
    print(f"SVF availability: {ascii(merged.loc[route_mask, 'SVF_Status'].value_counts().to_dict())}")
    merged = merged.drop(columns="__svf_start_time_s")
    return merged, matched, route_rows


def matched_phase_rows(data: pd.DataFrame, mode: str) -> pd.DataFrame:
    """Keep route windows 0..470 s in matched exports; preserve non-route rows."""
    if mode != "matched_8min":
        return data
    elapsed = phase_grid_elapsed(data, KEY_CSV)
    keep = ~data["PhaseID"].isin(ROUTE_PHASES) | elapsed.lt(MATCHED_SVF_SECONDS)
    return data.loc[keep].reset_index(drop=True)


def preserve_column_order(data: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Retain published positions; append genuinely new fields in pipeline order."""
    ordered = [c for c in columns if c in data.columns]
    return data[ordered + [c for c in data.columns if c not in columns]]


def build(
    base_path: Path,
    gvi_path: Path,
    sound_path: Path,
    ci_path: Path,
    svf_root: Path,
    output_path: Path,
    svf_source_mode: str,
) -> None:
    base = matched_phase_rows(read_table(base_path), svf_source_mode)
    base_row_count = len(base)
    published_columns = (pd.read_csv(output_path, nrows=0).columns.tolist()
                         if output_path.is_file() else base.columns.tolist())
    gvi = prepare_gvi(read_table(gvi_path))
    sound = prepare_sound(read_table(sound_path))
    ci = read_table(ci_path)

    assert_unique_keys(base, "Base table")
    assert_unique_keys(gvi, "GVI table")
    assert_unique_keys(ci, "Clearness Index table")

    base = base.drop(columns=[field for field in LEGACY_GVI_FIELDS if field in base.columns])
    base = replace_from_source(base, gvi, GVI_FIELDS, "GVI table")
    base = replace_from_source(base, ci, CI_FIELDS, "Clearness Index table")
    base, svf_rows, route_rows = add_svf(
        base, svf_root, KEY_CSV, svf_source_mode
    )
    base, sound_rows, route_rows = add_sound(base, sound)

    removed_ucm = [
        field for field in REMOVE_UCM_AIR_QUALITY_FIELDS if field in base.columns
    ]
    base = base.drop(columns=removed_ucm)
    removed_export_fields = [field for field in EXCLUDED_EXPORT_FIELDS if field in base.columns]
    base = base.drop(columns=removed_export_fields)

    # Keep the ceiling percentage beside the other retained SVF class fields.
    if "Ceiling_%" in base.columns and "Wall_%" in base.columns:
        ordered = [column for column in base.columns if column != "Ceiling_%"]
        ordered.insert(ordered.index("Wall_%") + 1, "Ceiling_%")
        base = base[ordered]

    if len(base) != base_row_count:
        raise AssertionError("Final table row count differs from its base table")
    if not set(ROUTE_PHASES).intersection(set(base["PhaseID"].dropna().unique())):
        raise AssertionError("Final table has no route phases")

    base = preserve_column_order(base, published_columns)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(".tmp.csv")
    base.to_csv(temporary, index=False)
    temporary.replace(output_path)

    ucm_columns = [column for column in base.columns if column.startswith("ucm_")]
    print(f"Saved: {output_path}")
    print(
        f"Rows: {len(base)}; columns: {len(base.columns)}; "
        f"UCM columns: {len(ucm_columns)}"
    )
    print(f"Removed UCM air-quality fields: {removed_ucm}")
    print(f"Excluded export fields: {removed_export_fields}")
    print(f"Sound matched rows: {sound_rows}/{route_rows} route rows")
    print(f"SVF source: {svf_source_mode}; matched rows: {svf_rows}/{route_rows} route rows")
    print(f"GVI fields present: {[field for field in GVI_FIELDS if field in base.columns]}")
    print(f"SVF fields present: {[field for field in SVF_FIELDS if field in base.columns]}")
    print(f"Sound fields present: {[field for field in SOUND_MERGE_FIELDS if field in base.columns]}")
    print(f"Clearness fields present: {[field for field in CI_FIELDS if field in base.columns]}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--svf-only", action="store_true",
                        help="Refresh only SVF from current 10-second means; preserve all other merged streams.")
    args = parser.parse_args()
    jobs = [
        (
            PROJECT_ROOT.parent / "Paper3_Github" / "output" / "merged_all_11participants_8min.csv",
            GVI_INPUT_DIR / "merged_all_11participants_8min_GVI.csv",
            OUTPUT_DIR / "Sound_ucm" / "all_participants_sound_analysis_matched_8min.csv",
            OUTPUT_DIR / "Clearness Index" / "merged_all_11participants_8min_CI.csv",
            OUTPUT_DIR / "SVF Output",
            OUTPUT_DIR / "merged_all_11participants_8min.csv",
            "matched_8min",
        ),
        (
            PROJECT_ROOT.parent / "Paper3_Github" / "output" / "merged_all_11participants.csv",
            GVI_INPUT_DIR / "merged_all_11participants_GVI.csv",
            OUTPUT_DIR / "Sound_ucm" / "all_participants_sound_analysis.csv",
            OUTPUT_DIR / "Clearness Index" / "merged_all_11participants_CI.csv",
            OUTPUT_DIR / "SVF Output",
            OUTPUT_DIR / "merged_all_11participants.csv",
            "full",
        ),
    ]

    for job in jobs:
        if args.svf_only:
            base = matched_phase_rows(read_table(job[5]), job[6])
            assert_unique_keys(base, "Existing merged table")
            updated, matched, route_rows = add_svf(base, job[4], KEY_CSV, job[6])
            updated = updated.drop(columns=EXCLUDED_EXPORT_FIELDS, errors="ignore")
            if len(updated) != len(base):
                raise AssertionError("SVF-only refresh changed the base row count")
            updated = preserve_column_order(updated, base.columns.tolist())
            temporary = job[5].with_suffix(".tmp.csv")
            updated.to_csv(temporary, index=False)
            temporary.replace(job[5])
            print(f"Saved SVF-only refresh: {job[5]} ({len(updated)} rows; {matched}/{route_rows} route rows matched)")
        else:
            build(*job)


if __name__ == "__main__":
    main()
