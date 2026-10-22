"""
compute_svf_camup.py
====================
UCM backpack CamUp pipeline for semantic class shares and annular SVF, TVF and BVF.
Relies on neginja/MSE-SkyViewFactor-master for optical rectification & Holmer (1992) SVF math,
combined with DeepLabV3+ Xception-65 on ADE20K for semantic sky segmentation.
"""

from __future__ import annotations

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "4")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")

import sys
import csv
import re
import math
import types
import json
import hashlib
import argparse
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

def load_pixel_lib():
    """Load PixelLib only when this script runs Xception segmentation."""
    global semantic_segmentation, ade20k_map_color_mask
    global labelAde20k_to_color_image, create_ade20k_label_namemap
    try:
        import tensorflow.keras as keras_api
        import tensorflow.keras.layers as keras_layers
        import tensorflow.python.keras.models as internal_models
        layers_module = sys.modules.get("tensorflow.python.keras.layers")
        if layers_module is None:
            layers_module = types.ModuleType("tensorflow.python.keras.layers")
            sys.modules["tensorflow.python.keras.layers"] = layers_module
        for attr in dir(keras_layers):
            setattr(layers_module, attr, getattr(keras_layers, attr))
        internal_models.Model = keras_api.Model
    except ImportError:
        pass
    from pixellib.semantic import (semantic_segmentation, ade20k_map_color_mask,
                                  labelAde20k_to_color_image, create_ade20k_label_namemap)

# ---------------------------------------------------------------------------
# Fixed scientific configuration
# ---------------------------------------------------------------------------
MODEL_PATH = SCRIPT_DIR / "deeplabv3_xception65_ade20k.h5"
UCM_RAW_ROOT = Path(r"C:\Users\pandya\OneDrive - UCL\Field experiment raw data\Complete Participantwise data\Final Data\ucm")
OUTPUT_ROOT = SCRIPT_DIR.parent.parent / "analysis_output" / "SVF Output"
KEY_CSV = SCRIPT_DIR.parent.parent.parent / "Paper3_Github" / "output" / "key.csv"

MSE_REFERENCE_CALIBRATION_TABLE = [24, 25, 25, 26, 26, 27, 27, 29, 29, 32, 32, 36, 36, 40, 41, 43, 44]
MSE_REFERENCE_RADIUS_PX = float(sum(MSE_REFERENCE_CALIBRATION_TABLE))
DEFAULT_SVF_RINGS = 120
EXPECTED_PHASES = ("BikeG", "BikeU", "WalkG", "WalkU", "Tram")
MATCHED_SECONDS = 480.0
MATCHED_PARTICIPANTS = ("P4", "P8", "P9", "P10", "P11", "P12", "P13", "P14", "P15", "P16", "P17")
# Tram is now part of the canonical ceiling-inclusive SVF outputs.
# Keep this empty so Tram values are retained in merged and 10-second tables.
SVF_PENDING_PHASES = set()
# Scene class shares and totals are percent of the fisheye circle (0-100).
# Total_Vegetation_% = Tree_% + Plant_% + Grass_%.
# Total_Builtup_% = Building_% + Wall_%.
SVF_MEASUREMENT_FIELDS = [
    "Sky_%", "SVF", "TVF", "BVF", "Tree_%", "Plant_%", "Grass_%",
    "Total_Vegetation_%", "Building_%", "Wall_%", "Ceiling_%", "Total_Builtup_%",
]

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
def load_mse_core():
    """Load OpticalRectifier and SkyViewFactorCalculator from MSE-SkyViewFactor."""
    candidates = [
        SCRIPT_DIR / "MSE-SkyViewFactor-master" / "MSE-SkyViewFactor-master" / "core",
        SCRIPT_DIR / "MSE-SkyViewFactor-master" / "core",
        SCRIPT_DIR.parent / "scripts" / "SVF Computation" / "MSE-SkyViewFactor-master" / "MSE-SkyViewFactor-master" / "core",
        Path(r"C:\Users\pandya\Documents\Github\docker\Paper3_Final\scripts\SVF Computation\MSE-SkyViewFactor-master\MSE-SkyViewFactor-master\core"),
    ]
    mse_core = next((c for c in candidates if c.is_dir()), None)
    if mse_core is None:
        raise FileNotFoundError("MSE core repository not found.")

    tools_mod = types.ModuleType("tools")
    sys.modules.setdefault("tools", tools_mod)

    prof = types.ModuleType("tools.FuncTimeProfiler")
    prof.profile = lambda func: func
    sys.modules.setdefault("tools.FuncTimeProfiler", prof)

    fm = types.ModuleType("tools.FileManager")
    fm.FileManager = object()
    sys.modules.setdefault("tools.FileManager", fm)

    mc = types.ModuleType("tools.MaskCreator")
    class MaskCreator:
        @staticmethod
        def create_circle_mask(d):
            d = int(round(d))
            m = np.zeros((d, d), np.uint8)
            cv2.circle(m, (d // 2, d // 2), d // 2, 255, -1)
            return m
    mc.MaskCreator = MaskCreator
    sys.modules.setdefault("tools.MaskCreator", mc)

    # 1. Load SkyViewFactorCalculator in memory (patching print syntax in RAM)
    calc_src = re.sub(r"print\s+(.*)", r"print(\1)", (mse_core / "SkyViewFactorCalculator.py").read_text(encoding="utf-8"))
    calc_mod = types.ModuleType("SkyViewFactorCalculator")
    exec(calc_src, calc_mod.__dict__)

    # 2. Load OpticalRectifier in memory (patching print syntax in RAM)
    rect_src = re.sub(r"print\s+(.*)", r"print(\1)", (mse_core / "OpticalRectifier.py").read_text(encoding="utf-8"))
    rect_mod = types.ModuleType("OpticalRectifier")
    exec(rect_src, rect_mod.__dict__)

    return rect_mod.OpticalRectifier, calc_mod.SkyViewFactorCalculator

MSEOpticalRectifier, SkyViewFactorCalculator = load_mse_core()


class OpticalRectifier:
    """Adapter around the read-only MSE rectifier for rectangular CamUp frames."""
    def __init__(self, calibration_increments, width, height, center=None, raw_radius=None,
                 rectified_radius=None, calibration_ref_radius_px=MSE_REFERENCE_RADIUS_PX):
        self.width, self.height = int(width), int(height)
        self.center = (self.width / 2.0, self.height / 2.0) if center is None else (float(center[0]), float(center[1]))
        self.raw_radius = min(self.width, self.height) / 2.0 if raw_radius is None else float(raw_radius)
        self.rectified_radius = self.raw_radius if rectified_radius is None else float(rectified_radius)

        if not np.isfinite(self.raw_radius) or self.raw_radius <= 0:
            raise ValueError("raw_lens_radius_px must be finite and > 0")
        if not np.isfinite(self.rectified_radius) or self.rectified_radius <= 0:
            raise ValueError("rectified_radius_px must be finite and > 0")
        if not np.isclose(self.raw_radius, self.rectified_radius, rtol=0.0, atol=1e-6):
            raise ValueError(
                "This MSE adapter requires raw_lens_radius_px == rectified_radius_px; "
                "unequal radii are not silently approximated."
            )

        radius_px = int(round(self.rectified_radius))
        side = 2 * radius_px
        cx, cy = float(self.center[0]), float(self.center[1])
        x0 = int(round(cx - radius_px))
        y0 = int(round(cy - radius_px))
        x1, y1 = x0 + side, y0 + side
        if x0 < 0 or y0 < 0 or x1 > self.width or y1 > self.height:
            raise ValueError(
                f"Hemisphere bounding square [{x0}:{x1}, {y0}:{y1}] "
                f"does not fit inside image {self.width}x{self.height}."
            )

        # The legacy MSE OpticalRectifier assumes a square image and internally
        # defines the corrected hemisphere radius as imgWidth/2.  Therefore run
        # it on the actual hemisphere square (480x480 for the default 640x480
        # CamUp frame), then translate its source mappings back into full-frame
        # coordinates.  The vendored MSE implementation remains unmodified.
        scale = self.raw_radius / float(calibration_ref_radius_px)
        scaled_table = [v * scale for v in calibration_increments]
        self._engine = MSEOpticalRectifier(scaled_table, 180, side, side)

        self.mapping_x = np.full((self.height, self.width), -1.0, dtype=np.float32)
        self.mapping_y = np.full((self.height, self.width), -1.0, dtype=np.float32)
        self.mapping_x[y0:y1, x0:x1] = self._engine.mapping_x + float(x0)
        self.mapping_y[y0:y1, x0:x1] = self._engine.mapping_y + float(y0)

        self.mask = np.zeros((self.height, self.width), dtype=np.uint8)
        center_px = (int(round(cx)), int(round(cy)))
        cv2.circle(self.mask, center_px, radius_px, 255, -1)

        # Geometry sanity check.  The calibration polynomial can move the exact
        # center by a few pixels, but it must still map close to the measured
        # optical center.
        mx = float(self.mapping_x[center_px[1], center_px[0]])
        my = float(self.mapping_y[center_px[1], center_px[0]])
        if not np.isfinite(mx) or not np.isfinite(my) or math.hypot(mx - cx, my - cy) > 5.0:
            raise RuntimeError(
                f"MSE rectification center sanity check failed: mapped center=({mx:.2f},{my:.2f}), "
                f"expected near ({cx:.2f},{cy:.2f})."
            )

        self.virtual_side_px = side
        self.square_bounds = (x0, x1, y0, y1)
        print(
            f"MSE rectification geometry: virtual={side}x{side}, "
            f"real={self.width}x{self.height}, center=({cx:.1f},{cy:.1f}), radius={radius_px}px",
            flush=True,
        )

    def rectify_image(self, img_bgr):
        if img_bgr.shape[:2] != (self.height, self.width):
            raise ValueError(
                f"CamUp frame size changed from {self.width}x{self.height} to "
                f"{img_bgr.shape[1]}x{img_bgr.shape[0]}."
            )
        rectified = cv2.remap(
            img_bgr, self.mapping_x, self.mapping_y, cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        )
        rectified = cv2.bitwise_and(rectified, rectified, mask=self.mask)
        return rectified, self.mask


def compute_svf(binary_mask, number_of_steps, center, radius):
    """Execute Holmer (1992) SVF calculation via SkyViewFactorCalculator."""
    factor = SkyViewFactorCalculator.compute_factor(binary_mask, int(number_of_steps), center, radius)
    if not np.isfinite(factor) or factor < -1e-6 or factor > 1.0 + 1e-6:
        raise RuntimeError(f"Computed invalid SVF value: {factor}")
    return float(np.clip(factor, 0.0, 1.0))


def run_svf_self_test(number_of_steps=DEFAULT_SVF_RINGS):
    """Fail fast if the Holmer implementation violates simple geometric identities."""
    side = 480
    center = (side // 2, side // 2)
    radius = side // 2
    steps = min(int(number_of_steps), radius)

    circle = np.zeros((side, side), np.uint8)
    cv2.circle(circle, center, radius, 255, -1)
    empty = np.zeros_like(circle)
    full = circle.copy()

    half = np.zeros_like(circle)
    cv2.rectangle(half, (0, 0), (side - 1, center[1]), 255, -1)
    half = cv2.bitwise_and(half, circle)

    f0 = compute_svf(empty, steps, center, radius)
    f1 = compute_svf(full, steps, center, radius)
    fhalf = compute_svf(half, steps, center, radius)

    if abs(f0) > 1e-6 or abs(f1 - 1.0) > 0.01 or abs(fhalf - 0.5) > 0.02:
        raise RuntimeError(f"SVF self-test failed: empty={f0}, full={f1}, half={fhalf}")


def parse_calibration_table(text):
    """Parse comma-separated positive radial increments."""
    values = [float(x.strip()) for x in str(text).split(",") if x.strip()]
    if len(values) < 4 or any(v <= 0 or not np.isfinite(v) for v in values):
        raise ValueError("Calibration increments must contain >= 4 positive numbers")
    return values


# ---------------------------------------------------------------------------
# Telemetry & Validation Helpers
# ---------------------------------------------------------------------------
def load_ucm_telemetry(session_dir: Path) -> pd.DataFrame:
    csv_file = session_dir / "data.csv"
    if not csv_file.exists():
        return pd.DataFrame()
    lines = csv_file.read_text(encoding="utf-8", errors="replace").splitlines()
    header_i = next(i for i, line in enumerate(lines) if "GPS_time," in line)
    columns = [x.strip() for x in lines[header_i].lstrip("# ").split(",")]
    return pd.read_csv(csv_file, skiprows=header_i + 2, names=columns)


def default_telemetry():
    return {k: np.nan for k in ["GPS_lat", "GPS_lon", "Altitude_m", "Speed_kmh", "Heading_deg",
                                "PM25_ugm3", "PM10_ugm3", "CO2_ppm", "Temperature_C", "Humidity_%", "Noise_dB"]}


def parse_timestamp_from_filename(filename: str) -> str:
    parts = Path(filename).stem.split("_")
    if len(parts) >= 3 and len(parts[1]) == 8 and len(parts[2]) == 6:
        d, t = parts[1], parts[2]
        return f"{d[:4]}-{d[4:6]}-{d[6:8]} {t[:2]}:{t[2:4]}:{t[4:6]}"
    return ""


def load_phase_key(key_csv: Path) -> pd.DataFrame:
    if not key_csv.is_file():
        raise FileNotFoundError(f"Missing phase key file: {key_csv}")
    df = pd.read_csv(key_csv, dtype=str)
    if "Participant_ID" not in df.columns or "Date" not in df.columns:
        raise ValueError("key.csv must contain Participant_ID and Date columns")
    df["Participant_ID"] = df["Participant_ID"].map(
        lambda x: str(x).strip().upper() if str(x).strip().upper().startswith("P") else f"P{int(float(x))}"
    )
    return df.set_index("Participant_ID")


def select_images_to_key_phase(images: list[Path], key_df: pd.DataFrame, pid: str, phase: str) -> list[Path]:
    """Keep only images whose filename timestamps fall inside key.csv bounds."""
    if pid not in key_df.index:
        raise ValueError(f"No key.csv row for {pid}")
    row = key_df.loc[pid]
    start_col, end_col = f"{phase}_start", f"{phase}_end"
    if start_col not in row.index or end_col not in row.index:
        raise ValueError(f"key.csv is missing {start_col} or {end_col}")

    parsed = []
    for image in images:
        timestamp = pd.to_datetime(parse_timestamp_from_filename(image.name), errors="coerce")
        if pd.notna(timestamp):
            parsed.append((image, timestamp))
    if not parsed:
        return []

    image_date = parsed[0][1].date()
    key_date = datetime.strptime(f"{image_date.year}-{row['Date']}", "%Y-%d-%b").date()
    phase_start = datetime.combine(key_date, datetime.strptime(str(row[start_col]).strip(), "%H:%M:%S").time())
    phase_end = datetime.combine(key_date, datetime.strptime(str(row[end_col]).strip(), "%H:%M:%S").time())
    if phase_end <= phase_start:
        phase_end += timedelta(days=1)
    return [image for image, timestamp in parsed if phase_start <= timestamp.to_pydatetime() < phase_end]


def trim_svf_to_matched_8min(df: pd.DataFrame, key_df: pd.DataFrame, pid: str, phases: list[str]) -> pd.DataFrame:
    """Keep the first 480 s of each phase using key.csv start times."""
    if df.empty:
        return df.copy()
    if pid not in key_df.index:
        raise ValueError(f"No key.csv row for {pid}")
    row = key_df.loc[pid]
    timestamps = pd.to_datetime(df["Timestamp"], errors="coerce")
    keep = pd.Series(False, index=df.index)
    for phase in phases:
        phase_rows = df["PhaseID"].eq(phase)
        valid_ts = timestamps[phase_rows].dropna()
        if valid_ts.empty:
            continue
        image_date = valid_ts.iloc[0].date()
        key_date = datetime.strptime(f"{image_date.year}-{row['Date']}", "%Y-%d-%b").date()
        start_time = datetime.strptime(str(row[f"{phase}_start"]).strip(), "%H:%M:%S").time()
        phase_start = datetime.combine(key_date, start_time)
        phase_end = phase_start + timedelta(seconds=MATCHED_SECONDS)
        keep |= phase_rows & timestamps.ge(phase_start) & timestamps.lt(phase_end)
    return df.loc[keep].copy()


def normalize_svf_export_frame(data: pd.DataFrame) -> pd.DataFrame:
    """Apply current derived totals to fresh and resumed records; drop unused empty fields."""
    out = data.copy()
    legacy = [c for c in ("Manual_Mask_Path", "Manual_Sky_IoU", "Manual_Sky_Dice",
                           "Manual_Pixel_Accuracy", "Manual_SVF", "SVF_Abs_Error_vs_Manual") if c in out]
    populated = [c for c in legacy if out[c].notna().any()]
    if populated:
        raise ValueError(f"Refusing to discard populated manual-validation fields: {populated}")
    out = out.drop(columns=legacy)
    if {"Building_%", "Wall_%"}.issubset(out.columns):
        out["Total_Builtup_%"] = out[["Building_%", "Wall_%"]].apply(pd.to_numeric, errors="raise").sum(axis=1)
    return out


def _write_merged_svf_table(stem: str, filename: str) -> Path | None:
    """Concatenate canonical per-participant SVF tables when all 11 exist."""
    files = [OUTPUT_ROOT / pid / f"{pid}_SVF{stem}.csv" for pid in MATCHED_PARTICIPANTS]
    if not all(path.is_file() for path in files):
        return None
    factor_fields = ["SVF", "TVF", "BVF"]
    missing_factors = [path.name for path in files
                       if not set(factor_fields).issubset(pd.read_csv(path, nrows=0).columns)]
    if missing_factors:
        print(f"Skipping merged SVF output; participant files lack TVF/BVF: {missing_factors}", flush=True)
        return None
    merged = normalize_svf_export_frame(pd.concat((pd.read_csv(path) for path in files), ignore_index=True))
    factors = merged[factor_fields].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(factors.to_numpy()).all() or ((factors < 0) | (factors > 1)).any().any():
        raise ValueError("Cannot merge missing or out-of-range SVF/TVF/BVF values.")
    pending = merged["PhaseID"].isin(SVF_PENDING_PHASES)
    fields = [c for c in SVF_MEASUREMENT_FIELDS if c in merged]
    merged.loc[pending, fields] = np.nan
    merged = merged.sort_values(["ParticipantID", "PhaseID", "Timestamp", "Filename"], na_position="last")
    output = OUTPUT_ROOT / filename
    temporary = OUTPUT_ROOT / f".{filename}.tmp.csv"
    merged.to_csv(temporary, index=False)
    os.replace(temporary, output)
    return output


def write_merged_svf() -> Path | None:
    """Write one merged full-phase table only when all 11 canonical participants exist."""
    return _write_merged_svf_table("", "merged_SVF.csv")


def write_merged_matched_svf() -> Path | None:
    """Write one merged matched table only when all 11 canonical participants exist."""
    return _write_merged_svf_table("_matched_8min", "merged_SVF_matched_8min.csv")


class ClassTableWriter:
    def __init__(self, output_root):
        self.path = output_root / "151_classes_SVF.csv"
        self.keys = ["ParticipantID", "PhaseID", "Filename"]
        names = create_ade20k_label_namemap()
        self.labels = [names[i] for i in range(151)]
        self.columns = self.keys + ["Decode_Error", "Class_Data_Source"] + self.labels
        self.saved = set()
        if self.path.exists():
            existing = pd.read_csv(self.path)
            if not set(self.columns).issubset(existing.columns):
                raise ValueError("Existing 151-class file has missing columns")
            values = existing[self.labels].to_numpy()
            if (existing[self.keys].isna().any().any() or existing.duplicated(self.keys).any()
                    or not np.isfinite(values).all() or (values < 0).any()
                    or not np.allclose(values.sum(axis=1), 100, atol=1e-8, rtol=0)):
                raise ValueError("Existing 151-class file contains invalid or duplicate records")
            self.columns = existing.columns.tolist()
            self.saved = set(existing[self.keys].itertuples(index=False, name=None))

    def save(self, record):
        key = tuple(record[column] for column in self.keys)
        if key in self.saved:
            return
        values = np.array([record[column] for column in self.labels], dtype=float)
        if (not np.isfinite(values).all() or (values < 0).any()
                or not np.isclose(values.sum(), 100, atol=1e-8, rtol=0)):
            raise ValueError(f"Invalid 151-class percentages for {key}")
        new_file = not self.path.exists()
        with self.path.open("a", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=self.columns)
            if new_file:
                writer.writeheader()
            writer.writerow(record)
            stream.flush()
            os.fsync(stream.fileno())
        self.saved.add(key)


def write_svf_class_exports(output_root=OUTPUT_ROOT, data=None):
    """Check all 151 model classes against the selected-class measurements."""
    if data is None:
        data = pd.read_csv(output_root / "merged_SVF.csv")
    keys = ["ParticipantID", "PhaseID", "Filename"]
    fields = ["Sky_%", "Tree_%", "Plant_%", "Grass_%", "Building_%", "Wall_%", "Ceiling_%"]
    retained = data[keys + fields].copy()
    values = retained[fields].to_numpy()
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Selected-class percentages must be finite and nonnegative")
    output = output_root / "retained_classes_SVF.csv"
    temporary = output.with_suffix(".tmp.csv")
    cache = output_root / "151_classes_SVF.csv"
    if not cache.exists():
        raise ValueError("151-class export missing; original labels must be recovered")
    classes = pd.read_csv(cache)
    names = create_ade20k_label_namemap()
    labels = [names[i] for i in range(151)]
    if classes.duplicated(keys).any() or not np.isfinite(classes[labels].to_numpy()).all():
        raise ValueError("151-class export has duplicate keys or missing labels")
    if (classes[labels] < 0).any().any():
        raise ValueError("151-class export contains negative percentages")
    totals = classes[labels].sum(axis=1).to_numpy()
    invalid = ~np.isclose(totals, 100, atol=1e-8, rtol=0)
    if invalid.any():
        examples = classes.loc[invalid, keys].head(10).to_dict("records")
        raise ValueError(
            f"151-class totals differ from 100% in {int(invalid.sum())} rows; "
            f"maximum absolute difference {np.max(np.abs(totals - 100)):.12g} percentage points; "
            f"examples: {examples}. No class exports were replaced."
        )
    wanted = pd.MultiIndex.from_frame(data[keys])
    indexed = classes.set_index(keys)
    if not wanted.isin(indexed.index).all():
        raise ValueError("151-class export is missing images; do not reuse stale data")
    classes = indexed.loc[wanted].reset_index()
    comparison = {"sky": "Sky_%", "tree": "Tree_%", "plant": "Plant_%",
                  "grass": "Grass_%", "building": "Building_%", "wall": "Wall_%",
                  "ceiling": "Ceiling_%"}
    for label, column in comparison.items():
        difference = np.abs(classes[label].to_numpy() - data[column].to_numpy())
        invalid = difference > 1e-8
        if invalid.any():
            examples = classes.loc[invalid, keys].head(10).to_dict("records")
            raise ValueError(
                f"Full-class export disagrees with merged {column} in {int(invalid.sum())} rows; "
                f"maximum absolute difference {difference.max():.12g} percentage points; "
                f"examples: {examples}. No class exports were replaced."
            )
    omitted = [label for label in labels if label not in comparison]
    retained["Other_%"] = classes[omitted].sum(axis=1).to_numpy()
    retained.to_csv(temporary, index=False)
    os.replace(temporary, output)
    classes.to_csv(cache.with_suffix(".tmp.csv"), index=False)
    os.replace(cache.with_suffix(".tmp.csv"), cache)
    print(f"Class exports: {cache} and {output}", flush=True)


def resolve_session_dir(phase_ucm_dir: Path) -> Path:
    candidates = [phase_ucm_dir] if (phase_ucm_dir / "CamUp").is_dir() else []
    candidates.extend(d for d in sorted(phase_ucm_dir.iterdir()) if d.is_dir() and (d / "CamUp").is_dir())
    if not candidates:
        raise FileNotFoundError(f"No CamUp source found under: {phase_ucm_dir}")
    if len(candidates) > 1:
        raise RuntimeError("Ambiguous UCM source: " + ", ".join(str(d) for d in candidates))
    return candidates[0]


def union_class_masks(extracted_objects, shape_hw, fisheye_mask):
    h, w = shape_hw
    class_masks = {}
    for obj in extracted_objects:
        cname, mask = obj.get("class_name"), obj.get("masks")
        if not cname or mask is None:
            continue
        m = np.asarray(mask)
        m = np.any(m > 0, axis=-1).astype(np.uint8) if m.ndim == 3 else (m > 0).astype(np.uint8)
        if m.shape[:2] != (h, w):
            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
        m = cv2.bitwise_and(np.where(m > 0, 255, 0).astype(np.uint8), fisheye_mask)
        class_masks[cname] = cv2.bitwise_or(class_masks[cname], m) if cname in class_masks else m
    return class_masks


def save_grouped_overlay(out_file, image, groups, valid, sky, vegetation, built, svf, tvf, bvf):
    palette = np.array([(128,128,128), (233,180,86), (0,100,0), (60,150,60),
                        (70,190,100), (0,94,213), (70,130,190), (170,100,160)], dtype=np.uint8)
    overlay = cv2.addWeighted(image, 0.3, palette[groups], 0.7, 0)
    overlay[~valid] = 0
    canvas = np.zeros((image.shape[0] + 66, image.shape[1], 3), dtype=np.uint8)
    canvas[66:] = overlay
    lines = [f"Sky%= {sky:.2f} (SVF= {svf:.4f})",
             f"Vegetation%= {vegetation:.2f} (TVF= {tvf:.4f})",
             f"Build%= {built:.2f} (BVF= {bvf:.4f})"]
    for index, text in enumerate(lines):
        cv2.putText(canvas, text, (6, 18 + 20 * index), cv2.FONT_HERSHEY_SIMPLEX,
                    0.48, (255,255,255), 1, cv2.LINE_AA)
    if not cv2.imwrite(str(out_file), canvas, [cv2.IMWRITE_JPEG_QUALITY, 90]):
        raise OSError(f"Could not write {out_file}")


def save_overlay(out_file, input_bgr, labels, eval_mask, eval_center, eval_radius, values):
    labels = cv2.resize(labels.astype(np.uint8), (input_bgr.shape[1], input_bgr.shape[0]),
                        interpolation=cv2.INTER_NEAREST)
    names = create_ade20k_label_namemap()
    retained = ["sky", "tree", "plant", "grass", "building", "wall", "ceiling"]
    groups = np.zeros(labels.shape, dtype=np.uint8)
    for index, name in enumerate(retained, 1):
        for label_id, label_name in names.items():
            if label_name == name:
                groups[labels == label_id] = index
    save_grouped_overlay(out_file, input_bgr, groups, eval_mask > 0, *values)


def append_partial_csv_rows(path: Path, rows: list) -> None:
    """Append checkpoint rows without rewriting the full partial CSV."""
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    write_header = not path.is_file() or path.stat().st_size == 0
    with open(path, "a", newline="", encoding="utf-8") as handle:
        frame.to_csv(handle, index=False, header=write_header)
        handle.flush()
        os.fsync(handle.fileno())


# ---------------------------------------------------------------------------
# Main Execution Pipeline
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="CamUp view factors (Xception/ADE20K + Holmer annular method)")
    parser.add_argument("--participant", default="P4", help="Participant ID")
    parser.add_argument("--phase", default="all", help="Phase ID, list, or 'all'")
    parser.add_argument("--key-csv", type=Path, default=KEY_CSV)
    parser.add_argument("--camera", default="CamUp", choices=["CamUp"], help="Restricted to CamUp")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--save-sky-masks", action="store_true")
    parser.add_argument("--svf-rings", type=int, default=DEFAULT_SVF_RINGS)
    parser.add_argument("--calibration-table", type=str, default=None)
    calibration_group = parser.add_mutually_exclusive_group()
    calibration_group.add_argument("--calibration-verified", action="store_true")
    calibration_group.add_argument("--allow-unverified-calibration", action="store_true")
    parser.add_argument("--center-x", type=float, default=None)
    parser.add_argument("--center-y", type=float, default=None)
    parser.add_argument("--raw-lens-radius-px", "--lens-radius-px", dest="raw_lens_radius_px", type=float, default=None)
    parser.add_argument("--rectified-radius-px", type=float, default=None)
    parser.add_argument("--calibration-ref-radius-px", type=float, default=MSE_REFERENCE_RADIUS_PX)
    parser.add_argument("--no-square-crop", action="store_true")
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT,
                        help="Output root for the CamUp SVF results.")
    args = parser.parse_args()

    # Pre-flight validations
    if args.step < 1:
        raise ValueError("--step must be >= 1")
    if args.svf_rings < 1:
        raise ValueError("--svf-rings must be >= 1")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be >= 1 when supplied")
    if args.calibration_ref_radius_px <= 0:
        raise ValueError("--calibration-ref-radius-px must be > 0")
    pid, phase = args.participant.strip().upper(), args.phase.strip()
    key_df = load_phase_key(args.key_csv)
    calibration = parse_calibration_table(args.calibration_table) if args.calibration_table else list(MSE_REFERENCE_CALIBRATION_TABLE)
    calib_qc = "VERIFIED_FOR_CURRENT_LENS" if args.calibration_verified else "REFERENCE_OR_CUSTOM_TABLE_NOT_MARKED_LENS_VERIFIED"
    calib_src = "custom_cli_table" if args.calibration_table else "neginja_MSE_SkyViewFactor_reference_table"

    print("=" * 80)
    print(f"UCM CAMUP SVF PIPELINE | Participant: {pid} | Phase: {phase} | Calib QC: {calib_qc}")
    print("=" * 80, flush=True)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    run_svf_self_test(args.svf_rings)

    # Resolve phases. Canonical "all" is the fixed study phase set; it must not
    # silently shrink/expand based on whichever folders happen to be present.
    if phase.lower() == "all":
        phases = list(EXPECTED_PHASES)
        missing_sources = []
        for ph in phases:
            ph_ucm = UCM_RAW_ROOT / pid / ph / "ucm"
            if not ph_ucm.is_dir():
                missing_sources.append(f"{ph}: missing ucm folder")
                continue
            try:
                session = resolve_session_dir(ph_ucm)
                cam_dir = session / "CamUp"
                if not any(p.suffix.lower() in {".jpg", ".jpeg", ".png"} for p in cam_dir.iterdir()):
                    missing_sources.append(f"{ph}: CamUp has no images")
            except Exception as exc:
                missing_sources.append(f"{ph}: {exc}")
        if missing_sources:
            raise RuntimeError(
                "Canonical phase=all source check failed: " + " | ".join(missing_sources)
            )
    else:
        phases = [p.strip() for p in phase.split(",") if p.strip()]
        if not phases:
            raise ValueError("No phase supplied")

    # Read first image to lock canvas geometry
    first_image = None
    for ph in phases:
        ph_dir = UCM_RAW_ROOT / pid / ph / "ucm"
        if ph_dir.exists():
            try:
                cands = sorted(p for p in (resolve_session_dir(ph_dir) / "CamUp").iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
                if cands and (first_image := cv2.imread(str(cands[0]), cv2.IMREAD_COLOR)) is not None:
                    break
            except Exception:
                continue
    if first_image is None:
        raise FileNotFoundError(f"No CamUp images found across phases {phases} for {pid}")

    h0, w0 = first_image.shape[:2]
    center = (w0 / 2.0 if args.center_x is None else float(args.center_x), h0 / 2.0 if args.center_y is None else float(args.center_y))
    raw_lens_radius = min(w0, h0) / 2.0 if args.raw_lens_radius_px is None else float(args.raw_lens_radius_px)
    rectified_radius = raw_lens_radius if args.rectified_radius_px is None else float(args.rectified_radius_px)

    rectifier = OpticalRectifier(calibration, w0, h0, center, raw_lens_radius, rectified_radius, args.calibration_ref_radius_px)
    if args.svf_rings > int(round(rectified_radius)):
        raise ValueError(
            f"--svf-rings ({args.svf_rings}) cannot exceed rectified radius "
            f"({int(round(rectified_radius))} px)."
        )

    # Execution signatures
    model_sha256 = hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest()
    run_config = {
        "participant": pid,
        "phase_selection": phases,
        "requested_phase": phase.lower(),
        "limit": args.limit,
        "step": args.step,
        "calibration_table": calibration,
        "calibration_ref_radius_px": args.calibration_ref_radius_px,
        "calibration_verified": bool(args.calibration_verified),
        "center": center,
        "raw_lens_radius_px": raw_lens_radius,
        "rectified_radius_px": rectified_radius,
        "square_crop": not args.no_square_crop,
        "rings": args.svf_rings,
        "model_sha": model_sha256,
    }
    run_config["include_ceiling"] = True
    run_config["view_factors"] = ["SVF:sky", "TVF:tree+plant+grass", "BVF:building+wall"]
    run_sig = hashlib.sha256(json.dumps(run_config, sort_keys=True).encode()).hexdigest()

    out_participant_dir = args.output_root / pid
    out_participant_dir.mkdir(parents=True, exist_ok=True)
    if phase.lower() == "all" and args.limit is None and args.step == 1:
        suffix = ""
    else:
        phase_tag = "_".join(re.sub(r"[^A-Za-z0-9_-]+", "_", p).strip("_") for p in phases)
        suffix = f"_{phase_tag}"
    out_csv = out_participant_dir / f"{pid}_SVF{suffix}.csv"
    partial_csv = out_participant_dir / f"{pid}_SVF{suffix}_PARTIAL.csv"
    failure_csv = out_participant_dir / f"{pid}_SVF_failures{suffix}.csv"

    load_pixel_lib()
    print("Loading DeepLabV3+ Xception-65 ADE20K...", flush=True)
    seg_engine = semantic_segmentation()
    seg_engine.load_ade20k_model(str(MODEL_PATH))
    model = seg_engine.model2

    class_table = ClassTableWriter(args.output_root)
    records, failures, processed_keys = [], [], set()
    existing_records = {}
    source_csvs = list(out_participant_dir.glob(f"{pid}_SVF*.csv"))
    comparison_dir = OUTPUT_ROOT.parent / "Image classifier sensitivity Xception vs SegFormer B4" / "SVF Output_TVF_BVF" / pid
    if args.output_root.resolve() == OUTPUT_ROOT.resolve() and comparison_dir.is_dir():
        source_csvs.extend(comparison_dir.glob(f"{pid}_SVF*.csv"))
    for source_csv in sorted(source_csvs, key=lambda path: path.stat().st_mtime):
        if "matched_8min" in source_csv.name or "failures" in source_csv.name:
            continue
        try:
            source_rows = normalize_svf_export_frame(pd.read_csv(source_csv))
        except Exception as exc:
            print(f"Ignoring unreadable existing CSV {source_csv.name}: {exc}", flush=True)
            continue
        for row in source_rows.to_dict("records"):
            try:
                same_configuration = (
                    str(row.get("Model_SHA256", "")) == model_sha256
                    and str(row.get("Model", "")) == "DeepLabV3+_Xception65_ADE20K"
                    and str(row.get("Camera", "")) == "CamUp"
                    and str(row.get("Calibration", "")) == calib_src
                    and str(row.get("Calibration_Table", "")) == ",".join(str(x) for x in calibration)
                    and str(row.get("Calibration_QC", "")) == calib_qc
                    and int(float(row.get("SVF_Rings", -1))) == args.svf_rings
                    and math.isclose(float(row.get("Calibration_Ref_Radius_px", np.nan)), args.calibration_ref_radius_px)
                    and math.isclose(float(row.get("Raw_Lens_Radius_px", np.nan)), raw_lens_radius)
                    and math.isclose(float(row.get("Rectified_Lens_Radius_px", np.nan)), rectified_radius)
                    and str(row.get("Square_Hemisphere_Crop", "")).lower() in {"true", "1"}
                )
            except (TypeError, ValueError):
                same_configuration = False
            if same_configuration:
                existing_records[(str(row.get("PhaseID", "")), str(row.get("Filename", "")))] = row

    partial_rows_on_disk = 0
    if partial_csv.is_file():
        df_exist = normalize_svf_export_frame(pd.read_csv(partial_csv))
        required_resume_cols = {"PhaseID", "Filename", "Run_Config_Signature"}
        missing_resume_cols = required_resume_cols - set(df_exist.columns)
        if missing_resume_cols:
            raise RuntimeError(
                f"Cannot safely resume {partial_csv.name}; missing columns: "
                f"{sorted(missing_resume_cols)}"
            )
        existing_sigs = set(df_exist["Run_Config_Signature"].dropna().astype(str))
        if len(existing_sigs) != 1 or next(iter(existing_sigs), None) != run_sig:
            raise RuntimeError(
                f"Cannot safely resume {partial_csv.name}; existing Run_Config_Signature "
                "does not match the current configuration. Remove/rename the partial file first."
            )
        for row in df_exist.to_dict("records"):
            existing_records[(str(row["PhaseID"]), str(row["Filename"]))] = row
        partial_rows_on_disk = len(df_exist)
        print(f"Resuming from {partial_rows_on_disk} rows saved in {partial_csv.name}.", flush=True)

    def flush_partial_checkpoint() -> None:
        nonlocal partial_rows_on_disk
        if len(records) <= partial_rows_on_disk:
            return
        append_partial_csv_rows(partial_csv, records[partial_rows_on_disk:])
        partial_rows_on_disk = len(records)

    total_images_count = 0
    expected_keys = set()
    with tempfile.TemporaryDirectory(prefix="svf_rectified_") as tmpdir:
        tmpdir = Path(tmpdir)
        for ph in phases:
            ph_ucm = UCM_RAW_ROOT / pid / ph / "ucm"
            if not ph_ucm.exists():
                continue
            session_dir = resolve_session_dir(ph_ucm)
            cam_dir = session_dir / "CamUp"
            out_cam_dir = out_participant_dir / ph / "CamUp_SVF"
            out_cam_dir.mkdir(parents=True, exist_ok=True)
            if args.save_sky_masks:
                (out_participant_dir / ph / "CamUp_sky_masks").mkdir(parents=True, exist_ok=True)

            df_tel = load_ucm_telemetry(session_dir)
            all_images = sorted(p for p in cam_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
            images = select_images_to_key_phase(all_images, key_df, pid, ph)[::args.step]
            if args.limit:
                images = images[:args.limit]
            if not images:
                continue

            expected_keys.update((ph, p.name) for p in images)
            pending = []
            for image_path in images:
                key = (ph, image_path.name)
                overlay_path = out_cam_dir / f"{image_path.stem}_svf.jpg"
                row = existing_records.get(key)
                if row is not None and (pid, ph, image_path.name) in class_table.saved:
                    row["Segmented_Image_Path"] = str(overlay_path)
                    row["Run_Config_Signature"] = run_sig
                    records.append(row)
                    processed_keys.add(key)
                else:
                    pending.append(image_path)
            total_images_count += len(images)
            print(f"\nPhase {ph}: {len(images)} total frames | {len(images) - len(pending)} complete/current | {len(pending)} to process.", flush=True)

            batch_size = max(1, args.batch_size)
            phase_t0 = time.monotonic()
            for b_start in range(0, len(pending), batch_size):
                batch_paths = pending[b_start : b_start + batch_size]
                batch_tensors, batch_items = [], []

                for img_path in batch_paths:
                    fn, ts = img_path.name, parse_timestamp_from_filename(img_path.name)
                    try:
                        raw_bgr = cv2.imread(str(img_path))
                        if raw_bgr is None:
                            raise ValueError("Image read failed")
                        rect_bgr, f_mask = rectifier.rectify_image(raw_bgr)

                        if not args.no_square_crop:
                            rx, cx_i, cy_i = int(round(rectified_radius)), int(round(center[0])), int(round(center[1]))
                            x0, x1, y0, y1 = max(0, cx_i - rx), min(w0, cx_i + rx), max(0, cy_i - rx), min(h0, cy_i + rx)
                            crop_bgr, eval_mask = rect_bgr[y0:y1, x0:x1], f_mask[y0:y1, x0:x1]
                            eval_center, eval_radius, crop_box = (cx_i - x0, cy_i - y0), rx, (x0, x1, y0, y1)
                        else:
                            crop_bgr, eval_mask = rect_bgr, f_mask
                            eval_center, eval_radius, crop_box = (int(round(center[0])), int(round(center[1]))), int(round(rectified_radius)), (0, w0, 0, h0)

                        circle_area = cv2.countNonZero(eval_mask)
                        if circle_area <= 0:
                            raise RuntimeError("Zero-area hemisphere mask")

                        # DeepLabV3+ 512x512 tensor prep
                        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
                        w_in, h_in = rgb.shape[:2]
                        ratio = 512.0 / max(w_in, h_in)
                        resized = np.array(Image.fromarray(rgb).resize((int(ratio * h_in), int(ratio * w_in))))
                        norm_img = (resized / 127.5) - 1.0
                        pad_x, pad_y = 512 - norm_img.shape[0], 512 - norm_img.shape[1]
                        padded = np.pad(norm_img, ((0, pad_x), (0, pad_y), (0, 0)), mode="constant")

                        batch_tensors.append(padded)
                        batch_items.append({"path": img_path, "fn": fn, "ts": ts, "crop": crop_bgr, "mask": eval_mask,
                                            "center": eval_center, "radius": eval_radius, "area": circle_area,
                                            "pad_x": pad_x, "pad_y": pad_y, "crop_box": crop_box})
                    except Exception as e:
                        failures.append({"ParticipantID": pid, "PhaseID": ph, "Filename": fn, "Timestamp": ts, "Error": str(e)})

                if not batch_items:
                    continue

                preds = model.predict(np.stack(batch_tensors, axis=0), batch_size=len(batch_tensors), verbose=0)

                for item_idx, itm in enumerate(batch_items):
                    labels = np.argmax(preds[item_idx], axis=-1)
                    if itm["pad_x"] > 0: labels = labels[:-itm["pad_x"]]
                    if itm["pad_y"] > 0: labels = labels[:, :-itm["pad_y"]]

                    _, objs = ade20k_map_color_mask(labels, extract_segmented_objects=True)
                    cls_masks = union_class_masks(objs, itm["crop"].shape[:2], itm["mask"])
                    pcts = {k: (cv2.countNonZero(m) / float(itm["area"])) * 100.0 for k, m in cls_masks.items()}
                    # Retain original label counts, not percentages reconstructed from JPEGs.
                    name_map = create_ade20k_label_namemap()
                    class_record = {"ParticipantID": pid, "PhaseID": ph, "Filename": itm["fn"],
                                    "Decode_Error": "", "Class_Data_Source": "direct_model_labels"}
                    class_record.update({name_map[i]: pcts.get(name_map[i], 0.0) for i in range(151)})
                    class_table.save(class_record)

                    sky_mask = cv2.bitwise_and(cls_masks.get("sky", np.zeros(itm["crop"].shape[:2], dtype=np.uint8)), itm["mask"])
                    svf = compute_svf(sky_mask, args.svf_rings, itm["center"], itm["radius"])

                    # View-factor masks follow the seven-class ADE20K mapping; walls are built surfaces, ceiling stays separate.
                    vegetation_mask = cv2.bitwise_or(
                        cv2.bitwise_or(cls_masks.get("tree", np.zeros_like(sky_mask)),
                                       cls_masks.get("plant", np.zeros_like(sky_mask))),
                        cls_masks.get("grass", np.zeros_like(sky_mask)),
                    )
                    vegetation_mask = cv2.bitwise_and(vegetation_mask, itm["mask"])
                    tvf = compute_svf(vegetation_mask, args.svf_rings, itm["center"], itm["radius"])
                    built_mask = cv2.bitwise_or(
                        cls_masks.get("building", np.zeros_like(sky_mask)),
                        cls_masks.get("wall", np.zeros_like(sky_mask)),
                    )
                    built_mask = cv2.bitwise_and(built_mask, itm["mask"])
                    bvf = compute_svf(built_mask, args.svf_rings, itm["center"], itm["radius"])

                    # Gong et al. (2018, doi:10.1016/j.buildenv.2018.02.042) use annular factors; ADE20K plant joins vegetation, wall joins buildings, ceiling stays separate.
                    out_img = out_cam_dir / f"{itm['path'].stem}_svf.jpg"
                    out_img.parent.mkdir(parents=True, exist_ok=True)
                    title = (pcts.get("sky", 0), sum(pcts.get(k, 0) for k in ("tree", "plant", "grass")),
                             pcts.get("building", 0) + pcts.get("wall", 0), svf, tvf, bvf)
                    save_overlay(out_img, itm["crop"], labels, itm["mask"], itm["center"], itm["radius"], title)

                    if args.save_sky_masks:
                        cv2.imwrite(str(out_participant_dir / ph / "CamUp_sky_masks" / f"{itm['path'].stem}_sky.png"), sky_mask)

                    # Telemetry matching
                    tel_data = default_telemetry()
                    matched = False
                    if not df_tel.empty and itm["ts"]:
                        m = df_tel[df_tel["GPS_time"] == itm["ts"]]
                        if not m.empty:
                            matched = True
                            tel_data = {col: pd.to_numeric(m.iloc[0].get(col), errors="coerce") for col in tel_data}

                    rec = {
                        "ParticipantID": pid, "PhaseID": ph, "Filename": itm["fn"], "Timestamp": itm["ts"],
                        "Camera": "CamUp", "Calibration": calib_src, "Calibration_Table": ",".join(str(x) for x in calibration),
                        "Run_Config_Signature": run_sig, "Model_SHA256": model_sha256, "Calibration_QC": calib_qc,
                        "Calibration_Ref_Radius_px": float(args.calibration_ref_radius_px), "Raw_Lens_Radius_px": float(raw_lens_radius),
                        "Rectified_Lens_Radius_px": float(rectified_radius), "Square_Hemisphere_Crop": not args.no_square_crop,
                        "SVF_Method": "Holmer_1992_Annular", "SVF_Rings": int(args.svf_rings),
                        # CamUp is assumed to remain approximately zenith-oriented; no dynamic pitch/roll correction is applied.
                        "Pitch_Roll_Correction": "None_MSE_Method_Assumed_Zenith",
                        "Sky_%": pcts.get("sky", 0.0), "SVF": round(float(svf), 6),
                        "TVF": round(float(tvf), 6), "BVF": round(float(bvf), 6),
                        "Tree_%": pcts.get("tree", 0.0), "Plant_%": pcts.get("plant", 0.0), "Grass_%": pcts.get("grass", 0.0),
                        "Total_Vegetation_%": pcts.get("tree", 0.0) + pcts.get("plant", 0.0) + pcts.get("grass", 0.0),
                        "Building_%": pcts.get("building", 0.0), "Wall_%": pcts.get("wall", 0.0),
                        "Ceiling_%": pcts.get("ceiling", 0.0),
                        "Total_Builtup_%": pcts.get("building", 0.0) + pcts.get("wall", 0.0),
                        "Model": "DeepLabV3+_Xception65_ADE20K", "Image_Path": str(itm["path"]),
                        "Segmented_Image_Path": str(out_img),
                        "telemetry_matched": matched,
                        **tel_data,
                    }
                    records.append(rec)
                    processed_keys.add((ph, itm["fn"]))

                if records:
                    flush_partial_checkpoint()
                    elapsed = time.monotonic() - phase_t0
                    attempted = min(b_start + batch_size, len(pending))
                    rate = attempted / elapsed if elapsed > 0 else 0.0
                    eta = (len(pending) - attempted) / rate / 60 if rate > 0 else float('inf')
                    print(f"Progress {pid}/{ph}: {attempted}/{len(pending)} pending frames attempted | "
                          f"{rate:.3f} frames/s | phase ETA {eta:.1f} min | "
                          f"saved rows {len(records)} | failures {len(failures)}", flush=True)

    df_out = normalize_svf_export_frame(pd.DataFrame(records))
    if args.output_root.resolve() == OUTPUT_ROOT.resolve() and not df_out.empty:
        pending = df_out["PhaseID"].isin(SVF_PENDING_PHASES)
        fields = [c for c in SVF_MEASUREMENT_FIELDS if c in df_out]
        df_out.loc[pending, fields] = np.nan
    if not expected_keys:
        raise RuntimeError("No CamUp images were selected for this run.")

    successful_keys = {
        (str(r.get("PhaseID")), str(r.get("Filename")))
        for r in records
    }
    duplicate_success_rows = len(records) - len(successful_keys)
    missing_success = sorted(expected_keys - successful_keys)
    unexpected_success = sorted(successful_keys - expected_keys)
    complete_run = (
        not failures
        and duplicate_success_rows == 0
        and not missing_success
        and not unexpected_success
        and successful_keys == expected_keys
    )

    if failures:
        pd.DataFrame(failures).to_csv(failure_csv, index=False)
    df_out.to_csv(partial_csv, index=False)

    matched_csv = out_participant_dir / f"{out_csv.stem}_matched_8min.csv"
    matched_partial_csv = out_participant_dir / f"{out_csv.stem}_matched_8min_PARTIAL.csv"
    merged_matched_csv = None
    should_promote = complete_run or (args.allow_partial and not df_out.empty)
    if should_promote:
        matched_df = trim_svf_to_matched_8min(df_out, key_df, pid, phases)
        if matched_df.empty:
            raise RuntimeError(f"No SVF images fall inside the matched 8-minute windows for {pid}")
        matched_df.to_csv(matched_partial_csv, index=False)
        os.replace(partial_csv, out_csv)
        os.replace(matched_partial_csv, matched_csv)
        if complete_run and failure_csv.exists():
            failure_csv.unlink()
        if suffix == "" and args.output_root.resolve() == OUTPUT_ROOT.resolve():
            merged_matched_csv = write_merged_matched_svf()
            write_merged_svf()
            if merged_matched_csv is not None:
                write_svf_class_exports()
        result_path = out_csv
    else:
        result_path = partial_csv

    print("\n" + "=" * 80)
    print(
        f"SVF Execution Complete | Expected: {len(expected_keys)} | "
        f"Success rows: {len(records)} | Failures: {len(failures)}"
    )
    print(
        f"Completeness | duplicate success rows: {duplicate_success_rows} | "
        f"missing: {len(missing_success)} | unexpected: {len(unexpected_success)}"
    )
    print(f"Output File: {result_path}")
    if should_promote:
        print(f"Matched 8-min output: {matched_csv} ({len(matched_df)} rows)")
        if merged_matched_csv:
            print(f"Merged matched output: {merged_matched_csv}")
    print("=" * 80)

    if not should_promote:
        raise RuntimeError(
            "SVF run is incomplete and was NOT promoted to canonical/pilot final output. "
            f"failures={len(failures)}, duplicate_rows={duplicate_success_rows}, "
            f"missing={len(missing_success)}, unexpected={len(unexpected_success)}. "
            f"Inspect {partial_csv} and {failure_csv if failures else 'the completeness summary'}."
        )


if __name__ == "__main__":
    main()
