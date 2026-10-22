"""SVF_10sec.py: key-anchored 10 s means from merged_SVF*.csv (compute_svf_camup windowing).

Retained classes (7): Sky_%, Tree_%, Plant_%, Grass_%, Building_%, Wall_%, Ceiling_%.
Derived totals (0–100 % of fisheye): Total_Vegetation_% = Tree_+Plant_+Grass_;
Total_Builtup_% = Building_+Wall_.
Gong annular view factors [0,1] (same Holmer compute_svf as SVF): TVF, BVF —
TVF from Tree∪Plant∪Grass; BVF from Building∪Wall (Ceiling not in BVF).
Per frame: exclude from window mean if sum of those seven < 50%. Windows with no eligible frames retain blank measurements and a zero frame count.
Writes merged_SVF_10sec_avg.csv and merged_SVF_matched_8min_10sec_avg.csv under analysis_output/SVF Output/.
"""
import os
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "analysis_output" / "SVF Output"
KEY_CSV = ROOT.parent / "Paper3_Github" / "output" / "key.csv"
MATCHED = 480.0
F = ["Sky_%", "Tree_%", "Plant_%", "Grass_%", "Building_%", "Wall_%", "Ceiling_%"]
MIN = 50.0
MF = [
    "Sky_%", "Tree_%", "Plant_%", "Grass_%", "Total_Vegetation_%",
    "Building_%", "Wall_%", "Ceiling_%", "Total_Builtup_%",
    "SVF", "TVF", "BVF",
]


def ensure_svf_totals(df: pd.DataFrame) -> pd.DataFrame:
    """Validate frame measurements and derive totals from their component classes."""
    out = df.copy()
    required = F + ["SVF", "TVF", "BVF"]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise ValueError(f"Missing required SVF measurements: {', '.join(missing)}")
    out[required] = out[required].apply(pd.to_numeric, errors="raise")
    invalid = ~np.isfinite(out[required].to_numpy(dtype=float))
    if invalid.any():
        columns = [c for c, bad in zip(required, invalid.any(axis=0)) if bad]
        raise ValueError(f"Missing or non-finite SVF measurements: {', '.join(columns)}")
    if "Total_Vegetation_%" not in out.columns and "Combined_Vegetation_%" in out.columns:
        out = out.rename(columns={"Combined_Vegetation_%": "Total_Vegetation_%"})
    elif "Combined_Vegetation_%" in out.columns and "Total_Vegetation_%" in out.columns:
        out = out.drop(columns=["Combined_Vegetation_%"])
    for total, components in {
        "Total_Vegetation_%": ["Tree_%", "Plant_%", "Grass_%"],
        "Total_Builtup_%": ["Building_%", "Wall_%"],
    }.items():
        expected = out[components].sum(axis=1)
        out[total] = expected
    return out


def load_key():
    df = pd.read_csv(KEY_CSV, dtype=str)
    df["Participant_ID"] = df["Participant_ID"].map(
        lambda x: str(x).strip().upper() if str(x).strip().upper().startswith("P") else f"P{int(float(x))}"
    )
    return df.set_index("Participant_ID")


def windowed(data, key_df, mode):
    w = data.copy()
    w["_ts"] = pd.to_datetime(w["Timestamp"], errors="raise")
    out = []
    for (pid, phase), g in w.groupby(["ParticipantID", "PhaseID"], sort=False):
        row = key_df.loc[pid]
        ok = g["_ts"].dt.strftime("%d-%b").str.lower().eq(str(row["Date"]).lower())
        if not ok.any():
            raise ValueError(f"date mismatch {pid}/{phase}")
        date = g.loc[ok, "_ts"].iloc[0].date()
        start = pd.Timestamp(datetime.combine(date, datetime.strptime(str(row[f"{phase}_start"]), "%H:%M:%S").time()))
        end = start + pd.Timedelta(seconds=MATCHED) if mode == "matched_8min" else pd.Timestamp(
            datetime.combine(date, datetime.strptime(str(row[f"{phase}_end"]), "%H:%M:%S").time())
        )
        if end <= start:
            end += pd.Timedelta(days=1)
        g = g.loc[g["_ts"].ge(start) & g["_ts"].lt(end)].copy()
        g["SVF_Window_Start_s"] = ((g["_ts"] - start).dt.total_seconds() // 10).astype(int) * 10
        out.append(g)
    return pd.concat(out, ignore_index=True)


def run(st, mode):
    d = ensure_svf_totals(pd.read_csv(OUT / f"{st}.csv"))
    ni = len(d)
    w = windowed(d, load_key(), mode)
    nf = len(w)
    g = ["ParticipantID", "PhaseID", "SVF_Window_Start_s"]
    n_win_all = w.groupby(g).ngroups
    windows = w[g].drop_duplicates().set_index(g).index
    w = w.loc[w[F].apply(pd.to_numeric, errors="raise").sum(axis=1).ge(MIN)]
    ex = nf - len(w)
    wi = w.groupby(g).ngroups
    if n_win_all > wi:
        print(f"  warning: {n_win_all - wi} windows retained with blank measurements (no frame with class sum >= {MIN}%)")
    o = w.groupby(g, sort=False)[MF].mean().reindex(windows)
    o["SVF_Frames_In_Window"] = w.groupby(g, sort=False).size().reindex(windows, fill_value=0).astype(int)
    o["SVF_Status"] = np.where(o["SVF_Frames_In_Window"].gt(0), "available", "no_frames_with_coverage_>=50%")
    assert o.loc[o["SVF_Frames_In_Window"].eq(0), MF].isna().all().all()
    o = o.reset_index()
    p = OUT / f"{st}_10sec_avg.csv"
    t = p.with_suffix(".tmp.csv")
    o.to_csv(t, index=False)
    os.replace(t, p)
    print(f"{st}.csv rows_in={ni} phase_frames={nf} frames_excluded={ex} windows_out={wi}/{n_win_all} rows_out={len(o)} -> {p.name}")


if __name__ == "__main__":
    run("merged_SVF", "full")
    run("merged_SVF_matched_8min", "matched_8min")
