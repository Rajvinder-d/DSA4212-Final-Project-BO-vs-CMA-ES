"""CALCE source acquisition and auditable preparation; no simulated fallback."""
import hashlib
import io
import json
import subprocess
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from bo_vs_es import ROOT

BASE_URL = "https://web.calce.umd.edu/batteries/data/"
ARCHIVES = {
    "dst": "SP2_25C_DST.zip",
    "fuds": "SP2_25C_FUDS.zip",
    "ocv": "SP3_25C_LC_OCV_11_16_2015.zip",
}
COLUMNS = {
    "Data_Point": "record", "Test_Time(s)": "time_s", "Step_Index": "step",
    "Cycle_Index": "cycle", "Current(A)": "current_charge_a", "Voltage(V)": "voltage_v",
    "Charge_Capacity(Ah)": "charge_ah", "Discharge_Capacity(Ah)": "discharge_ah",
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def acquire(root=ROOT):
    """Download only missing archives, preserving verified TLS and original bytes."""
    root = Path(root)
    folder = root / "data/raw"
    folder.mkdir(parents=True, exist_ok=True)
    records = []
    expected_path = root / "configs/protocols/data_sources.json"
    expected = json.loads(expected_path.read_text()) if expected_path.exists() else {}
    for name in ARCHIVES.values():
        path = folder / name
        if not path.exists():
            temporary = path.with_suffix(".zip.part")
            try:
                try:
                    with urllib.request.urlopen(BASE_URL + name, timeout=120) as response:
                        temporary.write_bytes(response.read())
                except (OSError, urllib.error.URLError):
                    # System curl can use a different, valid platform CA store.
                    subprocess.run(["curl", "--fail", "--location", "--silent", "--show-error",
                                    "--max-time", "120", BASE_URL + name, "-o", str(temporary)], check=True)
                with zipfile.ZipFile(temporary) as archive:
                    if archive.testzip() is not None:
                        raise ValueError(f"Corrupt archive: {name}")
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
        digest = sha256(path)
        if name in expected and expected[name] != digest:
            raise ValueError(f"Checksum mismatch: {name}; do not mix source revisions")
        records.append({"file": name, "url": BASE_URL + name, "sha256": digest,
                        "bytes": path.stat().st_size})
    return records


def read_workbook(archive_path, filename_contains):
    """Read OOXML bytes even when CALCE uses an .xls extension."""
    with zipfile.ZipFile(archive_path) as archive:
        names = [n for n in archive.namelist() if filename_contains in n and not n.startswith("__MACOSX")]
        if len(names) != 1:
            raise ValueError(f"Expected one workbook matching {filename_contains!r}: {names}")
        data = archive.read(names[0])
    return parse_workbook(data, names[0])


def parse_workbook(data, name="memory"):
    if not data.startswith(b"PK"):
        raise ValueError("Expected an OOXML workbook; refusing to guess a legacy Excel schema")
    book = pd.ExcelFile(io.BytesIO(data), engine="openpyxl")
    channels = [s for s in book.sheet_names if s.startswith("Channel")]
    if not channels:
        raise ValueError("No CALCE channel sheets")
    tables = [pd.read_excel(book, sheet_name=s) for s in channels]
    for frame in tables:
        missing = set(COLUMNS) - set(frame)
        if missing:
            raise ValueError(f"Missing measured columns: {missing}")
    frame = pd.concat([t[list(COLUMNS)].rename(columns=COLUMNS) for t in tables], ignore_index=True)
    if not np.isfinite(frame.to_numpy(dtype=float)).all():
        raise ValueError("Missing/nonfinite measurements; preprocessing must be revised explicitly")
    before = len(frame)
    duplicate_records = frame[frame.duplicated("record", keep=False)]
    if not duplicate_records.empty and (duplicate_records.groupby("record").nunique() > 1).any().any():
        raise ValueError("Conflicting measurements for the same record ID")
    frame = frame.drop_duplicates("record").sort_values("record")
    if np.any(np.diff(frame.time_s) < 0):
        raise ValueError("Test clock reset; cannot concatenate without an explicit policy")
    repeated_times = int(frame.duplicated("time_s").sum())
    frame = frame.drop_duplicates("time_s", keep="last").reset_index(drop=True)
    if np.any(np.diff(frame.time_s) <= 0):
        raise ValueError("Non-increasing timestamps")
    audit = {"workbook": name, "sheets": channels, "input_rows": before, "output_rows": len(frame),
             "overlapping_records_removed": before - len(frame) - repeated_times,
             "repeated_timestamps_keep_last": repeated_times}
    return frame, audit


def dynamic_trace(frame, initial_soc, duration_s=3600):
    """Step 7 starts the dynamic profile; retain later rests and one preceding rest row."""
    dynamic = np.flatnonzero(frame.step.to_numpy() == 7)
    if not len(dynamic) or dynamic[0] == 0:
        raise ValueError("Missing dynamic profile or preparatory rest")
    start = int(dynamic[0])
    previous = frame.iloc[start - 1]
    if int(previous.step) != 6 or abs(previous.current_charge_a) > 0.01:
        raise ValueError("Dynamic profile does not follow the expected step-6 rest")
    trace = frame.iloc[start - 1:].copy()
    trace["time_s"] -= previous.time_s
    trace = trace[trace.time_s <= duration_s].copy().reset_index(drop=True)
    if trace.time_s.iloc[-1] < duration_s - 15:
        raise ValueError("Trace shorter than the declared window")
    trace["current_a"] = -trace.current_charge_a
    dt = np.diff(trace.time_s.to_numpy(), prepend=0)
    previous_current = np.r_[0, trace.current_a.to_numpy()[:-1]]
    trace["soc"] = initial_soc - np.cumsum(previous_current * dt) / (3600 * 2.0)
    if not trace.soc.between(0, 1).all():
        raise ValueError("SOC leaves [0,1]; no clipping or OCV extrapolation is permitted")
    return trace


def ocv_curve(frame):
    """Normalise each low-current branch by its own measured capacity span; average voltages."""
    grid = np.linspace(0, 1, 1001)
    curves, spans = [], {}
    for step, capacity, discharge in [(6, "discharge_ah", True), (8, "charge_ah", False)]:
        branch = frame[frame.step == step].copy()
        if len(branch) < 100 or not branch.current_charge_a.abs().between(0.08, 0.12).all():
            raise ValueError("Unexpected low-current OCV branch")
        span = float(branch[capacity].max() - branch[capacity].min())
        if not 1.5 < span < 2.5:
            raise ValueError("Unexpected OCV capacity span")
        soc = (branch[capacity] - branch[capacity].min()) / span
        if discharge:
            soc = 1 - soc
        points = pd.DataFrame({"soc": soc, "v": branch.voltage_v}).groupby("soc", sort=True).v.mean()
        curves.append(np.interp(grid, points.index, points.values))
        spans[capacity] = span
    return pd.DataFrame({"soc": grid, "discharge_v": curves[0], "charge_v": curves[1],
                         "ocv_v": np.mean(curves, axis=0)}), spans


def prepare(root=ROOT):
    """Prepare measured traces and an immutable source/preprocessing audit."""
    root = Path(root)
    sources = acquire(root)
    folder = root / "data/processed"
    folder.mkdir(parents=True, exist_ok=True)
    audits, raw, traces = {}, {}, {}
    for label, archive, match, soc in [
        ("dst80", "dst", "80SOC", .8), ("fuds80", "fuds", "80SOC", .8),
        ("fuds50", "fuds", "50SOC", .5),
    ]:
        raw[label], audits[label] = read_workbook(root / "data/raw" / ARCHIVES[archive], match)
        traces[label] = dynamic_trace(raw[label], soc)
        traces[label].to_csv(folder / f"{label}.csv", index=False)
    raw["ocv"], audits["ocv"] = read_workbook(root / "data/raw" / ARCHIVES["ocv"], "OCV")
    curve, spans = ocv_curve(raw["ocv"])
    curve.to_csv(folder / "ocv.csv", index=False)
    rest = raw["dst80"][raw["dst80"].step == 6]
    rest = rest[rest.time_s >= rest.time_s.max() - 60]
    offset = float(rest.voltage_v.mean() - np.interp(.8, curve.soc, curve.ocv_v))
    audit = {"prepared_utc": datetime.now(timezone.utc).isoformat(), "sources": sources,
             "workbooks": audits, "ocv_cell": "SP20-3", "dynamic_cell": "SP20-2",
             "ocv_branch_capacity_ah": spans, "calibration_rest_offset_v": offset,
             "nominal_capacity_ah": 2.0, "window_s": 3600,
             "interval_current": "left endpoint; discharge positive",
             "output_sha256": {p.name: sha256(p) for p in sorted(folder.glob("*.csv"))}}
    (folder / "manifest.json").write_text(json.dumps(audit, indent=2))
    return raw, traces, curve, audit
