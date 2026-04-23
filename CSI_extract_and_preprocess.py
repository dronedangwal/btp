"""
CSI_extract_and_preprocess.py
-----------------------------
Stage 1 of the CSI-Doppler pipeline.

Reads raw PicoScenes .csi files, extracts CSI frames within a timestamp
window (read from the sibling .json file), applies preprocessing, and saves
the results to disk as .npz files.

Input directory structure expected:
    input_dir/
        subdir_A/
            something.csi
            something.json      ← contains startUnixEpochNano / endUnixEpochNano
            (other files ignored)
        subdir_B/
            ...

Output layout (one .npz per subdir, mirroring the subdir name):
    output_dir/
        subdir_A.npz        # contains: ts, csi
        subdir_B.npz
        ...
            ts  : int64 (T,)        – system timestamps in nanoseconds
            csi : complex64 (T, S)  – preprocessed CSI matrix

Usage
-----
    python CSI_extract_and_preprocess.py \
        --input_dir  /path/to/raw_csi   \
        --output_dir /path/to/preprocessed

Optional flags (all have defaults):
    --tx_idx      0
    --rx_idx      0
    --csi_idx     0
    --min_len     998
    --num_workers 4
"""

import argparse
import json
import numpy as np
from pathlib import Path
from typing import Tuple
from picoparser import PicoParser          # pip install picoparser
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Timestamp window loader
# ---------------------------------------------------------------------------

def load_exam_window_ns(json_path: Path) -> Tuple[int, int]:
    """Read ts_start and ts_end (nanoseconds) from the sidecar JSON file."""
    with open(json_path) as f:
        data = json.load(f)
    responses = data.get("responses", [])
    if not responses:
        raise ValueError(f"'responses' is empty in {json_path}")
    start_ns = int(responses[0]["startUnixEpochNano"])
    end_ns   = int(responses[-1]["endUnixEpochNano"])
    if end_ns <= start_ns:
        raise ValueError(f"end_ns <= start_ns in {json_path}")
    return start_ns, end_ns


# ---------------------------------------------------------------------------
# CSI extraction 
# ---------------------------------------------------------------------------

def extract_csi_parallely(
    csi_path,
    ts_start,
    ts_end,
    tx_idx=0,
    rx_idx=0,
    csi_idx=0,
    min_len=998,
    num_workers=4,
):
    """Parse a .csi file and return (ts_arr [int64], csi_arr [complex64])."""
    with PicoParser(csi_path, num_workers) as parser:
        i = 0
        ts_list, csi_list = [], []

        for fr in tqdm(parser.getFrames()):
            try:
                ts  = fr.rxSBasic.systemTime
                csi = fr.csi.csi
                csi = np.array(
                    [sub_csi[tx_idx][rx_idx][csi_idx] for sub_csi in csi]
                )
            except Exception:
                continue

            if ts is None:
                continue
            if int(ts) < int(ts_start):
                continue
            if int(ts) > int(ts_end):
                break
            if csi.ndim == 0:
                continue
            if csi.shape[0] < min_len:
                continue

            ts_list.append(int(ts))
            csi_list.append(csi[:min_len])

            i += 1

        print(f"    {i} frames parsed in total")
        return (
            np.array(ts_list, dtype=np.int64),
            np.array(csi_list, dtype=np.complex64),
        )


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def preprocess_csi(ts_arr, csi_arr):
    """
    Preprocess raw CSI frames.

    PLACEHOLDER: Do nothing

    Parameters
    ----------
    ts_arr  : int64 ndarray, shape (T,)       – timestamps [ns]
    csi_arr : complex64 ndarray, shape (T, S) – raw CSI

    Returns
    -------
    ts_proc  : int64 ndarray, shape (T',)
    csi_proc : complex64 ndarray, shape (T', S)
    """
    # --- PLACEHOLDER START ---------------------------------------------------
    # Do nothing

    return ts_arr, csi_arr


# ---------------------------------------------------------------------------
# Subdir discovery helpers
# ---------------------------------------------------------------------------

def find_csi_and_json(subdir: Path) -> Tuple[Path, Path]:
    """
    Return the single .csi and single .json file inside a subdir.
    Raises FileNotFoundError / ValueError if files are missing or ambiguous.
    """
    csi_files  = list(subdir.glob("*.csi"))
    json_files = list(subdir.glob("*.json"))

    if not csi_files:
        raise FileNotFoundError(f"No .csi file found in {subdir}")
    if not json_files:
        raise FileNotFoundError(f"No .json file found in {subdir}")
    if len(csi_files) > 1:
        raise ValueError(f"Multiple .csi files in {subdir}: {csi_files}")
    if len(json_files) > 1:
        raise ValueError(f"Multiple .json files in {subdir}: {json_files}")

    return csi_files[0], json_files[0]


# ---------------------------------------------------------------------------
# Per-subdir pipeline
# ---------------------------------------------------------------------------

def process_subdir(subdir: Path, output_dir: Path, args) -> None:
    """Load JSON window → extract CSI → preprocess → save for one subdir."""
    out_path = output_dir / f"{subdir.name}.npz"

    if out_path.exists():
        print(f"  [SKIP] {out_path.name} already exists.")
        return

    # -- locate files --------------------------------------------------------
    try:
        csi_path, json_path = find_csi_and_json(subdir)
    except (FileNotFoundError, ValueError) as e:
        print(f"  [WARN] {e}. Skipping.")
        return

    # -- read timestamp window from JSON ------------------------------------
    try:
        ts_start, ts_end = load_exam_window_ns(json_path)
    except (KeyError, ValueError) as e:
        print(f"  [WARN] Could not read timestamps from {json_path}: {e}. Skipping.")
        return

    print(f"  Subdir : {subdir.name}")
    print(f"    .csi  : {csi_path.name}")
    print(f"    .json : {json_path.name}")
    print(f"    window: {ts_start} → {ts_end} ns  "
          f"({(ts_end - ts_start) / 1e9:.2f} s)")

    # -- extract -------------------------------------------------------------
    ts_raw, csi_raw = extract_csi_parallely(
        csi_path    = csi_path,
        ts_start    = ts_start,
        ts_end      = ts_end,
        tx_idx      = args.tx_idx,
        rx_idx      = args.rx_idx,
        csi_idx     = args.csi_idx,
        min_len     = args.min_len,
        num_workers = args.num_workers,
    )

    if ts_raw.size == 0:
        print(f"  [WARN] No valid frames extracted. Skipping.")
        return

    # -- preprocess ----------------------------------------------------------
    print(f"  Preprocessing ({ts_raw.shape[0]} frames) ...")
    ts_proc, csi_proc = preprocess_csi(ts_raw, csi_raw)

    # -- save ----------------------------------------------------------------
    np.savez_compressed(out_path, ts=ts_proc, csi=csi_proc)
    print(f"  Saved → {out_path.name}  "
          f"[ts: {ts_proc.shape}, csi: {csi_proc.shape}, dtype: {csi_proc.dtype}]\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Stage 1 – CSI extraction & preprocessing")
    p.add_argument("--input_dir",   required=True,
                   help="Root directory whose immediate subdirs each contain a .csi + .json pair")
    p.add_argument("--output_dir",  required=True,
                   help="Where to write .npz files (one per subdir)")
    p.add_argument("--tx_idx",      type=int, default=0)
    p.add_argument("--rx_idx",      type=int, default=0)
    p.add_argument("--csi_idx",     type=int, default=0)
    p.add_argument("--min_len",     type=int, default=998)
    p.add_argument("--num_workers", type=int, default=4)
    return p.parse_args()


def main():
    args       = parse_args()
    input_dir  = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    subdirs = sorted([d for d in input_dir.iterdir() if d.is_dir()])
    if not subdirs:
        print(f"No subdirectories found in {input_dir}")
        return

    print(f"Found {len(subdirs)} subdir(s) under {input_dir}\n")
    for subdir in subdirs:
        process_subdir(subdir, output_dir, args)

    print("Stage 1 complete.")


if __name__ == "__main__":
    main()
