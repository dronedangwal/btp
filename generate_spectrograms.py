"""
generate_spectrograms.py
────────────────────────
Batch-process a list of CSI recordings and save sliding-window spectrograms
for each one.

Pipeline:
  1. extract_csi_parallely  – parse .csi file → (ts_arr, csi_arr)
  2. build DataFrame         – raw complex CSI + timestamps
  3. get_amplitude_matrix    – |CSI|, timestamps normalised to seconds from start
  4. save_sliding_window_spectrograms – 2-second window, 1-second stride → PNGs

Usage
─────
Edit the FILENAMES list (and CONFIG block if needed), then run:

    python generate_spectrograms.py

Or import and call process_all() from another script / notebook.
"""

import os
import sys
import time
import traceback

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")           # headless – no display required
import matplotlib.pyplot as plt
from pathlib import Path
from picoparser import PicoParser

import json


# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG  –  edit these before running
# ══════════════════════════════════════════════════════════════════════════════

# List of recording filenames (relative to CSI_BASE_DIR, without .csi extension)
FILENAMES: list[str] = [
    "5/rx_2_260411_121735",
]

CSI_BASE_DIR  = Path("/home/drone/BTP/collected_data")   # root of .csi files
SPECTROGRAM_DIR = Path("model_data")                      # root of PNG output

# Extraction params
TX_IDX      = 0
RX_IDX      = 0
CSI_IDX     = 0
MIN_LEN     = 998
NUM_WORKERS = 4

# Spectrogram params
WINDOW_SEC  = 2.0
STRIDE_SEC  = 1.0
IMG_SIZE    = (224, 224)
CMAP        = "jet"

# ══════════════════════════════════════════════════════════════════════════════


# ── Step 1: CSI extraction ────────────────────────────────────────────────────

def extract_csi_parallely(
    csi_path: Path,
    ts_start: int,
    ts_end: int,
    tx_idx: int   = TX_IDX,
    rx_idx: int   = RX_IDX,
    csi_idx: int  = CSI_IDX,
    min_len: int  = MIN_LEN,
    num_workers: int = NUM_WORKERS,
) -> tuple[np.ndarray, np.ndarray]:
    """Parse a .csi file and return (ts_arr [ns, int64], csi_arr [complex64])."""
    with PicoParser(csi_path, num_workers) as parser:
        i = 0
        ts_list, csi_list = [], []

        for fr in parser.getFrames():
            try:
                ts  = fr.rxSBasic.systemTime
                csi = fr.csi.csi
                csi = np.array([sub_csi[tx_idx][rx_idx][csi_idx] for sub_csi in csi])
            except Exception:
                continue

            if ts is None:
                continue
            if (int(ts) < int(ts_start)):
                continue
            if (int(ts) > int(ts_end)):
                break
            if csi.ndim == 0:
                continue
            if csi.shape[0] < min_len:
                continue

            ts_list.append(int(ts))
            csi_list.append(csi[:min_len])

            if (i + 1) % 1000 == 0:
                print(f"    ...parsed {i + 1} frames")
            i += 1

        print(f"    {i} frames parsed in total")
        return np.array(ts_list, dtype=np.int64), np.array(csi_list)


# ── Step 2 + 3: DataFrame → amplitude matrix ─────────────────────────────────

def build_dataframe(ts_arr: np.ndarray, csi_arr: np.ndarray) -> pd.DataFrame:
    """Assemble raw complex CSI + timestamps into a DataFrame."""
    df = pd.DataFrame(csi_arr, columns=range(1, csi_arr.shape[1] + 1))
    df.insert(0, "timestamp", ts_arr)
    return df


def get_amplitude_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert complex CSI DataFrame to amplitude, normalising timestamps
    to seconds relative to the recording start.
    """
    amp_values = np.abs(df.drop(columns=["timestamp"]).to_numpy())
    ts = df["timestamp"].to_numpy()
    ts = (ts - ts[0]) * 1e-9                       # ns → seconds from start
    amp_matrix = pd.DataFrame(
        np.hstack((ts.reshape(-1, 1), amp_values))
    )
    amp_matrix.columns = ["timestamp"] + list(range(1, amp_values.shape[1] + 1))
    return amp_matrix


# ── Step 4: sliding-window spectrogram export ─────────────────────────────────

def save_sliding_window_spectrograms(
    amp_df: pd.DataFrame,
    output_dir: str | Path,
    window_sec: float = WINDOW_SEC,
    stride_sec: float = STRIDE_SEC,
    img_size: tuple[int, int] = IMG_SIZE,
    cmap: str = CMAP,
    video_name: str = "video",
) -> list[dict]:
    """
    Slide a window of `window_sec` over `amp_df` in steps of `stride_sec`,
    saving one axis-free PNG per step.

    Window k covers  [k*stride - window_sec,  k*stride)  seconds.
    The 'second' index in returned metadata is 1-based and aligns with the
    'second' column produced by parse_via_annotations.py.

    Returns a list of dicts: {filename, t_start, t_end, second}.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ts  = amp_df["timestamp"].to_numpy()
    amp = amp_df.drop(columns=["timestamp"]).to_numpy()   # (frames, subcarriers)

    total_duration = float(ts[-1])
    dpi   = 72
    fig_w = img_size[0] / dpi
    fig_h = img_size[1] / dpi

    # Global colour scale — consistent across all windows of this recording
    vmin, vmax = float(amp.min()), float(amp.max())

    metadata: list[dict] = []
    second = 2
    t_end  = window_sec

    while t_end <= total_duration + 1e-9:
        t_start = t_end - window_sec
        mask       = (ts >= t_start) & (ts < t_end)
        window_amp = amp[mask]                          # (frames_in_window, subcarriers)

        if window_amp.shape[0] == 0:
            print(f"    [skip] second={second:04d}  no frames in "
                  f"[{t_start:.2f}, {t_end:.2f})")
            second += 1
            t_end  += stride_sec
            continue

        # Render – fill the entire figure canvas, no whitespace
        fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
        ax  = fig.add_axes([0, 0, 1, 1])
        ax.imshow(
            window_amp.T,                               # (subcarriers, frames)
            aspect="auto",
            origin="lower",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
        )
        ax.axis("off")

        fname    = f"{second}.png"
        out_path = output_dir / fname
        fig.savefig(out_path, dpi=dpi, bbox_inches=None, pad_inches=0)
        plt.close(fig)

        metadata.append({
            "filename": fname,
            "t_start" : round(t_start, 6),
            "t_end"   : round(t_end,   6),
            "second"  : second,
        })

        if second % 10 == 0:
            print(f"    second={second:04d}  [{t_start:.2f}s, {t_end:.2f}s)  "
                  f"frames={window_amp.shape[0]}")

        second += 1
        t_end  += stride_sec

    return metadata


# ── Per-file pipeline ─────────────────────────────────────────────────────────

def process_file(filename: str) -> bool:
    """
    Run the full pipeline for a single recording.

    Parameters
    ----------
    filename : str
        Path relative to CSI_BASE_DIR, without the .csi extension.
        e.g.  "test2/rx_2_260321_172246"

    Returns True on success, False on failure.
    """
    csi_path   = CSI_BASE_DIR / f"{filename}.csi"
    out_dir    = SPECTROGRAM_DIR / filename.split('/')[0]
    video_name = Path(filename).name
    data_dir = csi_path.parent
    json_files = []
    for file_path in data_dir.glob("*.json"):
        json_files.append(file_path)

    print(f"\n{'─'*60}")
    print(f"  FILE  : {filename}")
    print(f"  INPUT : {csi_path}")
    print(f"  OUTPUT: {out_dir}")
    print(f"{'─'*60}")

    # ── Validate input ────────────────────────────────────────────────────────
    if not csi_path.exists():
        print(f"  [ERROR] .csi file not found: {csi_path}")
        return False
    
    if len(json_files) == 0:
        print(f"  [ERROR] no .json files found")
        return False

    # ── Skip if already processed ─────────────────────────────────────────────
    meta_path = out_dir / "metadata.csv"
    if meta_path.exists():
        existing = pd.read_csv(meta_path)
        print(f"  [SKIP]  already processed ({len(existing)} spectrograms found)")
        return True

    t0 = time.perf_counter()

    try:
        # Step 1 – extract
        with open(json_files[0], 'r') as file:
            meta_data = json.load(file)
        ts_start = meta_data["responses"][0]["startUnixEpochNano"]
        ts_end = meta_data["responses"][0]["endUnixEpochNano"]    
        
        print("  [1/4] Extracting CSI frames ...")
        ts_arr, csi_arr = extract_csi_parallely(csi_path, ts_start, ts_end)

        total_time = (ts_arr[-1] - ts_arr[0]) / 1e9
        freq       = len(ts_arr) / total_time
        print(f"        shape={csi_arr.shape}  "
              f"duration={total_time:.2f}s  freq={freq:.1f}Hz")

        # Step 2 – DataFrame
        print("  [2/4] Building DataFrame ...")
        df = build_dataframe(ts_arr, csi_arr)

        # Step 3 – amplitude matrix
        print("  [3/4] Computing amplitude matrix ...")
        amp_matrix = get_amplitude_matrix(df)

        # Step 4 – spectrograms
        print(f"  [4/4] Saving spectrograms to {out_dir} ...")
        metadata = save_sliding_window_spectrograms(
            amp_df     = amp_matrix,
            output_dir = out_dir,
            video_name = video_name,
        )

        # Save metadata CSV alongside the PNGs
        meta_df = pd.DataFrame(metadata)
        meta_df.to_csv(meta_path, index=False)

        elapsed = time.perf_counter() - t0
        print(f"\n  ✓  {len(metadata)} spectrograms saved  "
              f"({elapsed:.1f}s total)")
        return True

    except Exception:
        print(f"\n  [ERROR] processing failed for '{filename}':")
        traceback.print_exc()
        return False


# ── Batch entry point ─────────────────────────────────────────────────────────

def process_all(filenames: list[str] = FILENAMES) -> None:
    """Process every filename in the list and print a final summary."""
    print(f"\n{'═'*60}")
    print(f"  Batch spectrogram generation")
    print(f"  {len(filenames)} file(s) to process")
    print(f"  Window={WINDOW_SEC}s  Stride={STRIDE_SEC}s  "
          f"Size={IMG_SIZE}  Cmap={CMAP}")
    print(f"{'═'*60}")

    results: dict[str, bool] = {}
    for fn in filenames:
        results[fn] = process_file(fn)

    # ── Summary ───────────────────────────────────────────────────────────────
    passed = [f for f, ok in results.items() if ok]
    failed = [f for f, ok in results.items() if not ok]

    print(f"\n{'═'*60}")
    print(f"  DONE  {len(passed)} succeeded  /  {len(failed)} failed")
    if failed:
        print("\n  Failed files:")
        for f in failed:
            print(f"    ✗  {f}")
    print(f"{'═'*60}\n")


if __name__ == "__main__":
    # Optionally accept filenames as CLI arguments:
    #   python generate_spectrograms.py test2/rx_a test2/rx_b
    cli_files = sys.argv[1:]
    process_all(cli_files if cli_files else FILENAMES)
