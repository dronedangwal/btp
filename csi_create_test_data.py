"""
Build SHARP-compatible "complete_antennas" test data from Doppler .npz files.

This script mirrors the output format of `SHARP/Python_code/CSI_doppler_create_dataset_test.py`
for the "complete" split:

    <output_dir>/<subdir>/
      complete_antennas_<activities_tag>/0.txt, 1.txt, ...
      labels_complete_antennas_<activities_tag>.txt
      files_complete_antennas_<activities_tag>.txt
      num_windows_complete_antennas_<activities_tag>.txt

Each sample file stores a pickled float32 ndarray with shape:
    (n_tot, feature_length, window_length)

For Doppler files produced by this repository pipeline, use `n_tot=1`.
"""

from __future__ import annotations

import argparse
import csv
import os
import pickle
import shutil
from pathlib import Path

import numpy as np

SHARP_SUFFIX = ".txt"


def read_labels_csv(path: Path) -> list[tuple[float, str]]:
    rows: list[tuple[float, str]] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("CSV has no header row")
        lower = {h.lower().strip(): h for h in reader.fieldnames}
        if "second" not in lower or "activity" not in lower:
            raise ValueError(
                f"CSV must have columns 'second' and 'activity'; got {reader.fieldnames!r}"
            )
        k_second = lower["second"]
        k_activity = lower["activity"]
        for row in reader:
            sec_raw = row[k_second]
            act_raw = row[k_activity]
            if sec_raw is None or str(sec_raw).strip() == "":
                continue
            sec = float(sec_raw)
            activity = (act_raw or "").strip()
            rows.append((sec, activity))
    return rows


def load_doppler_file(npz_path: Path) -> tuple[np.ndarray, float, int, int]:
    data = np.load(npz_path, allow_pickle=False)
    try:
        doppler_map = np.asarray(data["doppler_map"], dtype=np.float32)
        fs = float(np.asarray(data["fs"]).reshape(()))
        hop = int(np.asarray(data["hop"]).reshape(()))
        win_len = int(np.asarray(data["win_len"]).reshape(()))
    finally:
        data.close()
    return doppler_map, fs, hop, win_len


def stft_row_range(
    second: float,
    window_seconds: float,
    fs: float,
    win_len: int,
    hop: int,
) -> tuple[int, int]:
    rows_per_second = fs / float(hop)
    overlap_term = float(win_len) / float(hop)
    start = max(0.0, (second - window_seconds) * rows_per_second - overlap_term + 1.0)
    end = second * rows_per_second - overlap_term + 1.0
    return int(np.floor(start)), int(np.floor(end))


def create_windows_antennas(
    csi_list: list[np.ndarray],
    labels_list: list[int],
    sample_length: int,
    stride_length: int,
) -> tuple[list[np.ndarray], list[int]]:
    csi_matrix_stride: list[np.ndarray] = []
    labels_stride: list[int] = []
    for i in range(len(labels_list)):
        csi_i = csi_list[i]
        label_i = labels_list[i]
        len_csi = csi_i.shape[2]
        for ii in range(0, len_csi - sample_length + 1, stride_length):
            csi_wind = csi_i[:, :, ii:ii + sample_length, ...]
            csi_matrix_stride.append(csi_wind.astype(np.float32, copy=False))
            labels_stride.append(label_i)
    return csi_matrix_stride, labels_stride


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--doppler_dir", type=Path, required=True, help="Directory containing *_doppler.npz files")
    parser.add_argument("--labels_csv", type=Path, required=True, help="CSV with columns: second, activity")
    parser.add_argument("--output_dir", type=Path, required=True, help="Base directory where SHARP-like outputs are written")
    parser.add_argument("--subdir", type=str, default="dataset_out_test", help="Subdirectory name inside output_dir")
    parser.add_argument("--activities_tag", type=str, default="Still,Scrolling,Flipping,Typing",
                        help="Activity labels in order, comma-separated")
    parser.add_argument("--window_seconds", type=float, required=True,
                        help="Look-back window in seconds for per-label block extraction")
    parser.add_argument("--window_length", type=int, required=True,
                        help="Number of Doppler rows per final sliding window")
    parser.add_argument("--stride_length", type=int, required=True,
                        help="Sliding-window stride over Doppler rows")
    parser.add_argument("--n_tot", type=int, default=1,
                        help="Number of streams*antennas in each sample (pipeline Doppler uses 1)")
    args = parser.parse_args()

    if args.n_tot != 1:
        raise ValueError(
            "This converter currently supports n_tot=1 for pipeline Doppler maps. "
            "Each *_doppler.npz is a single collapsed stream."
        )
    if args.window_length <= 0 or args.stride_length <= 0:
        raise ValueError("window_length and stride_length must be > 0")

    labels_rows = read_labels_csv(args.labels_csv)
    if not labels_rows:
        raise ValueError("No rows found in labels CSV")

    label_names = [x.strip() for x in args.activities_tag.split(",") if x.strip()]
    label_to_id = {name.lower(): idx for idx, name in enumerate(label_names)}
    if not label_to_id:
        raise ValueError("activities_tag must include at least one label")

    exp_dir = args.output_dir / args.subdir
    exp_dir.mkdir(parents=True, exist_ok=True)

    activities_literal = args.activities_tag
    complete_dir = exp_dir / f"complete_antennas_{activities_literal}"
    if complete_dir.exists():
        shutil.rmtree(complete_dir)
    complete_dir.mkdir(parents=True, exist_ok=True)

    doppler_files = sorted(args.doppler_dir.glob("*_doppler.npz"))
    if not doppler_files:
        raise FileNotFoundError(f"No *_doppler.npz files found in {args.doppler_dir}")

    csi_complete: list[np.ndarray] = []
    labels_complete: list[int] = []
    lengths: list[int] = []

    for npz_path in doppler_files:
        doppler_map, fs, hop, win_len = load_doppler_file(npz_path)
        total_rows, _freq_bins = doppler_map.shape

        for second, activity in labels_rows:
            cls_id = label_to_id.get(activity.lower())
            if cls_id is None:
                continue

            start_i, end_i = stft_row_range(second, args.window_seconds, fs, win_len, hop)
            end_i = min(end_i, total_rows)
            if end_i <= start_i or start_i >= total_rows:
                continue

            block = doppler_map[start_i:end_i, :]  # (T, F)
            if block.shape[0] < args.window_length:
                continue

            # SHARP "complete_antennas" expected orientation per sample:
            # (n_tot, feature_length, sample_length). With n_tot=1 => (1, F, T).
            sample = np.transpose(block, (1, 0))[np.newaxis, :, :]
            csi_complete.append(sample.astype(np.float32, copy=False))
            labels_complete.append(cls_id)
            lengths.append(sample.shape[2])

    if not csi_complete:
        raise RuntimeError(
            "No complete samples generated. Check labels.csv, window_seconds, and Doppler file lengths."
        )

    csi_matrices_wind, labels_wind = create_windows_antennas(
        csi_complete,
        labels_complete,
        args.window_length,
        args.stride_length,
    )

    num_windows = np.floor((np.asarray(lengths) - args.window_length) / args.stride_length + 1)
    if len(csi_matrices_wind) != int(np.sum(num_windows)):
        raise RuntimeError(
            f"Window count mismatch: generated={len(csi_matrices_wind)} expected={int(np.sum(num_windows))}"
        )

    names_complete: list[str] = []
    for i, sample in enumerate(csi_matrices_wind):
        file_path = complete_dir / f"{i}{SHARP_SUFFIX}"
        with open(file_path, "wb") as fp:
            pickle.dump(sample, fp)
        names_complete.append(str(file_path.resolve()))

    labels_path = exp_dir / f"labels_complete_antennas_{activities_literal}{SHARP_SUFFIX}"
    files_path = exp_dir / f"files_complete_antennas_{activities_literal}{SHARP_SUFFIX}"
    num_windows_path = exp_dir / f"num_windows_complete_antennas_{activities_literal}{SHARP_SUFFIX}"

    with open(labels_path, "wb") as fp:
        pickle.dump(labels_wind, fp)
    with open(files_path, "wb") as fp:
        pickle.dump(names_complete, fp)
    with open(num_windows_path, "wb") as fp:
        pickle.dump(num_windows, fp)

    print(f"Saved complete samples : {len(csi_matrices_wind)}")
    print(f"Saved labels file      : {labels_path}")
    print(f"Saved files list       : {files_path}")
    print(f"Saved num_windows file : {num_windows_path}")
    print(
        "Per-sample shape        : "
        f"(n_tot={args.n_tot}, feature_length, window_length)=({args.n_tot}, ?, {args.window_length})"
    )


if __name__ == "__main__":
    main()

