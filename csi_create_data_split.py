"""
Build a train/val/test dataset from Doppler .npz files and two labels CSVs
(one for scrolling, one for flipping).

Directory structure expected under ``doppler_dir``::

    doppler_dir/
      subdir_a/
        1_doppler.npz
        3_doppler.npz
        7_doppler.npz
        ...
      subdir_b/
        ...

For each ``n_doppler.npz`` file found (recursively through subdirectories):
  - n in [1, 5]  → labels are read from ``scroll_labels_csv``
  - n > 5        → labels are read from ``flip_labels_csv``
  - n < 1        → file is skipped

STFT row indices for each label row ``(s, activity)``::

    start = max(0, (s - window_seconds) * (fs / hop) - (win_len / hop) + 1)
    end   = (s * (fs / hop) - (win_len / hop) + 1)

Indices are floored to integers; the sample is ``doppler_map[start:end, :]``.

Activity strings are mapped to class indices **in fixed order**:
``still`` → 0, ``scroll`` → 1, ``flip`` → 2, ``type`` → 3.
Rows whose ``activity`` is not one of these (after strip, case-insensitive) are skipped.

**SHARP layout (default):** under ``output_dir``::

    output_dir/
      train_antennas_<activities_tag>/0.txt  …   # pickle: ndarray (1, F, T), float32
      labels_train_antennas_<activities_tag>.txt
      files_train_antennas_<activities_tag>.txt
      … same for val_* and test_*

Each ``*.txt`` pickle stores ``(1, F_max, T_max)`` so ``dataset_utility.load_data_single``
yields ``(T_max, F_max, 1)``.

Train / val / test splitting uses nondeterministic ``numpy.random.default_rng()``.

Usage
-----
    python doppler_create_dataset_from_labels.py \\
        --scroll_labels_csv path/to/scroll_labels.csv \\
        --flip_labels_csv   path/to/flip_labels.csv \\
        --doppler_dir       path/to/root_doppler_dir \\
        --output_dir        dataset_out \\
        --window_seconds    5 \\
        --activities_tag    still,scroll,flip,type

Optional: ``--save_npz`` also writes ``train.npz``, ``val.npz``, ``test.npz``.
"""

from __future__ import annotations

import argparse
import csv
import os
import pickle
import re
from pathlib import Path

import numpy as np

# Fixed label order: integer class id == index in this list.
ACTIVITY_CLASSES: tuple[str, ...] = ("still", "scroll", "flip", "type")
_ACTIVITY_LOWER_TO_IDX: dict[str, int] = {
    name.lower(): i for i, name in enumerate(ACTIVITY_CLASSES)
}

SHARP_SUFFIX = ".txt"

# Regex to extract the leading integer n from filenames like "3_doppler.npz"
_NPZ_N_RE = re.compile(r"^(\d+)_doppler\.npz$", re.IGNORECASE)


def activity_to_class_id(activity: str) -> int | None:
    return _ACTIVITY_LOWER_TO_IDX.get(activity.strip().lower())


def read_labels_csv(path: Path) -> list[tuple[float, str]]:
    """Read a CSV with columns 'second' and 'activity'. Returns list of (second, activity)."""
    rows: list[tuple[float, str]] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header row: {path}")
        lower = {h.lower().strip(): h for h in reader.fieldnames}
        if "second" not in lower or "activity" not in lower:
            raise ValueError(
                f"CSV must have columns 'second' and 'activity'; got {reader.fieldnames!r} in {path}"
            )
        k_sec = lower["second"]
        k_act = lower["activity"]
        for row in reader:
            sec_raw = row[k_sec]
            act = (row[k_act] or "").strip()
            if sec_raw is None or str(sec_raw).strip() == "":
                continue
            rows.append((float(sec_raw), act))
    return rows


def find_doppler_files(doppler_dir: Path) -> list[tuple[Path, int]]:
    """
    Recursively find all n_doppler.npz files under doppler_dir.
    Returns list of (path, n) sorted by subdirectory then n.
    Files with n < 1 are skipped with a warning.
    """
    results: list[tuple[Path, int]] = []
    for npz_path in sorted(doppler_dir.rglob("*_doppler.npz")):
        m = _NPZ_N_RE.match(npz_path.name)
        if not m:
            print(f"  [skip] Cannot parse n from filename: {npz_path.name}")
            continue
        n = int(m.group(1))
        if n < 1:
            print(f"  [skip] n={n} < 1 in {npz_path.relative_to(doppler_dir)}")
            continue
        results.append((npz_path, n))

    if not results:
        raise FileNotFoundError(
            f"No parseable n_doppler.npz files found under {doppler_dir}"
        )
    return results


def load_doppler_file(p: Path) -> tuple[np.ndarray, float, int, int]:
    """Load one Doppler .npz: doppler_map, fs, hop, win_len."""
    data = np.load(p, allow_pickle=False)
    try:
        dm  = np.asarray(data["doppler_map"], dtype=np.float32)
        fs  = float(np.asarray(data["fs"]).reshape(()))
        hop = int(np.asarray(data["hop"]).reshape(()))
        wl  = int(np.asarray(data["win_len"]).reshape(()))
    finally:
        data.close()
    return dm, fs, hop, wl


def stft_row_range(
    s: float,
    window_seconds: float,
    fs: float,
    win_len: int,
    hop: int,
) -> tuple[int, int]:
    a = fs / float(hop)
    b = float(win_len) / float(hop)
    start = max(0.0, (s - window_seconds) * a - b + 1.0)
    end   = s * a - b + 1.0
    return int(np.floor(start)), int(np.floor(end))


def _reset_sharp_subdir(subdir: Path) -> None:
    subdir.mkdir(parents=True, exist_ok=True)
    for f in subdir.glob(f"*{SHARP_SUFFIX}"):
        f.unlink()


def write_sharp_split(
    output_dir: Path,
    activities_tag: str,
    split: str,
    X: np.ndarray,
    y: np.ndarray,
) -> None:
    """
    Write SHARP-compatible pickles.
    X shape (N, T, F) → each sample stored as (1, F, T) float32.
    """
    sub_name = f"{split}_antennas_{activities_tag}"
    subdir   = output_dir / sub_name
    _reset_sharp_subdir(subdir)

    file_paths: list[str] = []
    for i in range(X.shape[0]):
        path_i = subdir / f"{i}{SHARP_SUFFIX}"
        # (T, F) → (1, F, T)
        cube = np.ascontiguousarray(np.transpose(X[i], (1, 0)), dtype=np.float32)[np.newaxis]
        with open(path_i, "wb") as fp:
            pickle.dump(cube, fp)
        file_paths.append(os.path.abspath(str(path_i)))

    labels_path = output_dir / f"labels_{split}_antennas_{activities_tag}{SHARP_SUFFIX}"
    files_path  = output_dir / f"files_{split}_antennas_{activities_tag}{SHARP_SUFFIX}"
    with open(labels_path, "wb") as fp:
        pickle.dump([int(v) for v in y], fp)
    with open(files_path, "wb") as fp:
        pickle.dump(file_paths, fp)

    n = X.shape[0]
    print(f"  SHARP {split:5s}: {n:4d} samples → {subdir.name}/  "
          f"& {labels_path.name}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--scroll_labels_csv", type=Path, required=True,
        help="Labels CSV (second, activity) used for n_doppler.npz where 1 ≤ n ≤ 5",
    )
    ap.add_argument(
        "--flip_labels_csv", type=Path, required=True,
        help="Labels CSV (second, activity) used for n_doppler.npz where n > 5",
    )
    ap.add_argument(
        "--doppler_dir", type=Path, required=True,
        help="Root directory; subdirectories are searched recursively for n_doppler.npz",
    )
    ap.add_argument(
        "--output_dir", type=Path, required=True,
        help="Output root (SHARP layout written here)",
    )
    ap.add_argument(
        "--window_seconds", type=float, required=True,
        help="Look-back duration in seconds for the STFT window",
    )
    ap.add_argument(
        "--activities_tag",
        type=str,
        default="still,scroll,flip,type",
        help="Must match CSI_network.py activities string / folder suffix",
    )
    ap.add_argument(
        "--save_npz", action="store_true",
        help="Also write train.npz, val.npz, test.npz",
    )
    args = ap.parse_args()

    # ------------------------------------------------------------------
    # Load both label files upfront
    # ------------------------------------------------------------------
    print(f"Loading scroll labels: {args.scroll_labels_csv}")
    scroll_labels = read_labels_csv(args.scroll_labels_csv)
    print(f"  {len(scroll_labels)} rows")

    print(f"Loading flip labels:   {args.flip_labels_csv}")
    flip_labels = read_labels_csv(args.flip_labels_csv)
    print(f"  {len(flip_labels)} rows")

    # ------------------------------------------------------------------
    # Discover all n_doppler.npz files
    # ------------------------------------------------------------------
    print(f"\nScanning doppler_dir: {args.doppler_dir}")
    doppler_files = find_doppler_files(args.doppler_dir)
    print(f"  Found {len(doppler_files)} parseable n_doppler.npz files")

    # ------------------------------------------------------------------
    # Build samples
    # ------------------------------------------------------------------
    blocks:      list[np.ndarray] = []
    y_list:      list[int]        = []
    sec_list:    list[float]      = []
    fs_list:     list[float]      = []
    hop_list:    list[int]        = []
    win_len_list: list[int]       = []
    stem_list:   list[str]        = []
    n_val_list:  list[int]        = []   # which n the file came from

    skipped_files = 0
    for npz_path, n in doppler_files:
        # Choose label set based on n
        if 1 <= n <= 5:
            label_rows = scroll_labels
            label_source = "scroll"
        else:  # n > 5 (n < 1 already filtered)
            label_rows = flip_labels
            label_source = "flip"

        rel = npz_path.relative_to(args.doppler_dir)
        print(f"  [{label_source:6s} labels | n={n:2d}] {rel}")

        doppler_map, fs, hop, win_len = load_doppler_file(npz_path)
        w_total, _f_bins = doppler_map.shape
        stem = npz_path.stem
        samples_this_file = 0

        for s, activity in label_rows:
            if s == 1:
                continue
            cls_id = activity_to_class_id(activity)
            if cls_id is None:
                continue
            start_i, end_i = stft_row_range(s, args.window_seconds, fs, win_len, hop)
            if end_i <= start_i:
                continue
            end_i = min(end_i, w_total)
            if start_i >= w_total or start_i >= end_i:
                continue

            block = doppler_map[start_i:end_i, :].copy()
            blocks.append(block)
            y_list.append(cls_id)
            sec_list.append(s)
            fs_list.append(fs)
            hop_list.append(hop)
            win_len_list.append(win_len)
            stem_list.append(stem)
            n_val_list.append(n)
            samples_this_file += 1

        if samples_this_file == 0:
            skipped_files += 1
            print(f"    ↳ 0 samples extracted (index range empty or out of bounds)")
        else:
            print(f"    ↳ {samples_this_file} samples extracted")

    if not blocks:
        raise SystemExit(
            "No samples built from any file. "
            "Check --window_seconds, label CSVs, and doppler_map lengths."
        )

    # ------------------------------------------------------------------
    # Pad all samples to (T_max, F_max) and stack
    # ------------------------------------------------------------------
    t_max = max(b.shape[0] for b in blocks)
    f_max = max(b.shape[1] for b in blocks)
    n_total = len(blocks)

    X       = np.zeros((n_total, t_max, f_max), dtype=np.float32)
    lengths  = np.zeros(n_total, dtype=np.int64)
    freq_bins_arr = np.zeros(n_total, dtype=np.int64)

    for i, b in enumerate(blocks):
        tt, ff = b.shape
        X[i, :tt, :ff] = b
        lengths[i]      = tt
        freq_bins_arr[i] = ff

    y            = np.asarray(y_list,     dtype=np.int64)
    seconds_meta = np.asarray(sec_list,   dtype=np.float64)
    fs_arr       = np.asarray(fs_list,    dtype=np.float64)
    hop_arr      = np.asarray(hop_list,   dtype=np.int64)
    win_len_arr  = np.asarray(win_len_list, dtype=np.int64)
    source_file  = np.asarray(stem_list,  dtype=object)
    n_vals       = np.asarray(n_val_list, dtype=np.int64)

    # ------------------------------------------------------------------
    # Train / val / test split (60 / 20 / 20)
    # ------------------------------------------------------------------
    rng   = np.random.default_rng()
    perm  = rng.permutation(n_total)
    n_train = int(np.floor(0.6 * n_total))
    n_val   = int(np.floor(0.2 * n_total))
    i_tr = perm[:n_train]
    i_va = perm[n_train : n_train + n_val]
    i_te = perm[n_train + n_val :]

    # ------------------------------------------------------------------
    # Write SHARP output
    # ------------------------------------------------------------------
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tag = args.activities_tag

    print(f"\nWriting SHARP layout to: {args.output_dir}")
    write_sharp_split(args.output_dir, tag, "train", X[i_tr], y[i_tr])
    write_sharp_split(args.output_dir, tag, "val",   X[i_va], y[i_va])
    write_sharp_split(args.output_dir, tag, "test",  X[i_te], y[i_te])

    # ------------------------------------------------------------------
    # Optional .npz dumps
    # ------------------------------------------------------------------
    if args.save_npz:
        def save_npz_split(name: str, idx: np.ndarray) -> None:
            out = args.output_dir / f"{name}.npz"
            np.savez_compressed(
                out,
                X=X[idx],
                y=y[idx],
                seconds=seconds_meta[idx],
                length=lengths[idx],
                freq_bins=freq_bins_arr[idx],
                fs=fs_arr[idx],
                hop=hop_arr[idx],
                win_len=win_len_arr[idx],
                source_file=source_file[idx],
                n_file=n_vals[idx],
                window_seconds=np.float64(args.window_seconds),
                time_steps_max=np.int64(t_max),
                freq_bins_max=np.int64(f_max),
            )
            print(f"  npz {name}: {out}  shape X={X[idx].shape}")

        print("\nSaving .npz files:")
        save_npz_split("train", i_tr)
        save_npz_split("val",   i_va)
        save_npz_split("test",  i_te)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"Total samples : {n_total}  "
          f"(train={len(i_tr)}, val={len(i_va)}, test={len(i_te)})")
    print(f"Files skipped (0 samples): {skipped_files}")
    print(f"Tensor shape  : sample_length={t_max}, feature_length={f_max}, "
          f"channels=1, num_tot=1")
    print(f"Class ids     : " +
          ", ".join(f"{i}={c}" for i, c in enumerate(ACTIVITY_CLASSES)))
    print(f"Split shuffle : nondeterministic (numpy default_rng OS entropy)")
    parent = args.output_dir.resolve().parent
    leaf   = args.output_dir.resolve().name
    print(f"CSI_network.py: dir={parent}{os.sep}  subdirs={leaf}  "
          f"activities=<same as --activities_tag>")


if __name__ == "__main__":
    main()
