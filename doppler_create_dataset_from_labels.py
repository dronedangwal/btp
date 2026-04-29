"""
Build a train/val/test dataset from Doppler .npz files and a labels CSV.

The CSV must have columns 'second' and 'activity'. For each row ``(s, activity)``,
STFT row indices are taken as::

    start = max(0, (s - window_seconds) * (fs / hop) - (win_len / hop) + 1)
    end   = (s * (fs / hop) - (win_len / hop) + 1)

Indices are floored to integers; the sample is ``doppler_map[start:end, :]`` (half-open
slice). ``window_seconds`` controls how far back in time the window extends before
second ``s``.

Each ``*_doppler.npz`` in ``doppler_dir`` is processed **separately** with that
file’s own ``fs``, ``hop``, and ``win_len``. The same label rows are applied to
every file, so you get one sample per ( file , label row ) when the index range
is non-empty in that file.

Activity strings are mapped to class indices **in fixed order**:
``Still`` → 0, ``Scrolling`` → 1, ``Flipping`` → 2, ``Typing`` → 3.
Rows whose ``activity`` is not one of these (after strip, case-insensitive) are skipped.

**SHARP layout (default):** under ``output_dir`` the same structure as
``CSI_doppler_create_dataset_train.py`` + ``CSI_network.py`` expect::

    output_dir/
      train_antennas_<activities_tag>/0.txt  …   # pickle: ndarray (1, F, T), float32
      labels_train_antennas_<activities_tag>.txt # pickle: list[int]
      files_train_antennas_<activities_tag>.txt  # pickle: list[str] paths
      … same for val_* and test_*

Each ``*.txt`` pickle stores ``(1, feature_length, sample_length)`` =
``(1, F_max, T_max)`` so ``dataset_utility.load_data_single`` yields
``(T_max, F_max, 1)`` — use ``CSI_network.py`` with::

    num_tot = 1
    channels = 1
    sample_length = <T_max from prints / padded time axis>
    feature_length = <F_max>

Pass ``activities`` to ``CSI_network.py`` **exactly equal** to ``--activities_tag``.

Train / val / test splitting uses nondeterministic ``numpy.random.default_rng()``.

Usage
-----
    python doppler_create_dataset_from_labels.py \\
        --labels_csv path/to/labels.csv \\
        --doppler_dir path/to/test_compute_doppler \\
        --output_dir dataset_out \\
        --window_seconds 5 \\
        --activities_tag Still,Scrolling,Flipping,Typing

Optional: ``--save_npz`` also writes ``train.npz``, ``val.npz``, ``test.npz`` with
arrays ``X``, ``y``, etc., as before.
"""

from __future__ import annotations

import argparse
import csv
import os
import pickle
from pathlib import Path

import numpy as np

# Fixed label order: integer class id == index in this list.
ACTIVITY_CLASSES: tuple[str, ...] = ("Still", "Scrolling", "Flipping", "Typing")
# CSV activity matched case-insensitively to these canonical names.
_ACTIVITY_LOWER_TO_IDX: dict[str, int] = {
    name.lower(): i for i, name in enumerate(ACTIVITY_CLASSES)
}

SHARP_SUFFIX = ".txt"


def activity_to_class_id(activity: str) -> int | None:
    key = activity.strip().lower()
    return _ACTIVITY_LOWER_TO_IDX.get(key)


def list_doppler_files(doppler_dir: Path) -> list[Path]:
    paths = sorted(doppler_dir.glob("*_doppler.npz"))
    if not paths:
        raise FileNotFoundError(f"No *_doppler.npz files in {doppler_dir}")
    return paths


def load_doppler_file(p: Path) -> tuple[np.ndarray, float, int, int]:
    """Load one Doppler .npz: doppler_map, fs, hop, win_len."""
    data = np.load(p, allow_pickle=False)
    try:
        dm = np.asarray(data["doppler_map"], dtype=np.float32)
        f = float(np.asarray(data["fs"]).reshape(()))
        h = int(np.asarray(data["hop"]).reshape(()))
        wl = int(np.asarray(data["win_len"]).reshape(()))
    finally:
        data.close()
    return dm, f, h, wl


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
    end = s * a - b + 1.0
    start_i = int(np.floor(start))
    end_i = int(np.floor(end))
    return start_i, end_i


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
        k_sec = lower["second"]
        k_act = lower["activity"]
        for row in reader:
            sec_raw = row[k_sec]
            act = (row[k_act] or "").strip()
            if sec_raw is None or str(sec_raw).strip() == "":
                continue
            s = float(sec_raw)
            rows.append((s, act))
    return rows


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
    Write SHARP-compatible pickles: each sample is (1, F, T) float32,
    matching dataset_utility.load_data_single with num_tot=1.
    X shape (N, T, F) time-major STFT rows × Doppler bins.
    """
    sub_name = f"{split}_antennas_{activities_tag}"
    subdir = output_dir / sub_name
    _reset_sharp_subdir(subdir)

    n = X.shape[0]
    file_paths: list[str] = []
    for i in range(n):
        path_i = subdir / f"{i}{SHARP_SUFFIX}"
        # (T, F) -> (1, F, T) for pickle [stream, feature, time]
        ft = np.ascontiguousarray(np.transpose(X[i], (1, 0)), dtype=np.float32)
        cube = ft[np.newaxis, :, :]  # (1, F, T)
        with open(path_i, "wb") as fp:
            pickle.dump(cube, fp)
        file_paths.append(os.path.abspath(str(path_i)))

    labels_path = output_dir / f"labels_{split}_antennas_{activities_tag}{SHARP_SUFFIX}"
    files_path = output_dir / f"files_{split}_antennas_{activities_tag}{SHARP_SUFFIX}"
    with open(labels_path, "wb") as fp:
        pickle.dump([int(v) for v in y], fp)
    with open(files_path, "wb") as fp:
        pickle.dump(file_paths, fp)

    print(f"SHARP  {split}: {n} samples → {subdir.name}/ & {labels_path.name}, {files_path.name}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--labels_csv", type=Path, required=True, help="CSV with columns second, activity")
    p.add_argument("--doppler_dir", type=Path, required=True, help="Directory with *_doppler.npz")
    p.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Experiment folder (SHARP layout written here; pass parent as CSI_network dir + basename as subdir)",
    )
    p.add_argument(
        "--window_seconds",
        type=float,
        required=True,
        help="Look-back duration in seconds (used in start/end index formulas)",
    )
    p.add_argument(
        "--activities_tag",
        type=str,
        default="Still,Scrolling,Flipping,Typing",
        help="Must match CSI_network.py activities string / folder suffix (default: four-class tag)",
    )
    p.add_argument(
        "--save_npz",
        action="store_true",
        help="Also write train.npz, val.npz, test.npz with X, y, metadata arrays",
    )
    args = p.parse_args()

    labels_rows = read_labels_csv(args.labels_csv)
    if not labels_rows:
        raise SystemExit("No rows read from labels CSV.")

    blocks: list[np.ndarray] = []
    y_list: list[int] = []
    sec_list: list[float] = []
    fs_list: list[float] = []
    hop_list: list[int] = []
    win_len_list: list[int] = []
    stem_list: list[str] = []

    for npz_path in list_doppler_files(args.doppler_dir):
        doppler_map, fs, hop, win_len = load_doppler_file(npz_path)
        w_total, _f_bins = doppler_map.shape
        stem = npz_path.stem

        for s, activity in labels_rows:
            if s == 1:
                continue
            cls_id = activity_to_class_id(activity)
            if cls_id is None:
                continue
            start_i, end_i = stft_row_range(
                s, args.window_seconds, fs, win_len, hop
            )
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

    if not blocks:
        raise SystemExit(
            "No samples built (all rows empty after index range or out of bounds). "
            "Check 'second', window_seconds, and doppler_map length per file."
        )

    t_max = max(b.shape[0] for b in blocks)
    f_max = max(b.shape[1] for b in blocks)
    n = len(blocks)
    X = np.zeros((n, t_max, f_max), dtype=np.float32)
    lengths = np.zeros(n, dtype=np.int64)
    freq_bins = np.zeros(n, dtype=np.int64)
    for i, b in enumerate(blocks):
        tt, ff = b.shape
        X[i, :tt, :ff] = b
        lengths[i] = tt
        freq_bins[i] = ff

    y = np.asarray(y_list, dtype=np.int64)
    seconds_meta = np.asarray(sec_list, dtype=np.float64)
    fs_arr = np.asarray(fs_list, dtype=np.float64)
    hop_arr = np.asarray(hop_list, dtype=np.int64)
    win_len_arr = np.asarray(win_len_list, dtype=np.int64)
    source_file = np.asarray(stem_list, dtype=object)

    rng = np.random.default_rng()
    perm = rng.permutation(n)
    n_train = int(np.floor(0.6 * n))
    n_val = int(np.floor(0.2 * n))
    i_tr = perm[:n_train]
    i_va = perm[n_train : n_train + n_val]
    i_te = perm[n_train + n_val :]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tag = args.activities_tag

    write_sharp_split(args.output_dir, tag, "train", X[i_tr], y[i_tr])
    write_sharp_split(args.output_dir, tag, "val", X[i_va], y[i_va])
    write_sharp_split(args.output_dir, tag, "test", X[i_te], y[i_te])

    if args.save_npz:

        def save_npz_split(name: str, idx: np.ndarray) -> None:
            out = args.output_dir / f"{name}.npz"
            np.savez_compressed(
                out,
                X=X[idx],
                y=y[idx],
                seconds=seconds_meta[idx],
                length=lengths[idx],
                freq_bins=freq_bins[idx],
                fs=fs_arr[idx],
                hop=hop_arr[idx],
                win_len=win_len_arr[idx],
                source_file=source_file[idx],
                window_seconds=np.float64(args.window_seconds),
                time_steps_max=np.int64(t_max),
                freq_bins_max=np.int64(f_max),
            )
            print(f"npz   {name}: {out}  shape X={X[idx].shape}")

        save_npz_split("train", i_tr)
        save_npz_split("val", i_va)
        save_npz_split("test", i_te)

    print(f"Total samples: {n}  (train={len(i_tr)}, val={len(i_va)}, test={len(i_te)})")
    print(f"Per sample tensor (for CSI_network): sample_length={t_max}, feature_length={f_max}, channels=1, num_tot=1")
    print(f"Class ids (fixed): 0={ACTIVITY_CLASSES[0]}, 1={ACTIVITY_CLASSES[1]}, "
          f"2={ACTIVITY_CLASSES[2]}, 3={ACTIVITY_CLASSES[3]}")
    print("Train/val/test shuffle: nondeterministic (numpy default_rng with OS entropy)")
    parent = args.output_dir.resolve().parent
    leaf = args.output_dir.resolve().name
    print(f"CSI_network.py example: dir={parent}{os.sep}  subdirs={leaf}  activities=<same as --activities_tag>")


if __name__ == "__main__":
    main()
