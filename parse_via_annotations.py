"""
Parse VIA temporal annotation export CSV and produce per-second label CSVs
for each video.

VIA export format (relevant columns):
  - file_list          : e.g. ["Video1_1.mp4"]
  - temporal_coordinates: e.g. [10, 25]  -> segment from 10s to 25s
  - metadata           : e.g. {"1":"2"}  -> attribute id 1 = option id 2

Action label mapping (from VIA file header):
  "0" -> "Ignore"
  "1" -> "Still"
  "2" -> "Typing"
  "3" -> "Scrolling"
  "4" -> "Flipping"

Output:
  One CSV per video named <video_stem>_labels.csv with columns:
    second  : integer second index (1-based, i.e. second 1 = 0s to 1s)
    label   : activity label string
"""

import re
import csv
import os
from collections import defaultdict

# ── CONFIG ────────────────────────────────────────────────────────────────────
INPUT_FILE  = "model_data/ActionLabelingforBTP13Apr2026_03h27m00s_export.csv"
OUTPUT_DIR  = "model_data"           # output CSVs written here
VIDEO_LEN   = 74            # seconds (fixed for all videos)
MAJORITY_THRESHOLD = 0.5    # fraction of second needed to assign a label

ACTION_MAP = {
    "0": "Ignore",
    "1": "Still",
    "2": "Typing",
    "3": "Scrolling",
    "4": "Flipping",
}
# ─────────────────────────────────────────────────────────────────────────────


def parse_via_csv(filepath):
    """
    Read VIA export CSV (skipping comment lines starting with #).
    Returns list of dicts with keys: video, t_start, t_end, action_id
    """
    annotations = []

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            # Skip comment / header lines
            if line.startswith("#") or not line:
                continue

            # ── Parse the 6 comma-separated fields ───────────────────────────
            # Fields may themselves contain commas inside quoted structures,
            # so we use csv.reader for safe splitting.
            row = next(csv.reader([line]))
            if len(row) < 6:
                continue

            _meta_id, file_list_raw, _flags, temporal_raw, _spatial, metadata_raw = row[:6]

            # ── Extract video filename ────────────────────────────────────────
            # file_list_raw looks like: ["Video1_1.mp4"]
            video_match = re.search(r'"([^"]+\.mp4)"', file_list_raw)
            if not video_match:
                continue
            video = video_match.group(1)

            # ── Extract temporal segment [t_start, t_end] ────────────────────
            numbers = re.findall(r'[\d.]+', temporal_raw)
            if len(numbers) < 2:
                continue
            t_start = float(numbers[0])
            t_end   = float(numbers[1])

            # ── Extract action option id from metadata {"1":"<id>"} ───────────
            action_match = re.search(r'"1"\s*:\s*"(\d+)"', metadata_raw)
            if not action_match:
                continue
            action_id = action_match.group(1)

            annotations.append({
                "video"    : video,
                "t_start"  : t_start,
                "t_end"    : t_end,
                "action_id": action_id,
            })

    return annotations


def assign_labels(annotations, video_len=74):
    """
    For each video build a list of VIDEO_LEN per-second labels.

    Second k covers the interval [k-1, k) for k in 1..video_len.
    Label assignment uses majority overlap:
      - For each second, find which annotation segment covers it most.
      - If no annotation covers it at all, label = "Unlabeled".
    """
    # Group annotations by video
    by_video = defaultdict(list)
    for ann in annotations:
        by_video[ann["video"]].append(ann)

    video_labels = {}   # video -> list of (second, label)

    for video, anns in by_video.items():
        second_labels = []

        for k in range(1, video_len + 1):
            sec_start = float(k - 1)
            sec_end   = float(k)

            # Compute overlap of each annotation with this second
            best_label   = "Unlabeled"
            best_overlap = 0.0

            for ann in anns:
                overlap = min(ann["t_end"], sec_end) - max(ann["t_start"], sec_start)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_label   = ACTION_MAP.get(ann["action_id"], "Unknown")

            second_labels.append((k, best_label))

        video_labels[video] = second_labels

    return video_labels


def write_output_csvs(video_labels, output_dir="."):
    os.makedirs(output_dir, exist_ok=True)
    written = []

    for video, labels in sorted(video_labels.items()):
        # e.g. Video1_1.mp4  ->  Video1_1_labels.csv
        stem     = os.path.splitext(video)[0].split("_")[1]
        out_path = os.path.join(output_dir, f"{stem}", "labels.csv")

        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["second", "label"])
            writer.writerows(labels)

        written.append((video, out_path, labels))
        print(f"  Wrote {out_path}  ({len(labels)} seconds)")

    return written


def print_summary(written):
    print("\n── Label distribution per video ──────────────────────────────")
    for video, path, labels in written:
        counts = defaultdict(int)
        for _, lbl in labels:
            counts[lbl] += 1
        dist = ", ".join(f"{lbl}: {cnt}s" for lbl, cnt in sorted(counts.items()))
        print(f"  {video:20s}  {dist}")


# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Reading: {INPUT_FILE}")
    annotations = parse_via_csv(INPUT_FILE)
    print(f"  Parsed {len(annotations)} annotation segments across "
          f"{len({a['video'] for a in annotations})} videos\n")

    video_labels = assign_labels(annotations, video_len=VIDEO_LEN)

    print("Writing per-video label CSVs:")
    written = write_output_csvs(video_labels, output_dir=OUTPUT_DIR)

    print_summary(written)
    print("\nDone.")
