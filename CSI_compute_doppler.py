"""
CSI_compute_doppler.py
----------------------
Stage 2 of the CSI-Doppler pipeline.

Reads preprocessed CSI .npz files (output of 01_extract_and_preprocess.py),
computes the Doppler stack and collapsed Doppler-profile maps, and saves the
results to disk.

Output layout (one .npz per input .npz file):
    <output_dir>/
        <stem>_doppler.npz      # contains: doppler_map, freqs
            doppler_map : float32 (W, F)   – normalised Doppler profile
                          W = number of observation windows
                          F = nfft (Doppler frequency bins)
            freqs       : float64 (F,)     – Doppler frequency axis [Hz]

Usage
-----
    python CSI_compute_doppler.py \
        --input_dir  /path/to/preprocessed  \
        --output_dir /path/to/doppler        \
        --fs         100.0

Optional flags (all have defaults):
    --win_len  256
    --hop      30
    --nfft     256
"""

import argparse
import numpy as np
from pathlib import Path
from scipy.signal.windows import hann


# ---------------------------------------------------------------------------
# Doppler stack generation
# ---------------------------------------------------------------------------

def generate_doppler_stack(csi_matrix_processed, fs, win_len=256, hop=30, nfft=256):
    """
    Compute a short-time Fourier transform along the time axis of the CSI
    amplitude matrix.

    Parameters
    ----------
    csi_matrix_processed : complex ndarray, shape (T, S)
        Preprocessed CSI – T time samples, S subcarriers.
    fs      : float – CSI packet sampling rate [Hz]
    win_len : int   – STFT window length (samples)
    hop     : int   – hop size between consecutive windows (samples)
    nfft    : int   – FFT size (zero-padding if nfft > win_len)

    Returns
    -------
    stack : float32 ndarray, shape (W, nfft, S)
        Power spectrum per observation window per subcarrier.
        W = (T - win_len) // hop + 1
    freqs : float64 ndarray, shape (nfft,)
        Doppler frequency axis in Hz (fftshifted, centred on 0).
    """
    T, S = csi_matrix_processed.shape
    amp  = np.abs(csi_matrix_processed)           # (T, S)

    num_frames = (T - win_len) // hop + 1
    hann_win   = hann(win_len)[:, None]           # (win_len, 1) – broadcast over S

    stack = np.empty((num_frames, nfft, S), dtype=np.float32)

    for i in range(num_frames):
        start   = i * hop
        segment = amp[start : start + win_len]    # (win_len, S)

        if segment.shape[0] < win_len:
            stack = stack[:i]                     # trim if last window is short
            break

        windowed = segment * hann_win             # (win_len, S)
        S_fft    = np.fft.fft(windowed, n=nfft, axis=0)
        S_fft    = np.fft.fftshift(S_fft, axes=0)

        stack[i] = np.abs(S_fft) ** 2            # power spectrum (equivalent to S_fft * conj)

    freqs = np.fft.fftshift(np.fft.fftfreq(nfft, d=1.0 / fs))
    return stack, freqs                           # (W, nfft, S), (nfft,)


# ---------------------------------------------------------------------------
# Doppler-profile map collapse
# ---------------------------------------------------------------------------

def generate_csi_d_maps(stack):
    """
    Collapse the per-subcarrier Doppler stack into a single Doppler-profile
    map by summing across the subcarrier axis (axis=-1 / axis=2).

    Parameters
    ----------
    stack : float32 ndarray, shape (W, F, S)
        Output of generate_doppler_stack.

    Returns
    -------
    doppler_map : float32 ndarray, shape (W, F)
        Row-normalised (by row max) Doppler energy profile.
        Each row is one observation window.

    Notes
    -----
    The original code contained a NameError (csi_d_profile_array used before
    assignment).  That bug is fixed here.
    """
    # Sum power across subcarriers → (W, F)
    doppler_profile = stack.sum(axis=2)                           # (W, F)

    row_max = doppler_profile.max(axis=1, keepdims=True)          # (W, 1)
    # Guard against all-zero windows (shouldn't happen with real data)
    row_max = np.where(row_max == 0, 1.0, row_max)
    doppler_map = (doppler_profile / row_max).astype(np.float32)  # (W, F)

    return doppler_map


# ---------------------------------------------------------------------------
# Per-file pipeline
# ---------------------------------------------------------------------------

def process_file(npz_path, output_dir, args):
    """Load preprocessed CSI → compute Doppler stack → save."""
    stem     = Path(npz_path).stem
    out_path = Path(output_dir) / f"{stem}_doppler.npz"

    if out_path.exists():
        print(f"  [SKIP] {out_path} already exists.")
        return

    print(f"  Loading: {npz_path}")
    data = np.load(npz_path, allow_pickle=False)
    ts   = data["ts"]                   # (T,)
    csi  = data["csi"]                  # (T, S), complex64

    print(f"    csi shape: {csi.shape}, dtype: {csi.dtype}")

    # Estimate fs from timestamps if not supplied via --fs
    fs = args.fs
    if fs is None:
        if len(ts) > 1:
            # timestamps are in nanoseconds
            dt_ns = np.median(np.diff(ts.astype(np.float64)))
            fs    = 1e9 / dt_ns
            print(f"    Estimated fs = {fs:.2f} Hz from timestamps")
        else:
            raise ValueError(
                "Cannot estimate fs: only one timestamp available. "
                "Please supply --fs explicitly."
            )

    print(f"  Computing Doppler stack (fs={fs:.2f} Hz, win={args.win_len}, "
          f"hop={args.hop}, nfft={args.nfft}) ...")
    stack, freqs = generate_doppler_stack(
        csi_matrix_processed = csi,
        fs      = fs,
        win_len = args.win_len,
        hop     = args.hop,
        nfft    = args.nfft,
    )
    print(f"    stack shape: {stack.shape}  (windows × freq_bins × subcarriers)")

    print("  Collapsing to Doppler-profile map ...")
    doppler_map = generate_csi_d_maps(stack)
    print(f"    doppler_map shape: {doppler_map.shape}  (windows × freq_bins)")

    np.savez_compressed(
        out_path,
        doppler_map = doppler_map,     # float32 (W, F)
        freqs       = freqs,           # float64 (F,)
    )
    print(f"  Saved → {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Stage 2 – Doppler vector computation")
    p.add_argument("--input_dir",  required=True,
                   help="Directory containing preprocessed .npz files")
    p.add_argument("--output_dir", required=True,
                   help="Where to write Doppler .npz files")
    p.add_argument("--fs",  type=float, default=None,
                   help="CSI sampling rate [Hz]. If omitted, estimated from timestamps.")
    p.add_argument("--win_len", type=int, default=256,
                   help="STFT window length in samples (default: 256)")
    p.add_argument("--hop",     type=int, default=30,
                   help="Hop size between windows in samples (default: 30)")
    p.add_argument("--nfft",    type=int, default=256,
                   help="FFT size (default: 256)")
    return p.parse_args()


def main():
    args = parse_args()
    input_dir  = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    npz_files = sorted(input_dir.glob("*.npz"))
    if not npz_files:
        print(f"No .npz files found in {input_dir}")
        return

    print(f"Found {len(npz_files)} .npz file(s) in {input_dir}")
    for npz_path in npz_files:
        process_file(npz_path, output_dir, args)

    print("\nStage 2 complete.")


if __name__ == "__main__":
    main()
