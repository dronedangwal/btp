# btp

## Pipeline

### 1. CSI Extraction & Preprocessing

Raw CSI data collected from the lab setup is first filtered to extract data corresponding to the exam duration and then preprocessed.

Run:
```sh
python CSI_extract_and_preprocess.py \
    --input_dir harsh_data \
    --output_dir test_extraction_and_preprocessing
```
This step outputs cleaned and structured CSI data ready for Doppler analysis.

### 2. Doppler Computation

The preprocessed CSI is then used to compute Doppler traces using a sliding window FFT.

Run:
```sh
python3 CSI_compute_doppler.py \
    --input_dir test_extraction_and_preprocessing/ \
    --output_dir test_compute_doppler/ \
    --win_len 100 \
    --hop 10 \
    --nfft 100
```

#### Parameters
- `win_len` → Window length (number of samples per FFT)
- `hop` → Step size between consecutive windows
- `nfft` → Number of FFT points (controls frequency resolution)

### 3. Visualization

The generated Doppler traces can be visualized as spectrograms.

Use the notebook:
```sh
doppler_visualization.py
```
This notebook demonstrates how to plot Doppler spectrograms for analysis.
