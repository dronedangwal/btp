# btp

## Pipeline

The script `data_pipeline.sh` runs the first 4 steps of the pipeline, thus resulting in data for model training and testing. To run it, first set its parameters and then run it using your shell.

### 1. CSI Extraction & Preprocessing

Raw CSI data collected from the lab setup is first filtered to extract data corresponding to the exam duration and then preprocessed.

Run:
```sh
python csi_extract_and_preprocess.py \
    --input_dir harsh_data \
    --output_dir test_extraction_and_preprocessing
```
This step outputs cleaned and structured CSI data ready for Doppler analysis.

### 2. Doppler Computation

The preprocessed CSI is then used to compute Doppler traces using a sliding window FFT.

Run:
```sh
python csi_compute_doppler.py \
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

### 4. Dataset Creation

The train, test and validation datasets can be created using the following command.

```sh
python csi_create_data_split.py --labels_csv labels.csv --doppler_dir test_compute_doppler --output_dir dataset_out --window_seconds 2
```

To create additional test data, run the following command.
```sh
python csi_create_test_data.py \
  --doppler_dir test_compute_doppler \
  --labels_csv labels.csv \
  --output_dir . \
  --subdir test_dataset \
  --activities_tag Still,Scrolling,Flipping,Typing \
  --window_seconds 2 \
  --window_length 80 \
  --stride_length 10 \
  --n_tot 1
```

### 5. Model training and inference

After creation of datasets, to train the model, run the command

```sh
python3 CSI_network_train.py \
    --data_dir data_model \
    --results_dir results_new \
    --batch_size 32 \
    --epochs 50 \
    --lr 0.001 \
    --activity_tags still,scroll,flip,type
```

For inference, run the model with the test data.
