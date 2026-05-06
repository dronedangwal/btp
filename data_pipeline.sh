INPUT_DIRECTORY="data_collected"
PREPROCESSED_DATA="data_preprocessed"
DOPPLER_TRACE="data_doppler"
MODEL_DATA="data_model"
SCROLL_LABELS="scroll_labels.csv"
FLIP_LABELS="flip_labels.csv"

print "Input data collected from..."

for d in $INPUT_DIRECTORY/*(/); do
    base=${d:t}
    print -- "    $base"
done

print "Starting pre processing..."

mkdir -p "$PREPROCESSED_DATA"

for d in $INPUT_DIRECTORY/*(/); do
    base=${d:t}
    
    if [[ -z $(print -l -- $PREPROCESSED_DATA/$base/*(N)) ]]; then
        python3 CSI_extract_and_preprocess.py \
        --input_dir "$d" \
        --output_dir "$PREPROCESSED_DATA/$base" >> logs/CSI_extract_and_preprocess
    else
        echo "Preprocessed data already found for $d, skipping this step"
    fi

done

print "Preprocessing complete!"
print "Starting doppler trace computation"

for d in $PREPROCESSED_DATA/*(/); do
    base=${d:t}
    
    if [[ -z $(print -l -- $DOPPLER_TRACE/$base/*(N)) ]]; then
        python3 CSI_compute_doppler.py \
        --input_dir "$d"/ \
        --output_dir "$DOPPLER_TRACE/$base"/ \
        --win_len 100 \
        --hop 10 \
        --nfft 100 >> logs/CSI_compute_doppler
    else
        echo "Doppler traces already found for $d, skipping this step"
    fi

done

print "Doppler trace computation complete"
print "Starting dataset creation"

python3 doppler_create_dataset_from_labels.py \
    --scroll_labels_csv $SCROLL_LABELS \
    --flip_labels_csv $FLIP_LABELS \
    --doppler_dir $DOPPLER_TRACE \
    --output_dir $MODEL_DATA \
    --window_seconds 2 >> logs/doppler_create_dataset_from_labels

print "Dataset creation complete"
