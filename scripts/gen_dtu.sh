ROOT_DIR=/users/axing2/data/datasets/mvs_training/dtu

SEQS=( 24 37 40 55 63 65 69 83 97 105 106 110 114 118 122 )

for SEQ in "${SEQS[@]}"
do
    echo "Processing Sequence $SEQ"

    python fmb/generate_dtu_inputs.py \
        --in_dir $ROOT_DIR \
        --sequence $SEQ \
        --out_dir ./data/dtu
done