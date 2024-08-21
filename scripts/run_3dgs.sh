#!/bin/bash

FMB_DIR=/users/axing2/data/users/axing2/GSExperiments/Experiments/FMB/fmb-plus/outputs/INIT_GS

SEQS=( 24 37 40 55 63 65 69 83 97 105 106 110 114 118 122 )

for SEQ in "${SEQS[@]}"
do
    echo "Training Sequence $SEQ"
    python train.py \
        -s ./data/dtu/$SEQ \
        -m ./output/fmb/$SEQ \
        --scene_type DTU \
        --eval \
        --fmb_path $FMB_DIR/$SEQ/g1000_sh0 \
        --sh_degree 0

    for ITER in {1000..30001..1000};
    do
        echo "Rendering Sequence $SEQ at Iteration $ITER"
        python render.py \
            -m ./output/fmb-test/$SEQ \
            --scene_type DTU \
            --iteration $ITER
    done
done

python utils/generate_masks.py \
    --in_dir ./output/fmb

python utils/mask_psnr.py \
    --in_dir ./output/fmb

python utils/plot_graph.py \
    --in_dir ./output/fmb