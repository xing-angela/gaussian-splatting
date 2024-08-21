ROOT_DIR=/users/axing2/data/users/axing2/GSExperiments/Experiments/FMB/fmb-plus
DATA_DIR=/users/axing2/data/users/axing2/gaussian-splatting/data/dtu
SEQS=( 24 37 40 55 63 65 69 83 97 105 106 110 114 118 122 )

GAUS=100
SH=2
for SEQ in "${SEQS[@]}"
do
    echo "Running Sequence $SEQ with $GAUS Gaussians and SH $SH"

    python3 $ROOT_DIR/run_fmb.py \
        --data_dir $DATA_DIR/$SEQ \
        --output_dir $ROOT_DIR/outputs/3DGS/$SEQ/g$GAUS'_sh'$SH \
        --num_gaussians $GAUS \
        --sh_degree $SH \
        --stage 1
done

GAUS=1000
SH=0

for SEQ in "${SEQS[@]}"
do
    echo "Running Sequence $SEQ with $GAUS Gaussians and SH $SH"

    python3 $ROOT_DIR/run_fmb.py \
        --data_dir $DATA_DIR/$SEQ \
        --output_dir $ROOT_DIR/outputs/INIT_GS/$SEQ/g$GAUS'_sh'$SH \
        --num_gaussians $GAUS \
        --sh_degree $SH \
        --stage 1
done