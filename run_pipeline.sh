#!/bin/bash

#SBATCH -N 1
#SBATCH -t 1:00:00

# Check if there are exactly two arguments
if [ "$#" -ne 2 ]; then
    SCRIPT_NAME=$(basename "$0")
    echo "[ Usage ]: ${SCRIPT_NAME} <INPUT_SESSION_DIR> <OUTPUT_DIR>"
    exit 1
fi

INPUT_SESSION_DIR="${1}"
OUTPUT_DIR="${2}"
SESSION_NAME=$(basename "${INPUT_SESSION_DIR}")
OUTPUT_SESSION_DIR=${OUTPUT_DIR}/${SESSION_NAME}

# RAW_DATA_DIR=/users/axing2/data/brics/non-pii/brics-mobile
# DATA_DIR=/users/axing2/data/public/brics-mobile/data
# OUT_DIR=/users/axing2/data/public/brics-mobile/output
# SEQ=$1

# loads necessary modules and activates the envrionment
module load miniforge/23.11.0-0s
source /oscar/runtime/software/external/miniforge/23.11.0-0/etc/profile.d/conda.sh
module load cuda/12.1
conda activate brics-demo
export PATH=/users/axing2/.conda/envs/brics-demo/bin:/oscar/rt/9.2/software/0.20-generic/0.20.1/opt/spack/linux-rhel9-x86_64_v3/gcc-11.3.1/cuda-12.1.1-ebglvvqo7uhjvhvff2qlsjtjd54louaf/bin:/oscar/runtime/software/external/miniforge/23.11.0-0/bin:/oscar/runtime/bin:/usr/local/bin:/usr/bin:/usr/local/sbin:/usr/sbin:/users/axing2/bin:/usr/lpp/mmfs/bin:/usr/lpp/mmfs/sbin
module load ffmpeg
cd ~/data/users/axing2/gaussian-splatting

# running colmap to get the point cloud and camera parameters
echo "Running COLMAP"
python3 colmap_calib.py \
	-r $INPUT_SESSION_DIR \
	--no-subdir \
	-o $OUTPUT_SESSION_DIR \
    --inital_calib
	
# train using 3DGS
echo "Training 3DGS"
python train.py \
    -s $OUTPUT_SESSION_DIR \
    -m $OUTPUT_SESSION_DIR \
    --scene_type BRICS \
    --iterations 10_000 \
    --eval
    
# render the test views
echo "Rendering Results"
python render.py \
	-m $OUTPUT_SESSION_DIR \
	--scene_type BRICS \
    --eval \
    --skip_train \
    -r 1 

echo "Script done. Deleting processing lock."
rm ${OUTPUT_SESSION_DIR}/${SESSION_NAME}.lock