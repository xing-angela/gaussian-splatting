# !/bin/bash

RAW_DATA_DIR=/users/axing2/data/brics/non-pii/brics-mobile
DATA_DIR=/users/axing2/data/public/brics-mobile/data
OUT_DIR=/users/axing2/data/public/brics-mobile/output

SEQ=$1

# running colmap to get the point cloud and camera parameters
echo "Running COLMAP"
python3 colmap_calib.py \
	-r $RAW_DATA_DIR/$SEQ \
	--no-subdir \
	-o $DATA_DIR/$SEQ
	
# train using 3DGS
echo "Training 3DGS"
python train.py \
    -s $DATA_DIR/$SEQ \
    -m $OUT_DIR/$SEQ \
    --scene_type BRICS \
    --iterations 10_000 \
    --eval
    
# render the test views
echo "Rendering Results"
python render.py \
	-m $OUT_DIR/$SEQ \
	--scene_type BRICS \
    --eval \
    --skip_train
    # --iteration 2000