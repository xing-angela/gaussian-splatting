# !/bin/bash

# undistort the images
echo "Undistorting Images"
python mastr/undistort.py \
    --data_dir /users/axing2/data/brics/non-pii/brics-studio/ \
    --sequence 2024-08-17 \
    --output_dir ./data/brics-room-sub \
    --video \
    --subsample

# get the parameters and point cloud from Mast3R
echo "Running Mast3R"
python mastr/brics_demo.py \
	--data_dir ./data/brics-room-sub \
	--sequence 2024-08-17 \
	--output_dir ./data/brics-room-sub \
	--weights ./mastr/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth \
    --cam_preset \
    --subsample
	
# train using 3DGS
echo "Training 3DGS"
python train.py \
    -s ./data/brics-room-sub/2024-08-17 \
    -m ./output/brics-room-sub/2024-08-17 \
    --scene_type BRICS \
    --iterations 10_000 \
    --eval
    
# render the test views
echo "Rendering Results"
python render.py \
	-m ./output/brics-room-sub/2024-08-17 \
	--scene_type BRICS \
    --eval \
    --skip_train
    # --iteration 2000