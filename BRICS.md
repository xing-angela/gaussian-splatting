# 3D Gaussian Splatting for BRICS
Here are some instructions for running GS on BRICS data. For the environment, create a conda envrionment following the steps in the README (I potentially installed more packages )

## Dataset Structure
```
gaussian-splatting
|_ data
    |_ <sequence>
        |_ calib
            |_ params.txt
        |_ images
            |_ image
                |_ cam00
                ...
        |_ pc (this folder is produced by running the colmap_pc script)
            |_ dense
                |_ points3D.ply
            ...
```

## Extracting the Initial Pointcloud
I am using COLMAP to extract a dense point cloud. There is already a singularity in the public folder on CCV, so you can follow these steps to get the initial point cloud:
```
singularity shell --nv  --bind <path to the gaussian-splatting directory>:/code /oscar/data/ssrinath/public/singularity-images/colmap.sif

cd /code

python3 colmap_pc.py --data_dir data/<sequence>
```
Getting the dense point cloud from COLMAP will take a bit of time, so if there is a better/faster method, pls let me know :)

# Running and Rendering
I have changed the code so that it defaults to running on the BRICS baby data. Also, the trajectory rendering code is still messy especially because I basically hardcoded everything specific for the baby scene. All of this is run in the conda envrionment for gaussian splatting.

For training:
```
python train.py -s data/<sequence> -m output/<sequence>
```
You can add in the `--eval` flag if you want to set aside cameras for evaluation. Right now, I'm setting aside 2 cameras for eval for BRICS baby data. You have to have the `--trajectory` flag if you want to have the trajectory cameras. 

For rendering:
```
python render.py -m output/<sequence>
```
The `--skip_train` will skip rendering for the training views. The `--skip_test` will skip rendering for testing view. The `--skip_trajectory` will skip rendering for the trajectory views. 

For training and rendering there is also the `--scene_type` flag that can either be set to `BRICS`, `Colmap`, or `Blender` depending on what data you are using. The default for this is `BRICS`. Also, for brics data, I changed the `world_tranform_matrix` in `scene.cameras.py`, so if you want to go back to the other data, you will need to change it back to the commented line. 