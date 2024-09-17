#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import cv2
import sys
import math
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud
from utils import param_utils
from glob import glob

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name), "00000.jpg")
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readColmapSceneInfo(path, images, eval, traj, llffhold=8):
    # try:
    #     cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
    #     cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
    #     cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
    #     cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    # except:
    #     cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
    #     cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
    #     cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
    #     cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    cameras_extrinsic_file = os.path.join(path, "images.txt")
    cameras_intrinsic_file = os.path.join(path, "cameras.txt")
    cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
    cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir))
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    # ply_path = os.path.join(path, "sparse/0/points3D.ply")
    # bin_path = os.path.join(path, "sparse/0/points3D.bin")
    # txt_path = os.path.join(path, "sparse/0/points3D.txt")
    # if not os.path.exists(ply_path):
    #     print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
    #     try:
    #         xyz, rgb, _ = read_points3D_binary(bin_path)
    #     except:
    #         xyz, rgb, _ = read_points3D_text(txt_path)
    #     storePly(ply_path, xyz, rgb)

    ply_path = os.path.join(path, "points3D.ply")
    if not os.path.exists(ply_path):
        mastr_points = os.path.join(path, "mastr_points.ply")
        xyz, rgb = sample_pc(mastr_points, 300000)
        storePly(ply_path, xyz, rgb)
    
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy 
            FovX = fovx

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1]))
            
    return cam_infos

def readNerfSyntheticInfo(path, white_background, eval, extension=".png"):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

########################################### BRICS Room Data ###########################################

def sample_pc(path, sample_size):
    import trimesh

    pc = trimesh.load(path)
    points = pc.vertices
    colors = pc.colors[:, :3]

    sample_indices = np.random.choice(len(points), sample_size, replace=False)
    xyzs = points[sample_indices]
    rgbs = colors[sample_indices]

    return xyzs, rgbs


def readBricsCameras(params_path, images_folder):
    params = param_utils.read_params(params_path)

    cam_infos = []
    for idx, cam in enumerate(params):
        extr = param_utils.get_extr(cam)
        K, dist = param_utils.get_intr(cam)
        
        cam_name = cam["cam_name"]

        img_dir = os.path.join(images_folder, cam_name)
        img = os.listdir(img_dir)[0]
        img_path = os.path.join(img_dir, img)
        img_name = os.path.basename(img_path).split(".")[0]
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

        w, h = cam["width"], cam["height"]
        new_K, roi = param_utils.get_undistort_params(K, dist, (w, h))
        img = param_utils.undistort_image(K, new_K, dist, img)
        new_K = new_K.astype(np.float32)
        extr = extr.astype(np.float32)

        fx, fy = new_K[0, 0], new_K[1, 1]
        fovx = 2 * math.atan(w / (2 * fx))
        fovy = 2 * math.atan(h / (2 * fy))

        R = np.transpose(extr[:, :3])
        T = extr[:, 3]

        # handles alpha channel if there's segmentation
        if img.shape[-1] == 4:
            b, g, r, alpha = cv2.split(img)

            rgb = np.stack([r, g, b], axis=-1)
            alpha = alpha[..., np.newaxis] / 255.0
            mask = alpha

            rgb = rgb / 255.0
            rgb = rgb * alpha
        else:
            b, g, r = cv2.split(img)
            rgb = np.stack([r, g, b], axis=-1)
            rgb = rgb / 255.0

        image = Image.fromarray(np.uint8(rgb*255))

        cam_info = CameraInfo(uid=cam["cam_id"], R=R, T=T, FovY=fovy, FovX=fovx, image=image,
                              image_path=img_path, image_name=img_name, width=int(w), height=int(h))
        cam_infos.append(cam_info)
    return cam_infos

def readBricsSceneInfo(path, eval, traj):
    params_path = os.path.join(path, "calib", "params.txt")
    # params_path = os.path.join("/users/axing2/data/users/axing2/gaussian-splatting/data/brics-room/2024-08-17/calib/params.txt")
    images_folder = os.path.join(path, "images")
    cam_infos = readBricsCameras(params_path, images_folder)

    if eval:
        # eval_cams = [5, 17] # normal cameras
        eval_cams = [0, 7] # subsampled cams
        # eval_cams = [0, 1, 2, 3, 6, 7, 10, 11, 14, 15, 17, 18, 19, 22, 23, 26, 27, 30, 
        #              31, 34, 35, 38, 39, 40, 41, 44, 45, 48, 49, 52, 53, 56, 57, 60, 61]
        train_cam_infos = [c for c in cam_infos if c.uid not in eval_cams]
        test_cam_infos = [c for c in cam_infos if c.uid in eval_cams]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "reconstruction/0/points3D.ply")
    bin_path = os.path.join(path, "reconstruction/0/points3D.bin")
    txt_path = os.path.join(path, "reconstruction/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)

    # ply_path = os.path.join(path, "points3D.ply")
    # if not os.path.exists(ply_path):
    #     mastr_points = os.path.join(path, "mastr_points.ply")
    #     xyz, rgb = sample_pc(mastr_points, 300000)
    #     storePly(ply_path, xyz, rgb)
    
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

########################################### BRICS Room Data END ###########################################

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo,
    "BRICS" : readBricsSceneInfo
}