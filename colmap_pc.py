import os
import cv2
import shutil
import logging
import sqlite3
import subprocess
import numpy as np

from glob import glob
from argparse import ArgumentParser
from utils.colmap_database import COLMAPDatabase

def cam_db_from_file(cam_file, db):
    with open(cam_file) as f:
        for line in f.readlines():
            if line.startswith("#"):
                continue
            data = line.split()

            cam_id = int(data[0])
            model = 4
            width = int(data[2])
            height = int(data[3])
            params = np.array([float(datum) for datum in data[4:]])
            
            db.add_camera(model, width, height, params, camera_id=cam_id)
    
    db.commit()

def img_db_from_file(img_file, db):
    with open(img_file) as f:
        skip_next = False
        for line in f.readlines():
            if skip_next:
                skip_next = False
                continue
            if line.startswith("#"):
                continue
            data = line.split()

            img_id = int(data[0])
            qw = float(data[1])
            qx = float(data[2])
            qy = float(data[3])
            qz = float(data[4])
            tx = float(data[5])
            ty = float(data[6])
            tz = float(data[7])
            prior_q = np.array([qw, qx, qy, qz])
            prior_t = np.array([tx, ty, tz])
            cam_id = int(data[8])
            name = data[9]

            db.add_image(name, cam_id, image_id=img_id)
            skip_next = True
    
    db.commit()

def create_files(args, params_path, input_image_path):
    cam_out_path = os.path.join(args.data_dir, "pc", "sparse", "cameras.txt")
    img_out_path = os.path.join(args.data_dir, "pc", "sparse", "images.txt")
    points3d_path = os.path.join(args.data_dir,"pc", "sparse", "points3D.txt")
    open(points3d_path, 'a').close()

    image_path = os.path.join(args.data_dir, "pc", "images")
    image_dir = os.path.join(args.data_dir, "image")

    image_names = []
    image_dirs = list(sorted(glob(f"{input_image_path}/*cam*")))
    for image_dir in image_dirs:
        img_name = os.path.basename(sorted(glob(f"{image_dir}/*.png"))[0])
        name = os.path.join(os.path.basename(image_dir), img_name)
        image_names.append(name)

    img_params = []
    cam_params = []

    # read the params file
    with open(params_path) as f:
        for line in f.readlines():
            if line.startswith("#"):
                continue
            data = line.split()
            cam_param = []
            cam_param.append(int(data[0])) # camera id
            cam_param.append("OPENCV") # model
            cam_param.append(int(data[1])) # width
            cam_param.append(int(data[2])) # height
            cam_param += [float(datum) for datum in data[3:11]] # params
            cam_params.append(tuple(cam_param))

            img_param = []
            img_param.append(int(data[0])) # image id
            img_param.append(float(data[12])) # qw
            img_param.append(float(data[13])) # qx
            img_param.append(float(data[14])) # qy
            img_param.append(float(data[15])) # qz
            img_param.append(float(data[16])) # tx
            img_param.append(float(data[17])) # ty
            img_param.append(float(data[18])) # tz
            img_param.append(int(data[0])) # cam id
            img_param.append(image_names[int(data[0]) - 1]) # image_name
            img_params.append(tuple(img_param))

    np.savetxt(img_out_path, img_params, fmt="%s", newline="\n\n")
    np.savetxt(cam_out_path, cam_params, fmt="%s")


def point_cloud(args):
    db_path = os.path.join(args.data_dir, "pc", "db.db")
    if os.path.exists(db_path):
        logging.warning("Previous database found, deleting.")
        os.remove(db_path)
        try:
            os.remove(os.path.join(db_path, "db.db-wal"))
            os.remove(os.path.join(db_path, "db.db-shm"))
        except FileNotFoundError:
            pass

    db = COLMAPDatabase.connect(db_path)
    db.create_tables()

    # insert the cameras and images into the database
    logging.info("Importing cameras and images in database")
    cam_file = os.path.join(args.data_dir, "calib", "params.txt")
    input_image_path = os.path.join(args.data_dir, "images", "image")

    sparse_path = os.path.join(args.data_dir, "pc", "sparse")
    if not os.path.exists(sparse_path):
        os.makedirs(sparse_path)

    create_files(args, cam_file, input_image_path)

    cam_file_path = os.path.join(args.data_dir, "pc", "sparse", "cameras.txt")
    cam_db_from_file(cam_file_path, db)

    img_file_path = os.path.join(args.data_dir, "pc", "sparse", "images.txt")
    img_db_from_file(img_file_path, db)

    # image_path = os.path.join(args.data_dir, "images")

    logging.info("Feature Extraction")
    subprocess.run([
        "colmap", "feature_extractor", 
        "--database_path", db_path, 
        "--image_path", input_image_path
    ])

    logging.info("Exhaustive Matcher")
    subprocess.run([
        "colmap", "exhaustive_matcher", 
        "--database_path", db_path,
    ])

    logging.info("Point Triangulation")
    subprocess.run([
        "colmap", "point_triangulator", 
        "--database_path", db_path, 
        "--image_path", input_image_path, 
        "--input_path", os.path.join(args.data_dir, "pc", "sparse"), 
        "--output_path", os.path.join(args.data_dir, "pc", "sparse")
    ])

    logging.info("Image Undistortion")
    subprocess.run([
        "colmap", "image_undistorter", 
        "--image_path", input_image_path, 
        "--input_path", os.path.join(args.data_dir, "pc", "sparse"),
        "--output_path", os.path.join(args.data_dir, "pc", "dense"),
        "--output_type", "COLMAP"
    ])

    logging.info("Patch Match Stereo")
    subprocess.run([
        "colmap", "patch_match_stereo", 
        "--workspace_path", os.path.join(args.data_dir, "pc", "dense")
    ])

    logging.info("Stereo Fusion")
    subprocess.run([
        "colmap", "stereo_fusion", 
        "--workspace_path", os.path.join(args.data_dir, "pc", "dense"),
        "--output_path", os.path.join(args.data_dir, "pc", "dense", "points3D.ply")
    ])


def main(args):
    pc_dir = os.path.join(args.data_dir, "pc")
    if not os.path.exists(pc_dir):
        os.makedirs(pc_dir)
    point_cloud(args)
    

if __name__ == "__main__":
    parser = ArgumentParser()
    # parser.add_argument("--root_dir", required=True, help="Base directory")
    parser.add_argument("--data_dir", required=True, help="Directory to store the data")

    args = parser.parse_args()

    main(args)
