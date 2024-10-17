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

import subprocess as sp
import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import torchvision.transforms.functional as F

def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    parent_path = os.path.dirname(model_path)
    session_name = os.path.basename(model_path)

    output = os.path.join(parent_path, f"{session_name}.mp4")
    logo_path = "/users/axing2/data/users/axing2/gaussian-splatting/metadata/logo/logo_small.png"

    cmd_out = ['ffmpeg',
            '-y',  # (optional) overwrite output file if it exists
            '-hide_banner',
            '-loglevel', 'error',
            '-f', 'image2pipe',
            '-r', str(60),  # frames per second
            '-i', '-',  # The input comes from a pipe
            '-i', logo_path, # Second input stream
            '-filter_complex', "overlay=W-w-10:H-h-10", 
            # '-filter_complex', "[0:v]minterpolate=fps=60,setpts=4*PTS[video];[video][1:v]overlay=W-w-10:H-h-10", 
            '-c:v', 'h264',
            output]

    pipe = sp.Popen(cmd_out, stdin=sp.PIPE)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        rendering = torch.clamp(render(view, gaussians, pipeline, background)["render"], 0, 1)
        rendered_img = F.to_pil_image(rendering.to("cpu"), mode="RGB")
        rendered_img.save(pipe.stdin, "JPEG")
        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".jpg"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
    pipe.stdin.close()
    pipe.wait()

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, skip_traj : bool, scene_type : str):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, scene_type, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
            render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)

        if not skip_test:
            render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background)

        if not skip_traj:
            render_set(dataset.model_path, "trajectory", scene.loaded_iter, scene.getTrajCameras(), gaussians, pipeline, background)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--skip_traj", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--scene_type", default="BRICS", choices=["BRICS", "Colmap", "Blender", "DTU"])
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.skip_traj, args.scene_type)