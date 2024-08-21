import os
import shutil
import cv2
import numpy as np
from natsort import natsorted
from skimage.metrics import structural_similarity as ssim
import json
import torch
from torchmetrics import PeakSignalNoiseRatio
from argparse import ArgumentParser

def calculate_psnr(gt_image, predicted_image, mask):
    # Ensure the images and mask have the same dimensions
    assert gt_image.shape == predicted_image.shape == mask.shape
    
    # Convert images to torch tensors
    gt_image = torch.tensor(gt_image, dtype=torch.float32)
    predicted_image = torch.tensor(predicted_image, dtype=torch.float32)
    mask = torch.tensor(mask, dtype=torch.bool)
    
    # Apply the mask to the images
    gt_masked = gt_image[mask]
    predicted_masked = predicted_image[mask]
    
    # Calculate the PSNR using torchmetrics
    psnr_metric = PeakSignalNoiseRatio(data_range=255.0)
    psnr = psnr_metric(predicted_masked, gt_masked).item()
    
    return psnr


def calculate_ssim(gt_image, predicted_image):
    """Calculate the SSIM (Structural Similarity Index) between two images."""
    # Ensure the images are at least 7x7 in size
    min_dimension = min(gt_image.shape[0], gt_image.shape[1])
    win_size = min(7, min_dimension // 2 * 2 + 1)  # Ensure win_size is odd and <= min_dimension

    ssim_value, _ = ssim(gt_image, predicted_image, full=True, multichannel=True, win_size=win_size, channel_axis=2)
    return ssim_value

def calculate_average_metrics(gt_folder, predicted_folder, mask_folder):
    """Calculate the average PSNR and SSIM between images in gt_folder and predicted_folder."""
    psnr_values = []
    ssim_values = []

    gt_files = natsorted(os.listdir(gt_folder))
    pred_files = natsorted(os.listdir(predicted_folder))
    mask_files = natsorted(os.listdir(mask_folder))
    for i in range(len(gt_files)):
        gt_image_path = os.path.join(gt_folder, gt_files[i])
        predicted_image_path = os.path.join(predicted_folder, pred_files[i])
        mask_path = os.path.join(mask_folder, mask_files[i])

        # Check if both GT image and predicted image exist
        if os.path.exists(gt_image_path) and os.path.exists(predicted_image_path):
            # Read the images
            gt_image = cv2.imread(gt_image_path)
            predicted_image = cv2.imread(predicted_image_path)
            mask_image = cv2.imread(mask_path)
            mask_image = cv2.resize(mask_image, (predicted_image.shape[1], predicted_image.shape[0]))

            if gt_image is not None and predicted_image is not None:
                # Calculate PSNR
                psnr = calculate_psnr(gt_image, predicted_image, mask_image)
                psnr_values.append(psnr)

                # Calculate SSIM
                ssim_value = calculate_ssim(gt_image, predicted_image)
                ssim_values.append(ssim_value)
            else:
                print(f"Skipping {gt_files[i]}: Unable to read one of the images.")
        else:
            print(f"Skipping {gt_files[i]}: Corresponding predicted image not found.")

    if psnr_values and ssim_values:
        average_psnr = sum(psnr_values) / len(psnr_values)
        average_ssim = sum(ssim_values) / len(ssim_values)
        return average_psnr, average_ssim
    else:
        print("No PSNR or SSIM values were calculated.")
        return None, None
    

def calculate_average_psnr(gt_folder, predicted_folder, mask_folder):
    """Calculate the average PSNR between images in gt_folder and predicted_folder within the masked region."""
    psnr_values = []

    for image_name in os.listdir(gt_folder):
        gt_image_path = os.path.join(gt_folder, image_name)
        predicted_image_path = os.path.join(predicted_folder, image_name)
        mask_path = os.path.join(mask_folder, image_name)

        # Check if GT image, predicted image, and mask exist
        if os.path.exists(gt_image_path) and os.path.exists(predicted_image_path) and os.path.exists(mask_path):
            # Read the images and mask
            gt_image = cv2.imread(gt_image_path)
            predicted_image = cv2.imread(predicted_image_path)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            if gt_image is not None and predicted_image is not None and mask is not None:
                # Calculate PSNR only in the masked region
                psnr = calculate_psnr(gt_image, predicted_image, mask)
                psnr_values.append(psnr)
            else:
                print(f"Skipping {image_name}: Unable to read one of the images or mask.")
        else:
            print(f"Skipping {image_name}: Corresponding predicted image or mask not found.")

    if psnr_values:
        average_psnr = sum(psnr_values) / len(psnr_values)
        return average_psnr
    else:
        print("No PSNR values were calculated.")
        return None

def calculate_metrics(gt_dir, mask_dir, pred_dir):
        average_psnr, average_ssim = calculate_average_metrics(gt_dir, pred_dir, mask_dir)
        if average_psnr is not None and average_ssim is not None:
            print(f"Average PSNR [only masked region]: {average_psnr}")
            print(f"Average SSIM: {average_ssim}\n")
        metrics = dict()
        metrics['PSNR'] = average_psnr
        metrics['SSIM'] = average_ssim

        output_folder = gt_dir.replace("gt", "metrics")
        if os.path.exists(output_folder):
            shutil.rmtree(output_folder)
            print(f"Folder '{output_folder}' deleted successfully.")
        # else:
        #     print(f"Folder '{output_folder}' does not exist.")
            
        os.makedirs(output_folder)
        with open(os.path.join(output_folder, "metrics.json"), "w") as outfile: 
            json.dump(metrics, outfile)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--in_dir", type=str, required=True)
    args = parser.parse_args()

    scenes = [24, 37, 40, 55, 63, 69, 83, 97, 105, 106, 110, 114, 118]

    for scene in scenes:
        for i in range(1000, 30001, 1000):
            print(i)
            calculate_metrics(
                f"{args.in_dir}/{scene}/test/ours_{i}/gt",
                f"{args.in_dir}/{scene}/test/ours_{i}/masks",
                f"{args.in_dir}/{scene}/test/ours_{i}/renders"
            )
                
            