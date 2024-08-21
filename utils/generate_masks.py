import cv2
import os
import numpy as np
from argparse import ArgumentParser

def generate_mask(image_path, mask_path):
    """
    Generates a mask from a given image, where non-black pixels are white.

    Args:
        image_path: Path to the input image.
        mask_path: Path to save the generated mask.
    """

    img = cv2.imread(image_path)

    # Convert to grayscale for faster processing
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Create a mask where non-black pixels are white
    mask = gray > 0

    # Convert mask to uint8 for saving as an image
    mask = mask.astype(np.uint8) * 255

    cv2.imwrite(mask_path, mask)

def process_folder(input_folder, output_folder):
    """
    Processes images in a folder, generating masks and saving them to a different folder.

    Args:
        input_folder: Path to the input image folder.
        output_folder: Path to save the generated masks.
    """

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(input_folder, filename)
            mask_path = os.path.join(output_folder, filename)
            generate_mask(image_path, mask_path)

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--in_dir", type=str, required=True)
    args = parser.parse_args()

    scenes = [24, 37, 40, 55, 63, 69, 83, 97, 105, 106, 110, 114, 118]

    for scene in scenes:
        for i in range(1000, 30001, 1000):
            input_folder = f"{args.in_dir}/{scene}/test/ours_{i}/gt"
            output_folder = f"{args.in_dir}/{scene}/test/ours_{i}/masks"
            process_folder(input_folder, output_folder)