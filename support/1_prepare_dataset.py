import os
import glob
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))

ORIGINAL_IMAGES_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, "data")

HR_SIZE = (256, 256)
LR_SIZE = (64, 64)

def create_dir(path):
    """Creates a directory if it does not exist."""
    os.makedirs(path, exist_ok=True)

def process_and_save(image_paths, set_name):
    """
    Processes a list of image paths to create and save HR/LR pairs.
    """
    hr_dir = os.path.join(PROCESSED_DATA_DIR, set_name, 'hr')
    lr_dir = os.path.join(PROCESSED_DATA_DIR, set_name, 'lr')
    create_dir(hr_dir)
    create_dir(lr_dir)

    print(f"Processing '{set_name}' set with {len(image_paths)} images...")
    
    for img_path in tqdm(image_paths, desc=f"Generating {set_name} data"):
        try:
            img = cv2.imread(img_path)
            if img is None:
                print(f"Warning: Could not read image {img_path}. Skipping.")
                continue

            hr_img = cv2.resize(img, HR_SIZE, interpolation=cv2.INTER_AREA)
            lr_img = cv2.resize(hr_img, LR_SIZE, interpolation=cv2.INTER_CUBIC)

            filename = os.path.basename(img_path)
            filename_png = os.path.splitext(filename)[0] + '.png'

            cv2.imwrite(os.path.join(hr_dir, filename_png), hr_img)
            cv2.imwrite(os.path.join(lr_dir, filename_png), lr_img)

        except Exception as e:
            print(f"Error processing {img_path}: {e}")

def main():
    """Main function to orchestrate the dataset preparation."""
    print("Starting dataset preparation...")

    image_extensions = ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tiff"]
    all_image_paths = []
    for ext in image_extensions:
        all_image_paths.extend(glob.glob(os.path.join(ORIGINAL_IMAGES_DIR, ext)))

    if not all_image_paths:
        print(f"Error: No images found in '{ORIGINAL_IMAGES_DIR}'.")
        return

    print(f"Found {len(all_image_paths)} total images.")

    train_val_paths, test_paths = train_test_split(
        all_image_paths, test_size=0.2, random_state=42
    )
    train_paths, val_paths = train_test_split(
        train_val_paths, test_size=0.25, random_state=42
    )

    print(f"Dataset split:")
    print(f" - Training:   {len(train_paths)} images")
    print(f" - Validation: {len(val_paths)} images")
    print(f" - Testing:    {len(test_paths)} images")

    process_and_save(train_paths, 'train')
    process_and_save(val_paths, 'val')
    process_and_save(test_paths, 'test')

    print("\nDataset preparation complete!")
    print(f"Processed data saved in '{PROCESSED_DATA_DIR}'.")

if __name__ == "__main__":
    main()