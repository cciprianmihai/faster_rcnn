import os
import shutil
import gdown
import zipfile
from pathlib import Path
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def download_and_extract(url, dest_folder, zip_name):
    # Check if the folder already exists and contains files
    extracted_folder = os.path.join(dest_folder, os.path.splitext(zip_name)[0])  # Remove .zip extension

    # Check if the extracted folder exists and is not empty
    if os.path.exists(extracted_folder) and os.listdir(extracted_folder):
        print(f"Folder '{extracted_folder}' already exists and is not empty. Skipping download.")
        return
    
    os.makedirs(dest_folder, exist_ok=True)

    # Download the file using gdown
    zip_file_path = os.path.join(dest_folder, zip_name)
    gdown.download(url, zip_file_path, quiet=True)
    
    # Check if the file was downloaded and unzip it
    if os.path.exists(zip_file_path):
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(dest_folder)
        print(f"Unzipped {zip_name} successfully to {dest_folder}")
        
        # Remove the zip file after extraction
        try:
            os.remove(zip_file_path)
            print(f"Removed {zip_name} successfully.")
        except FileNotFoundError:
            print(f"File {zip_file_path} not found for removal.")
    else:
        print(f"File {zip_file_path} not found. Download might have failed.")


def safe_rename(old_path, new_path):
    old_path, new_path = Path(old_path), Path(new_path)
    if old_path.exists():
        # Copy the entire folder
        shutil.copytree(old_path, new_path, dirs_exist_ok=True)
        print(f"Copied '{old_path}' to '{new_path}'")

        # Delete the old folder after copying
        shutil.rmtree(old_path)
        print(f"Deleted old folder: {old_path}")
    else:
        print(f"Error: The folder '{old_path}' does not exist.")


def separate_files_respecting_txt(input_dir, output_dir, txt_file):
    """
    Extracts files from a source folder to a destination folder, 
    based on file names listed in a text file.

    Parameters:
    source_folder (str): The path to the source folder containing files.
    destination_folder (str): The path to the destination folder where files should be copied.
    txt_file (str): The path to the text file listing the files to extract.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Read the text file to get the list of target files
    with open(txt_file, 'r') as file:
        target_files = file.read().splitlines()

    # Loop through each file name and copy it if it exists in the source folder
    for file_name in target_files:
        source_file_path = os.path.join(input_dir, file_name)
        if os.path.exists(source_file_path):
            shutil.copy(source_file_path, output_dir)


def extract_and_store_data_from_datasets(data_dir, dataset_names):
    for dataset_name in dataset_names:
        dataset_dir = os.path.join(data_dir, dataset_name)
        if os.path.exists(dataset_dir):
            print(f"Dataset {dataset_name} already exists")
            continue

        all_img_dir = os.path.join(dataset_dir, "images")
        all_mask_dir = os.path.join(dataset_dir, "masks")

        splitted_data_dir = os.path.join(dataset_dir, "splitted_data")

        train_img_dir = os.path.join(splitted_data_dir, "train", "images")
        train_mask_dir = os.path.join(splitted_data_dir, "train", "masks")
        val_img_dir = os.path.join(splitted_data_dir,  "val", "images")
        val_mask_dir = os.path.join(splitted_data_dir,  "val", "masks")
        test_img_dir = os.path.join(splitted_data_dir,  "test", "images")
        test_mask_dir = os.path.join(splitted_data_dir,  "test", "masks")


        if dataset_name == "brats":
            file_id = "1zzX792C87wJ-WQtJvQGjGlPQ2UziJBHk"
            gdown_url = f"https://drive.google.com/uc?id={file_id}"
            download_and_extract(gdown_url, data_dir, "brats.zip")

        elif dataset_name == "bmshare":
            file_id = "1DTr-m3tlomBd8ypIfsmfj9uQn4JZj83U"
            gdown_url = f"https://drive.google.com/uc?id={file_id}"

            download_and_extract(gdown_url, data_dir, "bmshare.zip")

        elif dataset_name == "isles":
            file_id = "1ec6pxkEQ_gDQWBqjOIbZhXga9ctzX7Du"
            gdown_url = f"https://drive.google.com/uc?id={file_id}"

            download_and_extract(gdown_url, data_dir, "isles.zip")

        else:
            raise ValueError("Dataset not supported")


        train_instruction_txt = os.path.join(dataset_dir, "train.txt")
        val_instruction_txt = os.path.join(dataset_dir, "val.txt")
        test_instruction_txt = os.path.join(dataset_dir, "test.txt")

        separate_files_respecting_txt(all_img_dir, val_img_dir, val_instruction_txt)
        separate_files_respecting_txt(all_mask_dir, val_mask_dir, val_instruction_txt)
        separate_files_respecting_txt(all_img_dir, train_img_dir, train_instruction_txt)
        separate_files_respecting_txt(all_mask_dir, train_mask_dir, train_instruction_txt)
        separate_files_respecting_txt(all_img_dir, test_img_dir, test_instruction_txt)
        separate_files_respecting_txt(all_mask_dir, test_mask_dir, test_instruction_txt)


def get_model(num_classes):
    # Load pre-trained model
    model = fasterrcnn_resnet50_fpn_v2(weights="DEFAULT")

    # Replace classifier head
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model

