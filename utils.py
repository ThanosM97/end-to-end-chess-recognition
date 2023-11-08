"""This module implements util functions."""
import zipfile
from pathlib import Path

import torch
import wget
import yaml
from tqdm import tqdm


def progress_bar(
        current: int, total: int, width: int = 80, name: str = '') -> None:
    """Creates a custom progress bar.


    Args:
        current (int): Current number of downloaded bytes.
        total (int): Total number of bytes.
        width (int, optional): Width of the bar.
        name (str, optional): Name of the object being downloaded.

    Returns:
        None
    """
    file_size_gb = total / (1024**3)
    current_size_gb = current / (1024**3)
    print(
        f"\tDownloading {name}: {int(current / total * 100)}% ",
        f"[{current_size_gb:.2f} / {file_size_gb:.2f}] GB", end='\r')


def download_chessred(dataroot: str) -> None:
    """Downloads the ChessReD dataset.

    Args:
        dataroot (str): Path to the directory to save ChessReD.

    Returns:
        None
    """

    with open('cfg/chessred.yaml', 'r') as f:
        chessred_yaml = yaml.safe_load(f)

    print("Downloading Chess Recognition Dataset (ChessReD)...")

    url_json = chessred_yaml['annotations']['url']
    wget.download(url_json, Path(
        dataroot, 'annotations.json').as_posix(),
        bar=lambda *args: progress_bar(*args, "annotations"))

    print()

    url_images = chessred_yaml['images']['url']
    wget.download(url_images, Path(
        dataroot, 'images.zip').as_posix(),
        bar=lambda *args: progress_bar(*args, "images"))

    print("\nDownload completed.")


def extract_zipfile(zip_file: str, output_directory: str) -> None:
    """Extracts `zip_file` to `output_directory`.

    Args:
        zip_file (str): Path to zipfile to extract.
        output_directory (str): Path to extract `zipfile`.

    Returns:
        None
    """
    print(f"Exracting ChessReD images at {output_directory}...")
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        # Get the list of files in the ZIP file
        file_list = zip_ref.namelist()
        # Create a progress bar using tqdm
        for file_name in tqdm(file_list, desc="Extracting", unit="file"):
            zip_ref.extract(file_name, output_directory)

    print("Extraction completed.")


def recognition_accuracy(
        y: torch.Tensor,
        preds: torch.Tensor,
        tolerance: int = 0) -> torch.Tensor:
    """Returns the chess recognition accuracy with given `tolerance`.

    Args:
        y (Tensor): Ground truth labels.
        preds (Tensor): Model predictions.
        tolerance (int): Allowed mistakes per board. (Default: 0)
    """
    correct = ((preds == y).sum(axis=1) > 63-tolerance).sum()

    return correct / preds.shape[0]
