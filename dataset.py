"""This module implements the ChessRecognitionDataset class."""
import json
from pathlib import Path
from typing import Callable, Tuple, Union

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.io import read_image


class ChessRecognitionDataset(Dataset):
    """ChessRecognitionDataset class.

    The ChessRecognitionDataset class implements a custom pytorch dataset.
    """

    def __init__(self,
                 dataroot: Union[str, Path],
                 split: str,
                 transform: Union[Callable, None] = None) -> None:
        """Initialize a ChessRecognitionDataset.

        Args:
            dataroot (str, Path): Path to the directory containing the Chess
                                  Dataset.
            transform (callable, optional): Transform to be applied on the
                                            image samples of the dataset.
        """
        super(ChessRecognitionDataset, self).__init__()

        self.dataroot = dataroot
        self.split = split
        self.transform = transform

        # Load annotations
        data_path = Path(dataroot, "annotations.json")
        if not data_path.is_file():
            raise (FileNotFoundError(f"File '{data_path}' doesn't exist."))

        with open(data_path, "r") as f:
            annotations_file = json.load(f)

        # Load tables
        self.annotations = pd.DataFrame(
            annotations_file["annotations"]['pieces'],
            index=None)
        self.categories = pd.DataFrame(
            annotations_file["categories"],
            index=None)
        self.images = pd.DataFrame(
            annotations_file["images"],
            index=None)

        # Get split info
        self.length = annotations_file['splits'][split]['n_samples']
        self.split_img_ids = annotations_file['splits'][split]['image_ids']

        # Keep only the split's data
        self.annotations = self.annotations[self.annotations["image_id"].isin(
            self.split_img_ids)]
        self.images = self.images[self.images['id'].isin(self.split_img_ids)]

        assert (self.length == len(self.split_img_ids) and
                self.length == len(self.images)), (
            f"The numeber of images in "
            f"the dataset ({len(self.images)}) for split:{self.split}, does "
            f"not match neither the length specified in the annotations "
            f"({self.length}) or the length of the list of ids for the split "
            f"{len(self.split_img_ids)}")

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return self.length

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return a dataset sample.

        Args:
            index (int): Index of the sample to return.

        Returns:
            img (Tensor): A 3xHxW Tensor of the image corresponding to `index`.
            img_anns (Tensor): A 64x13 Tensor containing the annotations for
                               each of the chessboard's squares in one-hot
                               encoding.
        """
        # LOAD IMAGE
        img_id = self.split_img_ids[index]
        img_path = Path(
            self.dataroot,
            self.images[self.images['id'] == img_id].path.values[0])

        img = read_image(str(img_path)).float()

        if self.transform is not None:
            img = self.transform(img)

        # GET ANNOTATIONS
        cols = "abcdefgh"
        rows = "87654321"

        empty_cat_id = int(
            self.categories[self.categories['name'] == 'empty'].id.values[0])

        img_anns = self.annotations[
            self.annotations['image_id'] == img_id].copy()

        # Convert chessboard positions to 64x1 array indexes
        img_anns['array_pos'] = img_anns["chessboard_position"].map(
            lambda x: 8*rows.index(x[1]) + cols.index(x[0]))

        # Keep columns of interest
        img_anns = pd.DataFrame(
            img_anns['category_id']).set_index(img_anns['array_pos'])

        # Add category_id for 'empty' in missing row indexes and create tensor
        img_anns = torch.tensor(list(img_anns.reindex(
            range(64), fill_value=empty_cat_id)['category_id'].values))

        img_anns = F.one_hot(img_anns)
        img_anns = img_anns.flatten().float()

        return (img, img_anns)
