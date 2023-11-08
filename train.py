"""This module implements a modified ResNeXt network for chess recognition."""
import argparse
from pathlib import Path
from typing import Tuple

import pytorch_lightning as pl
import torch
import torchvision.models as models
from pytorch_lightning.callbacks import ModelCheckpoint
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import ChessRecognitionDataset
from utils import download_chessred, extract_zipfile, recognition_accuracy

pl.seed_everything(42, workers=True)


class ChessDataModule(pl.LightningDataModule):
    """PyTorch Lightning data module for ChessReD.

    Args:
        dataroot (str): Path to ChessReD directory.
        batch_size (int): Number of samples per batch.
    """

    def __init__(self, dataroot: str, batch_size: int, workers: int) -> None:
        """
        Args:
            dataroot (str): Path to ChessReD directory.
            batch_size (int): Number of samples per batch.
            workers (int): Number of workers for dataloading.
        """
        super().__init__()
        self.dataroot = dataroot
        self.transform = transforms.Compose([
            transforms.Resize(1024, antialias=None),
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.47225544, 0.51124555, 0.55296206],
                std=[0.27787283, 0.27054584, 0.27802786]),
        ])

        self.batch_size = batch_size
        self.workers = workers

    def setup(self, stage: str) -> None:
        """PyTorch Lightning required method to setup the dataset at `stage`.

        Args:
            stage (str): Stage at which the data module is loaded.
        """
        if stage == "fit":
            self.chess_train = ChessRecognitionDataset(
                dataroot=self.dataroot,
                split="train", transform=self.transform)

            self.chess_val = ChessRecognitionDataset(
                dataroot=self.dataroot,
                split="val", transform=self.transform)

        if stage == "test":
            self.chess_test = ChessRecognitionDataset(
                dataroot=self.dataroot,
                split="test", transform=self.transform)

        if stage == "predict":
            self.chess_predict = ChessRecognitionDataset(
                dataroot=self.dataroot,
                split="test", transform=self.transform)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.chess_train, batch_size=self.batch_size,
            num_workers=self.workers)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.chess_val, batch_size=self.batch_size,
            num_workers=self.workers)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.chess_test, batch_size=self.batch_size,
            num_workers=self.workers)

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(
            self.chess_predict, batch_size=self.batch_size,
            num_workers=self.workers)


class ChessResNeXt(pl.LightningModule):
    """Modified ResNeXt network for chess recognition on ChessReD.

    This class implements a modified ResNeXt network for chess recognition.
    The top-level classifier of ResNeXt101 is replaced with a linear layer
    of 64x13 outputs, where 64 are the squares of the chessboard and 13 the
    number of possible classes to classify each square (6 piece types per
    color and empty square).
    """

    def __init__(self) -> None:
        """Initializes ChessResNeXt."""
        super().__init__()

        backbone = models.resnext101_32x8d(weights="DEFAULT")
        num_filters = backbone.fc.in_features
        layers = list(backbone.children())[:-1]

        self.feature_extractor = nn.Sequential(*layers)

        num_target_classes = 64 * 13

        self.classifier = nn.Linear(num_filters, num_target_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_extractor(x).flatten(1)
        x = self.classifier(x)

        return x

    def cross_entropy_loss(
            self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(logits, labels)

    def common_step(
            self,
            batch: Tuple[torch.Tensor, torch.Tensor],
            batch_idx: int,
            return_accuracy: bool = False) -> torch.Tensor:

        x, y = batch
        logits = self.forward(x)

        if return_accuracy:
            y_cat = torch.argmax(y.reshape((-1, 64, 13)), dim=2)
            preds_cat = torch.argmax(logits.reshape((-1, 64, 13)), dim=2)

            return (self.cross_entropy_loss(logits, y),
                    recognition_accuracy(y_cat, preds_cat))

        return self.cross_entropy_loss(logits, y)

    def training_step(self,
                      train_batch: Tuple[torch.Tensor, torch.Tensor],
                      batch_idx: int) -> torch.Tensor:

        loss = self.common_step(train_batch, batch_idx)
        self.log('train_loss', loss)

        return loss

    def validation_step(self,
                        val_batch: Tuple[torch.Tensor, torch.Tensor],
                        batch_idx: int) -> None:

        x, y = val_batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)

        loss, accuracy = self.common_step(
            val_batch, batch_idx, return_accuracy=True)

        self.log('val_loss', loss)
        self.log('val_acc', accuracy)

    def test_step(self,
                  test_batch: Tuple[torch.Tensor, torch.Tensor],
                  batch_idx: int) -> None:

        loss = self.common_step(test_batch, batch_idx)
        self.log('test_loss', loss)

    def predict_step(
            self,
            predict_batch: Tuple[torch.Tensor, torch.Tensor],
            batch_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:

        x, y = predict_batch
        logits = self.forward(x)

        return (logits, y)

    def configure_optimizers(self) -> torch.optim.Adam:
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


def main(args):
    data_module = ChessDataModule(
        args.dataroot, args.nsamples, args.workers)

    model = ChessResNeXt()
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        save_last=True,
        mode="min",
        save_top_k=args.topk,
        filename="model_{epoch:02d}-{val_loss:.4f}",
        save_weights_only=False
    )

    trainer = pl.Trainer(accelerator=args.device, devices=args.ndevices,
                         deterministic=True, max_epochs=args.epochs,
                         callbacks=[checkpoint_callback])
    trainer.fit(model, data_module)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataroot', required=True,
                        help="Path to ChessReD data.")

    parser.add_argument('--epochs',
                        help="Number of epochs to train the model",
                        required=True)

    parser.add_argument('--nsamples',
                        help="Number of samples per batch.",
                        default=8)

    parser.add_argument('--topk',
                        help=("Number k of top performing" +
                              " model checkpoints to save"),
                        default=3)

    parser.add_argument('--device',
                        choices=["cpu", "gpu"],
                        default="gpu")

    parser.add_argument('--ndevices',
                        help="Number of devices to use for training",
                        default=1)

    parser.add_argument('--workers',
                        help="Number of workers to use for data loading",
                        default=4)

    parser.add_argument('--download', action="store_true",
                        help='Download the Chess Recognition Dataset.')

    args = parser.parse_args()

    dataroot = Path(args.dataroot)
    dataroot.mkdir(parents=True, exist_ok=True)

    if args.download:
        download_chessred(dataroot)

        zip_path = dataroot/"images.zip"
        print()

        extract_zipfile(zip_file=zip_path, output_directory=dataroot)

        # Remove zip file
        zip_path.unlink(True)

    main(args)
