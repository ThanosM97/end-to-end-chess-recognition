"""This module implements the Browser class."""
import argparse
import json
from pathlib import Path
from tkinter import END, messagebox, ttk
from typing import Union

import pandas as pd
from PIL import Image, ImageTk
from ttkthemes import ThemedTk

from utils import download_chessred, extract_zipfile

# Window width for the Browser App
WINDOW_WIDTH = 1280


class Browser:
    """Browser class.

    The Browser class implements an app that allows the user to
    browse the images of the ChessReD, along with their annotations.
    """

    def __init__(self, dataroot: Union[str, Path]) -> None:
        """Initialize Browser.

        Args:
            dataroot (str, Path): Path to the directory containing the Chess
                                  Dataset.
        """
        self.dataroot = dataroot

        # Load annotations
        data_path = Path(dataroot, "annotations.json")
        if not data_path.is_file():
            raise (FileNotFoundError(f"File '{data_path}' doesn't exist."))

        with open(data_path, "r") as f:
            annotations_file = json.load(f)

        # Load tables
        annotations = pd.DataFrame(
            annotations_file["annotations"]['pieces'],
            index=None)
        categories = pd.DataFrame(
            annotations_file["categories"],
            index=None)
        self.images = pd.DataFrame(
            annotations_file["images"],
            index=None)

        # Add category names to annotations
        self.annotations = pd.merge(
            annotations, categories, how="left", left_on="category_id",
            right_on="id")

        # Initialize app window
        self.window = ThemedTk(theme="yaru")
        self.window.title('Chess Recognition Dataset (ChessReD) Browser')
        self.window.iconbitmap("resources/pieces/icon.ico")
        self.window_width = WINDOW_WIDTH
        self.window.resizable(False, False)

        # Starting image
        self.current_image_id = 0

        # Helper lambda functions
        self.find_image_path = lambda x: Path(self.images[self.images['id']
                                                          == x].path.values[0])
        self.load_image = lambda x: ImageTk.PhotoImage(Image.open(x).resize(
            (self.window_width // 2, self.window_width // 2)))

        # Build UI
        self.build_ui(self.current_image_id)

        # Open window
        self.window.mainloop()

    def build_ui(self, image_number: Union[int, str]) -> None:
        """Build the UI of the Browser app.

        Args:
            image_number (int, str): ID of the dataset image to be displayed.
        """
        # Check if input not integer
        if not isinstance(image_number, int):
            try:
                image_number = int(image_number)
            except ValueError:
                messagebox.showerror(
                    "Invalid input", "You can only input integer image IDs.")
                image_number = self.current_image_id

        # Check whether input ID is within the dataset's IDs
        if image_number > len(self.images) or image_number < 0:
            messagebox.showwarning(
                "Invalid ID",
                f"Image with id:{image_number} not found.")
            image_number = self.current_image_id

        # Load chessred image
        self.current_image_path = self.find_image_path(image_number)
        self.current_image = self.load_image(
            self.dataroot / self.current_image_path)

        # Create an image of a 2D chess set from the annotations
        self.current_2Dimage = ImageTk.PhotoImage(
            self.create2D(image_number))

        self.current_image_id = image_number

        # build widgets with new images
        if not hasattr(self, "my_label"):
            self.build_widgets(image_number)
        else:
            self.rebuild_widgets(image_number)

    def build_widgets(self, image_number: int) -> None:
        """Initialize window widgets.

        Args:
            image_number (int): ID of the dataset image to be displayed.

        Returns:
            None
        """
        # Add "previous" button
        self.button_previous = ttk.Button(
            self.window, text="Previous", command=lambda: self.build_ui(
                image_number - 1))
        self.button_previous.grid(
            row=1, column=0, columnspan=2, sticky="nsew")
        # Starting image is 0, so there is no previous
        self.button_previous["state"] = "disabled"

        # Add text input field
        self.number_input = ttk.Entry(self.window)
        self.number_input.grid(row=1, column=4, sticky="nsew")

        # Add current image id to text input field
        self.number_input.insert(0, str(image_number))

        # Add "find" button
        self.button_find = ttk.Button(
            self.window, text="Find image", command=lambda: self.build_ui(
                self.number_input.get()))
        self.button_find.grid(row=1, column=5, sticky="nsew")

        # Add "next" button
        self.button_next = ttk.Button(
            self.window, text="Next", command=lambda: self.build_ui(
                image_number + 1))
        self.button_next.grid(row=1, column=8, columnspan=2, sticky="nsew")
        # Bind right key to "next image"
        self.window.bind("<Right>", lambda _: self.build_ui(
            image_number + 1))

        # Insert images to window
        self.insert_images()

    def rebuild_widgets(self, image_number: int) -> None:
        """Rebuild window widgets with new content.

        Args:
            image_number (int): ID of the dataset image to be displayed.
        Returns:
            None
        """
        # Remove old images
        self.my_label.destroy()
        self.my_label2D.destroy()

        # Configure "previous" button to select the correct image on click
        self.button_previous.configure(command=lambda: self.build_ui(
            image_number - 1))
        # Disable button and left key if there is no previous image
        if image_number-1 < 0:
            self.button_previous["state"] = "disabled"
            self.window.unbind("<Left>")
        else:
            self.button_previous["state"] = "enabled"
            self.window.bind("<Left>", lambda _: self.build_ui(
                image_number - 1))

        # Clear input contents
        self.number_input.delete(0, END)
        # Add current image id
        self.number_input.insert(0, str(image_number))

        # Configure "next" button to select the correct image on click
        self.button_next.configure(command=lambda: self.build_ui(
            image_number + 1))
        # Disable button and right key if there is no next image
        if image_number+1 > len(self.images) - 1:
            self.button_next["state"] = "disabled"
            self.window.unbind("<Right>")
        else:
            self.button_next["state"] = "enabled"
            self.window.bind("<Right>", lambda _: self.build_ui(
                image_number + 1))

        # Insert images to window
        self.insert_images()

    def insert_images(self) -> None:
        """Add dataset and annotation images to window."""
        # Add chessred image to window
        self.my_label = ttk.Label(image=self.current_image)
        self.my_label.grid(row=0, column=0, columnspan=5)

        # Add annotations image to window
        self.my_label2D = ttk.Label(image=self.current_2Dimage)
        self.my_label2D.grid(row=0, column=5, columnspan=5)

    def create2D(self, image_number) -> 'Image.Image':
        """Create a PIL Image of a 2D chess set.

        Create a synthetic PIL image of a 2D chess set based on the
        annotations for dataset image with id `image_number`.

        Args:
            image_number (int): ID of the dataset image to be displayed.
        Returns:
            A synthetic PIL.Image of a 2D chess set.
        """
        # Get the positions for pieces in image `image_number`
        positions = list(
            self.annotations
            [self.annotations["image_id"] == image_number]
            ["chessboard_position"])
        # Get the corresponding piece classes
        categories = list(
            self.annotations
            [self.annotations["image_id"] == image_number]["name"])

        # Naming convention of the rows and columns on the chessboard
        cols = "abcdefgh"
        rows = "87654321"

        # Open default chessboard background image
        board = Image.open("./resources/board.png").resize(
            (self.window_width//2, self.window_width//2))
        piece_size = self.window_width // 16

        for piece, pos in zip(categories, positions):
            piece_png = Image.open(f"./resources/pieces/{piece}.png").resize(
                (piece_size, piece_size))

            j = cols.index(pos[0])
            i = rows.index(pos[1])
            board.paste(piece_png, (j*piece_size, i *
                        piece_size), mask=piece_png)

        return board


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataroot', required=True,
                        help="Path to ChessReD data.")

    parser.add_argument('--browser', action="store_true",
                        help='Run ChessReD browser app.')

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

    if args.browser:
        if (Path(dataroot/'annotations.json').exists()
                and Path(dataroot/'images').exists()):
            Browser(dataroot=args.dataroot)
        else:
            raise FileNotFoundError(
                "Could not find the ChessReD data at the path " +
                f"{dataroot}. Either use the '--download' flag, or" +
                " manually download the dataset from https://data.4tu.nl" +
                "/datasets/99b5c721-280b-450b-b058-b2900b69a90f .")
