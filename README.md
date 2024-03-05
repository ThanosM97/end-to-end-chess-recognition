# End-to-End Chess Recognition

This repository hosts the official implementation of our paper "End-to-End Chess Recognition", published at the proceedings of the 19th International Joint Conference on Computer Vision, Imaging and Computer Graphics Theory and Applications (VISIGRAPP 2024).

Paper: [https://arxiv.org/abs/2310.04086](https://arxiv.org/abs/2310.04086).

Abstract:
>Chess recognition refers to the task of identifying the chess pieces configuration from a chessboard image. Contrary to the predominant approach that aims to solve this task through the pipeline of chessboard detection, square localization, and piece classification,
>we rely on the power of deep learning models and introduce two novel methodologies to circumvent this pipeline and directly predict the chessboard configuration from the entire image. In doing so, we avoid the inherent error accumulation of the sequential approaches
>and the need for intermediate annotations. Furthermore, we introduce a new dataset, Chess Recognition Dataset (ChessReD), specifically designed for chess recognition that consists of 10,800 images and their corresponding annotations. In contrast to existing synthetic
>datasets with limited angles, this dataset comprises a diverse collection of real images of chess formations captured from various angles using smartphone cameras; a sensor choice made to ensure real-world applicability. We use this dataset to both train our model and
>evaluate and compare its performance to that of the current state-of-the-art. Our approach in chess recognition on this new benchmark dataset outperforms related approaches, achieving a board recognition accuracy of 15.26% (≈7x better than the current state-of-the-art).


## Requirements
For our experiments we used `Python 3.8.16`. Before using any of the scripts of this repository, please create a virtual environment and install the libraries found in [requirements.txt](requirements.txt). 

## Chess Recognition Dataset (ChessReD)


The Chess Recognition Dataset (ChessReD) is a comprehensive collection of images of chess formations that were captured using various smartphone cameras. It comprises 10,800 images from 100 chess games, split into training, validation, and test sets. The dataset features a wide range of chess piece configurations, captured under different angles and lighting conditions. The dataset includes detailed annotations about the pieces' formations in chess algebraic notation, providing valuable information for chess recognition research. For more information about the data collection and annotation processes you can read our paper.


| Corner view | Player view | Low angle | Top view | 
| -------- | -------- | -------- | -------- |
| <img src="https://github.com/ThanosM97/end-to-end-chess-recognition/assets/41332813/3995a731-28ae-4e9c-b390-33accb038d94" width="200" height="200"> |<img src="https://github.com/ThanosM97/end-to-end-chess-recognition/assets/41332813/7148e84a-f648-4c90-bdc5-66a2f1d6762a" width="200" height="200"> | <img src="https://github.com/ThanosM97/end-to-end-chess-recognition/assets/41332813/72cfad11-5c53-42cc-aee7-d2825e542ea0" width="200" height="200"> | <img src="https://github.com/ThanosM97/end-to-end-chess-recognition/assets/41332813/222c8191-9062-429d-bbe2-b69b658f46d6" width="200" height="200"> | 


<p align="center"> <b>Image samples from ChessReD</b> </p>

### Download
You can manually download the Chess Recognition Dataset (ChessReD) from [here](https://data.4tu.nl/datasets/99b5c721-280b-450b-b058-b2900b69a90f) and extract the images, or you can use the [chessred.py](chessred.py) script as follows:

```
python chessred.py --dataroot 'path/to/save/dataset' --download
```

### Browser
The `chessred.py` script also builds an app to browse ChessReD's images and their annotations, depicted in 2D diagrams. 

```
python chessred.py --dataroot 'path/to/dataset' --browser
```


![ChessReD browser app](https://github.com/ThanosM97/end-to-end-chess-recognition/assets/41332813/0fd35982-52f5-40d1-a9a8-27aecf288938)


## Training

```
$ python train.py -h

usage: train.py [-h] --dataroot DATAROOT --epochs EPOCHS [--nsamples NSAMPLES] [--lr LR] [--decay DECAY] [--topk TOPK]
                [--device {cpu,gpu}] [--ndevices NDEVICES] [--workers WORKERS] [--download]

optional arguments:
  -h, --help           show this help message and exit
  --dataroot DATAROOT  Path to ChessReD data.
  --epochs EPOCHS      Number of epochs to train the model
  --nsamples NSAMPLES  Number of samples per batch.
  --lr LR              Initial learning rate.
  --decay DECAY        Period (epochs) of learning rate decay by a factor of 10.
  --topk TOPK          Number k of top performing model checkpoints to save
  --device {cpu,gpu}
  --ndevices NDEVICES  Number of devices to use for training
  --workers WORKERS    Number of workers to use for data loading
  --download           Download the Chess Recognition Dataset.
```

You can train the model using the [train.py](train.py) script following the implementation details described in our paper, or you can download our [pretrained model](https://drive.google.com/file/d/1sEkIj5MrFncGnmHQt66o_huKjqoMkNQ3/view?usp=drive_link).

## Citation

If you use this code for your research, consider citing our paper.

```
@conference{visapp24,
author={Athanasios Masouris. and Jan {van Gemert}.},
title={End-to-End Chess Recognition},
booktitle={Proceedings of the 19th International Joint Conference on Computer Vision, Imaging and Computer Graphics Theory and Applications - Volume 4: VISAPP},
year={2024},
pages={393-403},
publisher={SciTePress},
organization={INSTICC},
doi={10.5220/0012370200003660},
isbn={978-989-758-679-8},
issn={2184-4321},
}
```
