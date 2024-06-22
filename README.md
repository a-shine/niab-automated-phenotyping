# AI4ER MRes project 2024 - NIAB automated phenotyping

<table>
  <tr align="center">
    <!-- UKRI Logo -->
    <td align="center">
      <img src=".assets/imgs/readme/logo_ukri_colour.png" alt="UKRI Logo" width="600" />
    </td>
    <!-- University of Cambridge Logo -->
    <td align="center">
      <img src=".assets/imgs/readme/logo_cambridge_colour.jpg" alt="University of Cambridge logo" width="600" />
    </td>
  </tr>
</table>

**Keywords (general to specific)**: Artificial Intelligence (AI), Deep Learning (DL), Computer vision, Semantic segmentation, High-throughput phenotyping, Tef, Weed resilience

## Overview

## Repo structure

* [.assets/](./.assets/): assets such as images used in the README
* [archive/](./archive/): old files that may be useful for reference
* [notebooks/](./notebooks/): main Jupyter notebooks of the project
* [scripts/](./scripts/): scripts such as those used in the image pre/post-processing and for model training
* [utils/](./utils/) utility package with all the util scripts used across notebooks and scripts

## Data

The data used for this project is courtesy of the National Institute of of Agricultural Botany (NIAB).
The raw dataset contains X top down RGB images taken over the course of X weeks from a phenotyping platform.
The crop in question is tef.

The raw dataset is available at:...

As part of this project, 300 images were annotated for use in supervised learning of Deep Learning Semantic segmentation models.
The descriptions of the available annotated datasets is found in the table bellow.

| Dataset name | Description |
|--------------|-------------|
| Raw output from HSV segmentation pipeline | Collection of white balanced images and masks generated directly from the HSV segmentation pipeline. No manual corrections have been made. |
| Partially corrected | Collection of white balanced images and masks generated from the HSV segmentation pipeline with manual noise removal (removing pixels misclassified as shoots). |
| Fully corrected | Collection of white balanced images and masks generated from the HSV segmentation pipeline with manual noise removal and modified annotation to correctly classify pixels as shoot that were previously classified as background. |

We also looked at exploring an active learning approach, where images that generated predictions with high-levels of uncertainty were annotated to improve model performance.
The description of the available annotated datasets can be found in the following table.

| Dataset name | Description |
|--------------|-------------|
| Fully corrected w. active learning | This is the Fully corrected dataset mentioned above (280 images) with the addition of the 20 images in in the remainder of the dataset that introduced the most MC uncertainty. |
| Fully corrected w. random  | This is the Fully corrected dataset mentioned above (280 images) with the addition of 20 images picked at random throughout the remaining dataset. |

Stéphanie Swarbreck. (2024). NIAB teff phenotyping platform [Data set]. Kaggle. https://doi.org/10.34740/KAGGLE/DSV/8750027

Alexandre Shinebourne, and Stéphanie Swarbreck. (2024). Teff shoot semantic segmentation [Data set]. Kaggle. https://doi.org/10.34740/KAGGLE/DSV/8759050


## Installation and usage

To install the required dependencies and source files, follow these steps:

1. Clone the repository onto machine
    ```bash
    git clone https://github.com/a-shine/niab-automated-phenotyping.git
    ```
1. (If using [Anaconda](https://www.anaconda.com/download) or [Miniconda](https://docs.anaconda.com/free/miniconda/index.html) environment manager, create a new environment)
    ```bash
    conda create -n [YOUR_ENV_NAME] python=3.12
    ```
1. From a terminal within the root of the repository, install the required dependencies
    ```bash
    pip install -r requirements.txt
    ```
1. Setup the utils library to use throughout the repository
    ```bash
    pip install -e .
    ```

Note: The code was developed and tested on Python 3.12.3.

To get started we recommend taking a look at [notebooks/012-individual-identifying-algorithm.ipynb](./notebooks/012-individual-identifying-algorithm.ipynb).
This notebook provides a full overview of the pipeline, documenting each step.

## License

The code in this repository is made available for public use under the MIT OpenSource license. For full details see [LICENSE](./LICENSE).

## Acknowledgments
