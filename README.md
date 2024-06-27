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

## Abstract

<!-- TODO -->

**Keywords (general to specific)**: Artificial Intelligence (AI), Deep Learning (DL), Computer vision, Semantic segmentation, High-throughput phenotyping, Teff, Weed resilience

## Repo structure

* [.assets/](./.assets/): Assets such as images used in the README and notebooks
* [archive/](./archive/): Unused files that may be useful for reference
* [notebooks/](./notebooks/): Main Jupyter notebooks for the project
  *  [011-identifying-pot-rim.ipynb](./notebooks/011-identifying-pot-rim.ipynb): Pot rim tracking using the Circle Hough Transform algorithm
  * **[012-canopy-coverage-tracking.ipynb](./notebooks/012-canopy-coverage-tracking.ipynb)**: Complete canopy coverage tracking phenotyping pipeline
  * [013-visualising-uncertainty.ipynb](./notebooks/013-visualising-uncertainty.ipynb): Visualise uncertainty distribution and heatmaps for images in the dataset
  * [015-semantic-instance-model.ipynb](./notebooks/015-semantic-instance-model.ipynb): Early experiments using Discriminative loss function for semantic instance segmentation
  * [016-shoot-following-algo.ipynb](./notebooks/016-shoot-following-algo.ipynb): Shoot following algorithm based on kernel convolutions for end, crossing and branching point classification
  * [017-qualitative-segmentation-output.ipynb](./notebooks/017-qualitative-segmentation-output.ipynb): Side-by-side comparison of teff shoot semantic segmentation model output
* [scripts/](./scripts/): scripts such as those used in the image pre/post-processing and for model training
* [utils/](./utils/) utility package with all the util scripts used across notebooks and scripts

## Data

The data used for this project is courtesy of the National Institute of of Agricultural Botany (NIAB).
The raw dataset contains 1120 top down RGB images taken over the course of several weeks from a phenotyping platform.
The crop in question is teff.

The full raw dataset is available at [Stéphanie Swarbreck. (2024). NIAB teff phenotyping platform [Data set]. Kaggle. https://doi.org/10.34740/KAGGLE/DSV/8750027](https://doi.org/10.34740/KAGGLE/DSV/8750027).

### Semantic segmentation

As part of this project, 280 images were annotated for use in supervised learning of Deep Learning semantic segmentation models.
The annotated dataset has been made available at [Alexandre Shinebourne, and Stéphanie Swarbreck. (2024). Teff shoot semantic segmentation [Data set]. Kaggle. https://doi.org/10.34740/KAGGLE/DSV/8759050](https://doi.org/10.34740/KAGGLE/DSV/8759050).
The descriptions of the available annotated datasets is found in the table bellow.

| Dataset name | Description |
|--------------|-------------|
| `Base_Training/Raw_HSV` | White balanced images and masks generated directly from the HSV segmentation pipeline. No manual corrections have been made. |
| `Base_Training/Partially_Corrected` | White balanced images and masks generated from the HSV segmentation pipeline with manual noise removal (removing pixels misclassified as shoots). |
| `Base_Training/Fully_Corrected` | White balanced images and masks generated from the HSV segmentation pipeline with manual noise removal and modified annotation to correctly classify pixels as shoot that were previously classified as background. |

We also explored an active learning approach, where images that generated predictions with high-levels of uncertainty were annotated to augment the training dataset to maximise annotation budget value.
The description of the available annotated datasets can be found in the following table.

| Dataset name | Description |
|--------------|-------------|
| `Active_Learning/MC_Uncertainty` | `Base_Training/Fully_Corrected` dataset mentioned above with the addition of the 10 images that introduced the most MC dropout uncertainty. |
| 3 sets of  `Active_Learning/Random_XX` | `Base_Training/Fully_Corrected` dataset mentioned above with the addition of 10 images picked at random throughout the remaining dataset. |

A portion of the annotated dataset has been reserved to benchmark model performance and is found in the `Test/` directory.

### Shoot canopy coverage tracking

In addition to developing a Deep Learning model for semantic segmentation of teff shoots, as part of this project, we also used the model in a complete phenotyping pipeline to track teff shoot canopy coverage over time.

A time series can be loaded from the full phenotyping platform dataset with the following utility code:

```python
from utils.image_utils import load_image_ts

raw_images = load_image_ts(dataset_root=[PATH_TO_DATASET/niab], exp=1, block=[BLOCK_NUMBER], pot=[POT_NUMBER])
```

Sample data needed to generate the figures for report are included in the repository `sample_data/` directory.

## Installation and usage

To get started we recommend taking a look at [notebooks/012-canopy-coverage-tracking.ipynb](./notebooks/012-canopy-coverage-tracking.ipynb).
This notebook provides a full overview of the phenotyping pipeline, documenting each step.
A hosted version of notebook [notebooks/012-canopy-coverage-tracking.ipynb](./notebooks/012-canopy-coverage-tracking.ipynb) is available on [Kaggle](https://www.kaggle.com/code/alexandreshinebourne/012-canopy-coverage-tracking-kaggle-version).

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

## License

The code in this repository is made available for public use under the MIT OpenSource license. For full details see [LICENSE](./LICENSE).
