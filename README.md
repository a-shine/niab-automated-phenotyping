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

Note: The code was developed and tested on Python 3.12.3.

To get started we recommend taking a look at [notebooks/012-individual-identifying-algorithm.ipynb](./notebooks/012-individual-identifying-algorithm.ipynb).
This notebook provides a full overview of the pipeline, documenting each step.

## License

The code in this repository is made available for public use under the MIT OpenSource license. For full details see [LICENSE](./LICENSE).

## Acknowledgments
