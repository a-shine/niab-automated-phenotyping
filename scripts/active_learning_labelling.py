"""
Batch script that runs inference with uncertainty on the images in the dataset
and writes the results to a csv file. The script uses the MC dropout U-Net
model to predict the segmentation mask and calculate the uncertainty values.

The script uses the ActiveLearningDataset class to load the images from the
dataset and the IMG_TRANSFORMS_NO_JITTER to apply the same transformations used
during training.

The script uses the predict_with_uncertainty function to run the model multiple
times and calculate the uncertainty values.

The script writes the results to a csv file with the following columns:
- img_path: The path of the image
- mean_uncertainty: The mean uncertainty per pixel
- mcd_uncertainty: The MCD Uncertainty value

The script uses the following parameters:
- DATA_DIR: The path to the dataset directory
- MODEL_PATH: The path to the model weights
- THRESHOLD: The threshold to binarize the output mask

Example:
    python active_learning_labelling.py

Note: The script assumes that the dataset directory contains images in the
same format as the NIAB dataset and that the model weights are saved in the
same format as the MC dropout U-Net model.
"""

import torch

from utils.dl.dataset import IMG_TRANSFORMS_NO_JITTER, ActiveLearningDataset
from utils.dl.mc_dropout_uncertainty import predict_with_uncertainty
from utils.dl.models.mcd_unet import MCDUNet

DATA_DIR = "/home/users/ashine/gws/niab-automated-phenotyping/datasets/niab/EXP01/Top_Images/Top_Images_Clean_Rename"
MODEL_PATH = "/home/users/ashine/gws/niab-automated-phenotyping/models/20240605150017/best_model.pth"
THRESHOLD = 0.5  # Threshold to binarize the output mask


niab_dataset = ActiveLearningDataset(DATA_DIR, IMG_TRANSFORMS_NO_JITTER)

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Running model using {device} device")

# Create an instance of the model and move it to the device (GPU or CPU)
model = MCDUNet(
    n_channels=3,
    n_classes=1,
    bilinear=True,
    ddims=[32, 64, 128, 256, 512],
    UQ=True,
    activation=True,
).to(device)

# Load the model weights
model.load_state_dict(torch.load(MODEL_PATH))

# Create a csv file to store the uncertainty values
csv = open("/home/users/ashine/gws/niab-automated-phenotyping/uncertainty.csv", "w")
csv.write("img_path,mean_uncertainty,mcd_uncertainty\n")

# Loop through the dataset and run prediction with uncertainty
for name, image in niab_dataset:
    m, u = predict_with_uncertainty(model, image, device, n_times=500)

    # Calculate mean uncertainty per pixel (useful if you want to know how
    # uncertain the model is on average for each pixel)
    mean_uncertainty = torch.mean(u)

    # Compute MCD Uncertainty - Global score of how certain or uncertain the
    # model is given an input image, which can be useful if you want to get a
    # single uncertainty value for the entire image.

    mask = m > THRESHOLD

    mcd_uncertainty = (
        (torch.sum(u[mask]) / torch.sum(mask)).item() if torch.sum(mask) > 0 else 0.0
    )

    print(
        f"Image name: {name}, Mean uncertainty: {mean_uncertainty.item()}, MCD Uncertainty: {mcd_uncertainty}"
    )

    # Write the results to the csv file
    csv.write(f"{name},{mean_uncertainty.item()},{mcd_uncertainty}\n")
