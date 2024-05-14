import sys
import os

# Add the parent directory of current directory to the sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.image_utils import load_image_ts, reduce_resolution, white_balance
from utils.dl.niab import IMG_TRANSFORMS, ActiveLearningDataset
import torch
import segmentation_models_pytorch as smp
from torch import nn
from utils.dl.model import MCDUNet
from typing import List

dataset = ActiveLearningDataset("./datasets/niab/EXP01/Top_Images/Top_Images_Clean_Rename", IMG_TRANSFORMS)

MODEL_PATH = "./models/best_model.pth"

device = ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Running model using {device} device")

# TODO: let's see if there is any variance (if none then the model might not have dropout layers so we need to add them)
def predict_with_uncertainty(model, image, n_times=10):
    # Set model to training mode to enable dropout at test time
    model.train()

    # List to store predictions
    predictions = []

    with torch.no_grad():
        for _ in range(n_times):
            output = model(image.unsqueeze(0).to(device))
            predictions.append(output)

    # Convert predictions list to tensor
    predictions = torch.stack(predictions)

    # Calculate mean and variance
    mean = torch.mean(predictions, dim=0)
    variance = torch.var(predictions, dim=0)

    return mean, variance


# # Create an instance of the model and move it to the device (GPU or CPU) and load the model parameters
unet_dims: List = []
for i in range(5):
    unet_dims.append(2**(5 + i))

# Create an instance of the model and move it to the device (GPU or CPU)
model = MCDUNet(n_channels=3,
             n_classes=1,
             bilinear=True,
             ddims=unet_dims,
             UQ=True,
             ).to(device)

model.load_state_dict(torch.load(MODEL_PATH))

# TODO: not added dropout layers to the model - need to find a way to add dropout layers to the model
# Test adding dropout with the aux_ parameters (just train trhe model for 5 epochs and test with this script to see if there are any uncertainties i.e. that the model is non-deterministic)

# create a csv with two columns: image_name, uncertainty
csv = open("uncertainty.csv", "w")
csv.write("image_name,uncertainty\n")

# Loop through the dataset and run prediction with uncertainty
for name, image in dataset:
    m, u = predict_with_uncertainty(model, image)

    # Calculate overall uncertainty
    overall_uncertainty = torch.mean(u)

    print(f"Image name: {name}, Overall uncertainty: {overall_uncertainty.item()}")

    csv.write(f"{name},{overall_uncertainty.item()}\n")


# get the first image and run prediction with uncertainty
# name, image = dataset[0]

# print(f"Image name: {name}")

# m, u = predict_with_uncertainty(model, image)

# # TODO: plot an uncertainty mask/map (would be good to do cause it will tell me where the model struggles and also a good sanity check to know that my uncertainty values aren't noncense)

# # Calculate overall uncertainty
# overall_uncertainty = torch.mean(u)

# print(f"Overall uncertainty: {overall_uncertainty.item()}")
