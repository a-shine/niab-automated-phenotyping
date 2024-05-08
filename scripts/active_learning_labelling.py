import sys
import os

# Add the parent directory of current directory to the sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.image_utils import load_image_ts, reduce_resolution, white_balance
from utils.dl.niab import IMG_TRANSFORMS, ActiveLearningDataset
import torch
import segmentation_models_pytorch as smp
from torch import nn

dataset = ActiveLearningDataset("./datasets/niab/EXP01/Top_Images/Top_Images_Clean_Rename", IMG_TRANSFORMS)

MODEL_PATH = "./models/best_model_unet_dropout_05_08.pth"

device = ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Running model using {device} device")

# TODO: let's see if there is any variance (if none then the model might not have dropout layers so we need to add them)
def predict_with_uncertainty(model, image, n_times=10):
    # Set model to training mode to enable dropout at test time
    model.train()

    # List to store predictions
    predictions = []

    for _ in range(n_times):
        with torch.no_grad():
            output = model(image.unsqueeze(0).to(device))
            predictions.append(output)

    # Convert predictions list to tensor
    predictions = torch.stack(predictions)

    # Calculate mean and variance
    mean = torch.mean(predictions, dim=0)
    variance = torch.var(predictions, dim=0)

    return mean, variance


# # Create an instance of the model and move it to the device (GPU or CPU) and load the model parameters
model = smp.Unet(
    encoder_name="resnet34",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
    in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    classes=1,                      # model output channels (number of classes in your dataset)
).to(device)
model.load_state_dict(torch.load(MODEL_PATH))

# TODO: not added dropout layers to the model - need to find a way to add dropout layers to the model
# Test adding dropout with the aux_ parameters (just train trhe model for 5 epochs and test with this script to see if there are any uncertainties i.e. that the model is non-deterministic)

# Add dropout layers
for name, child in model.named_children():
    if isinstance(child, nn.Sequential):
        print(f"Adding dropout layer after {name}")
        for name2, child2 in child.named_children():
            if isinstance(child2, nn.ReLU):
                print(f"Adding dropout layer after {name2}")
                setattr(child, name2, nn.Sequential(child2, nn.Dropout(p=0.5)))

# get the first image and run prediction with uncertainty
image = dataset[0]

m, u = predict_with_uncertainty(model, image)

# Calculate overall uncertainty
overall_uncertainty = torch.mean(u)

print(f"Overall uncertainty: {overall_uncertainty.item()}")
