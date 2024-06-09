# Batch script that runes inference with uncertainty on the images in the 
# dataset.

from utils.dl.niab import IMG_TRANSFORMS_NO_JITTER, ActiveLearningDataset
import torch
from utils.dl.model import MCDUNet

niab_dataset = ActiveLearningDataset("/home/users/ashine/gws/niab-automated-phenotyping/datasets/niab/EXP01/Top_Images/Top_Images_Clean_Rename", IMG_TRANSFORMS_NO_JITTER)

MODEL_PATH = "/home/users/ashine/gws/niab-automated-phenotyping/models/20240605150017/best_model.pth"

device = ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Running model using {device} device")

def predict_with_uncertainty(model, image, n_times=500):
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

# Create an instance of the model and move it to the device (GPU or CPU)
model = MCDUNet(n_channels=3,
             n_classes=1,
             bilinear=True,
             ddims=[32, 64, 128, 256, 512],
             UQ=True,
             activation=True,
             ).to(device)

model.load_state_dict(torch.load(MODEL_PATH))

# create a csv with two columns: image_name, uncertainty
csv = open("/home/users/ashine/gws/niab-automated-phenotyping/uncertainty.csv", "w")
csv.write("img_path,mean_uncertainty,mcd_uncertainty\n")

# Loop through the dataset and run prediction with uncertainty
for name, image in niab_dataset:
    # print(f"Processing image: {name}")
    m, u = predict_with_uncertainty(model, image)

    # Calculate mean uncertainty per pixel (useful if you want to know how uncertain the model is on average for each pixel)
    mean_uncertainty = torch.mean(u)
    
    # The original definition of MCD uncertainty involves normalizing by the volume 
    # of the predicted mask, not the total number of pixels in the image. If the 
    # predicted mask only covers a portion of the image, then the MCD uncertainty 
    # would be a measure of average uncertainty within the mask, not the entire 
    # image.

    # If you want to calculate MCD uncertainty, you would need to sum and normalize
    # only over the pixels within the predicted mask. This would give you a 
    # different measure of uncertainty that might be more relevant if you're 
    # specifically interested in the areas of the image where the model is making 
    # predictions.
    
    # Calculate MCD Uncertainty
    # global score of how certain or uncertain the model is given an input image, which can be useful if you want to get a single uncertainty value for the entire image
    # mcd_uncertainty = torch.sum(u) / torch.numel(u)
    # Calculate MCD Uncertainty within the predicted mask
    mask = m > 0.5  # replace 'threshold' with appropriate value
    
    mcd_uncertainty = (torch.sum(u[mask]) / torch.sum(mask)).item() if torch.sum(mask) > 0 else 0.0

    print(f"Image name: {name}, Mean uncertainty: {mean_uncertainty.item()}, MCD Uncertainty: {mcd_uncertainty}")

    csv.write(f"{name},{mean_uncertainty.item()},{mcd_uncertainty}\n")
