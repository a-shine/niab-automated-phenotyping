import os
from typing import List
from torch.utils.data import random_split
from utils.dl.niab import SegmentationDataset, IMG_TRANSFORMS, MASK_TRANSFORMS
import segmentation_models_pytorch as smp
import torch
import torch.backends.mps
from torch.utils.data import DataLoader
from utils.dl.model import MCDUNet

# set torch seed
torch.cuda.manual_seed_all(42)

data_processed = SegmentationDataset(
    "/home/users/ashine/gws/niab-automated-phenotyping/datasets/niab/EXP01/Top_Images/Masked_Dataset/imgs", 
    "/home/users/ashine/gws/niab-automated-phenotyping/datasets/niab/EXP01/Top_Images/Masked_Dataset/masks",
    img_transform=IMG_TRANSFORMS,
    mask_transform=MASK_TRANSFORMS
    )

# Split the dataset into training and validation sets
total_size = len(data_processed)
train_size = int(0.8 * total_size)
val_size = total_size - train_size
training_dataset, validation_dataset = random_split(data_processed, [train_size, val_size])

# How much data?
print(f"Size of training dataset: {len(training_dataset)}")
print(f"Size of validation dataset: {len(validation_dataset)}")

# Post-transformation, what does the data look like?
print(f"Shape of first input entry (post-transformation): {data_processed[0][0].shape}")
print(f"Shape of first label entry (post-transformation): {data_processed[0][1].shape}")

# Parameters
# https://ai.stackexchange.com/questions/8560/how-do-i-choose-the-optimal-batch-size
# https://stats.stackexchange.com/questions/164876/what-is-the-trade-off-between-batch-size-and-number-of-iterations-to-train-a-neu
# It has been observed that with larger batch there is a significant degradation in the quality of the model, as
# measured by its ability to generalize i.e. large batch size is better for training but not for generalization
# (overfitting)
BATCH_SIZE = 2 ** 4  # should be divisible by the training dataset size
EPOCHS = 50

# Detect device for training and running the model
# Installing CUDA - https://docs.nvidia.com/cuda/cuda-quick-start-guide/
device = ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Training/fitting using {device} device")

# Create a data loader to handle loading data in and out of memory in batches

# Create data loaders.
train_dataloader = DataLoader(training_dataset, batch_size=BATCH_SIZE)
validation_dataloader = DataLoader(validation_dataset, batch_size=BATCH_SIZE)

# Look at the shape of the data coming out of the data loader (batch size, channels, height, width)
for X, y in validation_dataloader:
    print(f"Shape of X: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break

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

def fit(dataloader, model, loss_fn, optimizer, scheduler, device, log_freq=10) -> None:
    """
    Fit the model to the data using the loss function and optimizer
    Taken from the official PyTorch quickstart tutorial (https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html#optimizing-the-model-parameters)
    :param dataloader:
    :param model:
    :param loss_fn:
    :param optimizer:
    :param scheduler:
    :param device:
    :param log_freq:
    :return: None
    """
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # move data to device (GPU or CPU), ensure that the data and model are on the same device
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)  # forward pass
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()  # backward pass (calculate gradients)
        optimizer.step()  # update params
        optimizer.zero_grad()  # reset gradients to zero
        # scheduler.step()  # update learning rate (decay)

        # To avoid too much output, only print every n batches (log_freq), by default every 10 batches
        if batch % log_freq == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def validate(dataloader, model, loss_fn):
    num_batches = len(dataloader)
    model.eval()
    total_loss, total_precision, total_recall, total_f1, total_iou = 0, 0, 0, 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            total_loss += loss_fn(pred, y).item()

            # Convert predictions to binary class labels
            pred_binary = (pred > 0.5).float()

            # Calculate metrics


    # Calculate averages
    avg_loss = total_loss / num_batches
    # avg_precision = total_precision / num_batches
    # avg_recall = total_recall / num_batches
    # avg_f1 = total_f1 / num_batches
    # avg_iou = total_iou / num_batches

    print(f"Validation Metrics: Avg loss: {avg_loss:>8f} \n")
    return avg_loss

# loss_fn = torch.nn.BCEWithLogitsLoss()
loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)
optimizer = torch.optim.Adam(params = model.parameters(), lr = 3e-4) # high learning rate and allow it to decay
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)  # decay the learning rate by a factor of 0.1 every epoch

# Setting up directory to save models
os.makedirs("/home/users/ashine/gws/niab-automated-phenotyping/models", exist_ok=True)

best_loss = float('inf')
for t in range(EPOCHS):
    print(f"Epoch {t + 1}\n-------------------------------")
    fit(train_dataloader, model, loss_fn, optimizer, None, device, log_freq=2)
    val_loss = validate(validation_dataloader, model, loss_fn)
    if val_loss < best_loss:
        best_loss = val_loss
        torch.save(model.state_dict(), f"/home/users/ashine/gws/niab-automated-phenotyping/models/best_model.pth")
        print("Saved best model")
print("Done!")