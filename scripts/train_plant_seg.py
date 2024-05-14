# Train model for top down shoot segmentation. Can be called using slurm with 
# the training-job.sh script. The command is `sbatch training-job.sh`.

# Good example of training script using segmentation_models.pytorch can be found 
# here: https://github.com/qubvel/segmentation_models.pytorch/blob/master/examples/binary_segmentation_intro.ipynb

import os
from typing import List
from torch.utils.data import random_split
from utils.dl.niab import SegmentationDataset, IMG_TRANSFORMS, MASK_TRANSFORMS
import segmentation_models_pytorch as smp
import torch
import torch.backends.mps
from torch.utils.data import DataLoader
from utils.dl.model import MCDUNet
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt


torch.cuda.manual_seed_all(42)  # set torch seed

# Parameters
# https://ai.stackexchange.com/questions/8560/how-do-i-choose-the-optimal-batch-size
# https://stats.stackexchange.com/questions/164876/what-is-the-trade-off-between-batch-size-and-number-of-iterations-to-train-a-neu
# It has been observed that with larger batch there is a significant degradation in the quality of the model, as
# measured by its ability to generalize i.e. large batch size is better for training but not for generalization
# (overfitting)
BATCH_SIZE = 2 ** 4  # should be divisible by the training dataset size
EPOCHS = 50

data_processed = SegmentationDataset(
    "/home/users/ashine/gws/niab-automated-phenotyping/datasets/niab/EXP01/Top_Images/Masked_Dataset/imgs", 
    "/home/users/ashine/gws/niab-automated-phenotyping/datasets/niab/EXP01/Top_Images/Masked_Dataset/masks",
    img_transform=IMG_TRANSFORMS,
    mask_transform=MASK_TRANSFORMS
    )

# Split the dataset into training and validation sets
total_size = len(data_processed)

# Split the dataset into training, validation and test sets
total_size = len(data_processed)
train_size = int(0.8 * total_size)  # 80% for training
val_size = int(0.1 * total_size)  # 10% for validation
test_size = total_size - train_size - val_size  # ~10% for testing

training_dataset, validation_dataset, test_dataset = random_split(data_processed, [train_size, val_size, test_size])

# How much data?
print(f"Size of training dataset: {len(training_dataset)}")
print(f"Size of validation dataset: {len(validation_dataset)}")
print(f"Size of test dataset: {len(test_dataset)}")

# Post-transformation, what does the data look like?
print(f"Shape of first input entry (post-transformation): {data_processed[0][0].shape}")
print(f"Shape of first label entry (post-transformation): {data_processed[0][1].shape}")

# Detect device for training and running the model
# Installing CUDA - https://docs.nvidia.com/cuda/cuda-quick-start-guide/
device = ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Training/fitting using {device} device")

# Create data loaders
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
    total_loss = 0
    for batch, (X, y) in enumerate(dataloader):
        # move data to device (GPU or CPU), ensure that the data and model are on the same device
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)  # forward pass
        loss = loss_fn(pred, y)
        total_loss += loss.item()

        # Backpropagation
        loss.backward()  # backward pass (calculate gradients)
        optimizer.step()  # update params
        optimizer.zero_grad()  # reset gradients to zero
        # scheduler.step()  # update learning rate (decay)

        # To avoid too much output, only print every n batches (log_freq), by default every 10 batches
        if batch % log_freq == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    avg_loss = total_loss / len(dataloader)
    
    return {
        "loss": avg_loss
    }


def validate(dataloader, model, loss_fn, threshold=0.5):
    num_batches = len(dataloader)
    model.eval()
    total_loss = 0
    total_iou_score = torch.tensor(0.0).to(device)
    total_f1 = torch.tensor(0.0).to(device)
    
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            
            # Compute loss before converting the predictions to a binary mask
            # The reason is that most loss functions for segmentation tasks, 
            # such as Binary Cross Entropy or Dice Loss, operate on the raw 
            # output of the model (the logits) and not on the binarized 
            # version. These loss functions incorporate a form of thresholding 
            # as part of their calculation, and applying an additional 
            # thresholding step before calculating the loss can disrupt this.
            total_loss += loss_fn(pred, y).item()

            # Lets compute metrics for some threshold
            # first convert mask values to probabilities, then 
            # apply thresholding
            prob_mask = pred.sigmoid()
            pred_mask = (prob_mask > threshold).float()

            # Calculate true positives, false positives, false negatives, true negatives
            tp, fp, fn, tn = smp.metrics.get_stats(pred_mask.long(), y.long(), mode='binary')

            # compute metric
            total_iou_score += smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
            total_f1 += smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro")
    
    # Calculate averages
    avg_loss = total_loss / num_batches
    avg_iou_score = total_iou_score / num_batches
    avg_f1 = total_f1 / num_batches

    print(f"Validation Metrics:\n- Avg loss: {avg_loss:>8f}\n- IoU Score: {avg_iou_score:>8f}\n- F1 Score: {avg_f1:>8f}\n")
    return {
        "loss": avg_loss,
        "iou_score": avg_iou_score.item(),
        "f1": avg_f1.item()
    }

loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)
optimizer = torch.optim.Adam(params = model.parameters(), lr = 3e-4) # high learning rate and allow it to decay
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)  # decay the learning rate by a factor of 0.1 every epoch

# creating training job id based on timestamp
job_id = datetime.now().strftime("%Y%m%d%H%M%S")

# Setting up directory to save models
os.makedirs(f"/home/users/ashine/gws/niab-automated-phenotyping/models/{job_id}", exist_ok=True)

# create a pd dataframe to store metrics
val_metrics_df = pd.DataFrame([], columns=["epoch", "train_loss", "val_loss", "val_iou_score", "val_f1"])

best_loss = float('inf')
for t in range(EPOCHS):
    print(f"Epoch {t + 1}\n-------------------------------")
    train_metrics = fit(train_dataloader, model, loss_fn, optimizer, None, device, log_freq=2)
    val_metrics = validate(validation_dataloader, model, loss_fn)
    
    # add new row to the dataframe with the metrics
    metrics_df = pd.DataFrame([{"epoch": t, "train_loss": train_metrics["loss"], "val_loss": val_metrics["loss"], "val_iou_score": val_metrics["iou_score"], "val_f1": val_metrics["f1"]}])

    val_metrics_df = pd.concat([
        val_metrics_df if not val_metrics_df.empty else None,
        metrics_df],
        ignore_index=True)
    
    val_metrics_df.to_csv(f"/home/users/ashine/gws/niab-automated-phenotyping/models/{job_id}/val_metrics.csv")

    # if the loss of this epoch is better than the best loss, save the model
    if val_metrics["loss"] < best_loss:
        best_loss = val_metrics["loss"]
        torch.save(model.state_dict(), f"/home/users/ashine/gws/niab-automated-phenotyping/models/{job_id}/best_model.pth")
        print("Saved best model")

# test performance
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
test_metrics = validate(test_dataloader, model, loss_fn)
print(f"Test Metrics:\n- Avg loss: {test_metrics['loss']:>8f}\n- IoU Score: {test_metrics['iou_score']:>8f}\n- F1 Score: {test_metrics['f1']:>8f}\n")


# Initial plot of train and val loss
plt.figure()
plt.plot(val_metrics_df["epoch"], val_metrics_df["train_loss"], label="train_loss")
plt.plot(val_metrics_df["epoch"], val_metrics_df["val_loss"], label="val_loss")

# save the plot
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss over Epoch")
plt.legend()
plt.savefig(f"/home/users/ashine/gws/niab-automated-phenotyping/models/{job_id}/training_loss.png")

# Create a new figure for the second plot
plt.figure()
plt.plot(val_metrics_df["epoch"], val_metrics_df["val_iou_score"], label="iou_score")
plt.plot(val_metrics_df["epoch"], val_metrics_df["val_f1"], label="f1")

plt.xlabel("Epoch")
plt.ylabel("Metrics")
plt.title("Metrics over Epoch")
plt.legend()
plt.savefig(f"/home/users/ashine/gws/niab-automated-phenotyping/models/{job_id}/val_metrics.png")

print("Done!")