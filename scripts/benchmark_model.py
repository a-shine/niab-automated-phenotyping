# Benchmark model on fully corrected annotated dataset (accurate ground truth)

from utils.dl.niab import SegmentationDataset, IMG_TRANSFORMS_NO_JITTER, MASK_TRANSFORMS
import segmentation_models_pytorch as smp
import torch
import torch.backends.mps
from torch.utils.data import DataLoader
from utils.dl.model import MCDUNet

BATCH_SIZE = 16
MODEL_PATH = "/home/users/ashine/gws/niab-automated-phenotyping/models/20240624182742/best_model.pth"

test_dataset = SegmentationDataset(
    "/home/users/ashine/gws/niab-automated-phenotyping/datasets/niab/EXP01/Top_Images/Annotated_Test_Dataset/imgs", 
    "/home/users/ashine/gws/niab-automated-phenotyping/datasets/niab/EXP01/Top_Images/Annotated_Test_Dataset/masks",
    img_transforms=IMG_TRANSFORMS_NO_JITTER,
    mask_transforms=MASK_TRANSFORMS,
    common_transforms=None
)

# Split the dataset into training and validation sets
total_size = len(test_dataset)

print(f"Size of test dataset: {len(test_dataset)}")

# Detect device for training and running the model
# Installing CUDA - https://docs.nvidia.com/cuda/cuda-quick-start-guide/
device = ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Training/fitting using {device} device")

# Create an instance of the model and move it to the device (GPU or CPU)
# model = MCDUNet(n_channels=3,
#              n_classes=1,
#              bilinear=True,
#              ddims=[32, 64, 128, 256, 512],
#              UQ=True,
#              activation=True,
#              ).to(device)

model = smp.Unet(
    encoder_name="resnet34",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
    in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    classes=1,                      # model output channels (number of classes in your dataset)
    activation="sigmoid"            # model output activation function (e.g. `softmax` or `sigmoid`)
).to(device)

# model = smp.DeepLabV3Plus(
#     encoder_name="resnet34",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
#     encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
#     in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.
#     classes=1,                      # model output channels (number of classes in your dataset
#     activation="sigmoid"            # model output activation function (e.g. `softmax` or `sigmoid`)
# ).to(device)

# Load the model
model.load_state_dict(torch.load(MODEL_PATH))

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
            # prob_mask = pred.sigmoid()
            pred_mask = (pred > threshold).float()

            # Calculate true positives, false positives, false negatives, true negatives
            tp, fp, fn, tn = smp.metrics.get_stats(pred_mask.long(), y.long(), mode='binary')

            # compute metric
            total_iou_score += smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
            total_f1 += smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro")
    
    # Calculate averages
    avg_loss = total_loss / num_batches
    avg_iou_score = total_iou_score / num_batches
    avg_f1 = total_f1 / num_batches

    return {
        "loss": avg_loss,
        "iou_score": avg_iou_score.item(),
        "f1": avg_f1.item()
    }

loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=False)

# test performance
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
test_metrics = validate(test_dataloader, model, loss_fn)
print(f"Test Metrics:\n- Avg loss: {test_metrics['loss']:>8f}\n- IoU Score: {test_metrics['iou_score']:>8f}\n- F1 Score: {test_metrics['f1']:>8f}\n")

print("Done!")