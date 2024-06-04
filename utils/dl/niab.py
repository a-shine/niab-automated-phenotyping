from torchvision.transforms.v2 import ToTensor, Compose, RandomHorizontalFlip, ColorJitter, Normalize, Resize
from torch.utils.data import Dataset
import os
from PIL import Image
from utils.image_utils import white_balance
import glob
import cv2
import torch


IMG_TRANSFORMS = Compose([
    Resize((256, 256)),  # Resize the image to 256x256
    ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),  # Random color jitter
    ToTensor(),  # Convert the image to a PyTorch tensor
    Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0])  # Normalize
])

MASK_TRANSFORMS = Compose([
    Resize((256, 256), interpolation=Image.NEAREST),  # Resize the image to 256x256
    ToTensor(),
])

# Transformation that need to be applied to both the image and mask (at the same time)
COMMON_TRANSFORMS = Compose([
    RandomHorizontalFlip()
])

class SegmentationDataset(Dataset):
    def __init__(self, img_dir, mask_dir, img_transforms=None, mask_transforms=None, common_transforms=None):
        """
        Custom dataset for segmentation tasks.

        Args:
            img_dir (str): Path to the folder containing images.
            mask_dir (str): Path to the folder containing masks.
            img_transform (callable, optional): Optional transform to be applied to the image.
            mask_transform (callable, optional): Optional transform to be applied to the mask.
        """
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_transforms = img_transforms
        self.mask_transforms = mask_transforms
        self.common_transforms = common_transforms

        # List all image and mask files in the directories
        self.img_files = sorted(os.listdir(img_dir))
        self.mask_files = sorted(os.listdir(mask_dir))

        # Check if the number of images and masks match
        assert len(self.img_files) == len(self.mask_files), "Number of images and masks must be the same."
   
    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        # Get the file names for the corresponding image and mask
        img_name = os.path.join(self.img_dir, self.img_files[idx])
        mask_name = os.path.join(self.mask_dir, self.mask_files[idx])

        # Open image and mask files
        img = Image.open(img_name)
        mask = Image.open(mask_name)

        mask = mask.convert("L")

        if self.common_transforms:
            img, mask = self.common_transforms(img, mask)

        if self.img_transforms:
            img = self.img_transforms(img)

        if self.mask_transforms:
            mask = self.mask_transforms(mask)

            # Binarize the mask tensor
            mask = (mask > 0.5).float()
        
        return img, mask
    

class ActiveLearningDataset(Dataset):
    def __init__(self, img_dir, img_transform=None):
        """
        Custom dataset for segmentation tasks.

        Args:
            img_dir (str): Path to the folder containing images.
            mask_dir (str): Path to the folder containing masks.
            img_transform (callable, optional): Optional transform to be applied to the image.
            mask_transform (callable, optional): Optional transform to be applied to the mask.
        """
        self.img_dir = img_dir
        self.img_transform = img_transform

        # List all image and mask files in the directories
        self.img_files = sorted(glob.glob(f"{img_dir}/*/*/*.jpg"))
   
    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        # Get the file names for the corresponding image and mask
        img_name = os.path.join(self.img_files[idx])

        img = white_balance(cv2.imread(img_name))

        # Open image
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        # Apply transformations, if specified
        if self.img_transform:
            img = self.img_transform(img)

        return img_name, img
    

class InstanceDataset(Dataset):
    def __init__(self, binary_mask_dir, instance_mask_dir, img_transform=None, mask_transform=None):
        """
        Custom dataset for segmentation tasks.

        Args:
            img_dir (str): Path to the folder containing images.
            mask_dir (str): Path to the folder containing masks.
            img_transform (callable, optional): Optional transform to be applied to the image.
            mask_transform (callable, optional): Optional transform to be applied to the mask.
        """
        self.binary_mask_dir = binary_mask_dir
        self.instance_mask_dir = instance_mask_dir
        self.binary_mask_transform = img_transform
        self.instance_mask_transform = mask_transform

        # List all image and mask files in the directories
        self.binary_mask_files = sorted(os.listdir(binary_mask_dir))
        self.instance_mask_files = sorted(os.listdir(instance_mask_dir))

        # Check if the number of images and masks match
        assert len(self.binary_mask_files) == len(self.instance_mask_files), "Number of images and masks must be the same."

    def __len__(self):
        return len(self.binary_mask_files)

    def __getitem__(self, idx):
        # Get the file names for the corresponding image and mask
        binary_mask_name = os.path.join(self.binary_mask_dir, self.binary_mask_files[idx])
        instance_mask_name = os.path.join(self.instance_mask_dir, self.instance_mask_files[idx])

        # Open image and mask files
        binary_mask = Image.open(binary_mask_name)
        instance_mask = Image.open(instance_mask_name)

        binary_mask = binary_mask.convert("L")

        # Apply transformations, if specified
        if self.binary_mask_transform:
            binary_mask = self.binary_mask_transform(binary_mask)

            # binarize the mask tensor
            binary_mask = (binary_mask > 0.5).float()

        if self.instance_mask_transform:
            instance_mask = self.instance_mask_transform(instance_mask)

            # map instance mask to integer value
            unique_values, indices = torch.unique(instance_mask, return_inverse=True)

            instance_mask = indices
    
        return binary_mask, instance_mask