import torch
from torch.utils.data import Dataset
from glob import glob
from scipy.io import loadmat
from PIL import Image
import numpy as np
from torchvision.transforms import ToTensor, Compose, RandomHorizontalFlip, ColorJitter, Normalize
import cv2
import os
import torch.nn.functional as F
from utility import generate_density_map



class ShanghaiDataset(Dataset):
    """
    Dataset loader for the ShanghaiTech crowd counting dataset.
    Provides images, corresponding density maps, and point counts.
    """

    def __init__(self, root_dir, mode="train", transform=None, augmentations=None):
        """
        Initialize the dataset.

        Args:
            root_dir (str): Path to the dataset's root directory.
            mode (str): Subset to load, either 'train' or 'test'.
            transform (callable): Preprocessing transformations for the images.
            augmentations (callable): Data augmentations applied only during training.
        """
        # Ensure mode is valid
        assert mode in ["train", "test"], "Mode must be 'train' or 'test'."

        self.mode = mode
        self.transform = transform
        self.augmentations = augmentations

        # Define paths to images and ground truth annotations
        self.image_dir = os.path.join(root_dir, f"{mode}_data/images")
        self.gt_dir = os.path.join(root_dir, f"{mode}_data/ground-truth")

        # Get a list of all image files in the directory
        self.images = glob(os.path.join(self.image_dir, "*.jpg"))

        # Resize shape for both images and density maps
        self.resize_shape = (512, 512)

    def __len__(self):

        return len(self.images)

    def __getitem__(self, idx):
        """
        Retrieve an image, its corresponding density map, and point count.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            dict: A dictionary containing:
                - "image": Preprocessed image as a tensor.
                - "density_map": Corresponding density map as a tensor.
                - "point_count": Number of annotated points in the image.
                - "idx": Index of the sample.
        """
        # Convert tensor index to integer if needed
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Load the image using OpenCV and convert from BGR to RGB
        fname = self.images[idx]
        image = cv2.cvtColor(cv2.imread(fname), cv2.COLOR_BGR2RGB)

        # Load the corresponding ground truth file for the image
        base_name = os.path.basename(fname).replace(".jpg", "")
        mat_name = f"GT_{base_name}.mat"
        mat_path = os.path.join(self.gt_dir, mat_name)

        # Load ground truth point locations from the .mat file
        mat_data = loadmat(mat_path)
        location_data = mat_data["image_info"][0][0][0][0][0]  # (x, y) coordinates
        point_count = location_data.shape[0]  # Number of annotated points


        original_height, original_width, _ = image.shape
        density_map = generate_density_map((original_height, original_width), location_data,
                                           kernel_size=90, sigma=15)


        image = cv2.resize(image, self.resize_shape)


        density_map = F.interpolate(
            torch.tensor(density_map).unsqueeze(0).unsqueeze(0),  # Add batch & channel dims
            size=self.resize_shape,
            mode="bilinear",
            align_corners=False
        ).squeeze(0).squeeze(0).numpy()  # Remove batch & channel dims to get back a 2D numpy array


        # Apply augmentations (only during training mode)
        if self.augmentations and self.mode == "train":
            image, density_map = self.apply_augmentations(image, density_map)


        if self.transform:
            image = self.transform(image)
        else:
            # Default transformations: Convert to tensor and normalize
            image = Compose([
                ToTensor(),
                Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize pixel values to [-1, 1]
            ])(image)

        #scale the density map
        density_map *= 255
        density_map = torch.tensor(density_map, dtype=torch.float32)



        return {
            "image": image,
            "density_map": density_map.unsqueeze(0),  # Density map tensor with channel dimension
            "point_count": point_count,
            "idx": idx
        }

    def apply_augmentations(self, image, density_map):
        """
        Apply augmentations to the image and adjust the density map.

        Args:
            image (ndarray): Input image.
            density_map (ndarray): Corresponding density map.

        Returns:
            tuple: Augmented image and density map.
        """
        # Define augmentations using PyTorch's Compose
        transform = Compose([
            RandomHorizontalFlip(p=0.5),  # Random horizontal flip with 50% probability
            ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)  # Apply color jitter
        ])

        # Apply augmentations to the image only (density map remains unchanged)
        pil_image = Image.fromarray(image)
        augmented_image = transform(pil_image)

        return np.array(augmented_image), density_map
