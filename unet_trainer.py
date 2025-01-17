import os
import json
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize, RandomHorizontalFlip, ColorJitter
from visualization import visualize_density_map, plot_training_metrics
from dataset_class import ShanghaiDataset


class UNetTrainer:

    def __init__(self, config):
        """
        Initialize the training process.

        Args:
            config (dict): Configuration dictionary containing training parameters.
        """
        self.config = config
        self.device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")

        # Define data transformations and augmentations
        train_transform = Compose([
            ToTensor(),
            Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        train_augmentations = Compose([
            RandomHorizontalFlip(p=0.5),
            ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
        ])
        val_transform = Compose([
            ToTensor(),
            Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        # Load datasets
        self.train_dataset = ShanghaiDataset(
            config["root_path"], mode="train", transform=train_transform, augmentations=train_augmentations
        )
        self.val_dataset = ShanghaiDataset(
            config["root_path"], mode="test", transform=val_transform
        )

        # Create DataLoaders
        self.train_loader = DataLoader(self.train_dataset, batch_size=config["batch_size"], shuffle=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=config["batch_size"], shuffle=True)


        self.model = config["model"].to(self.device)
        self.train_criterion = config["train_criterion"]
        self.val_criterion = config["val_criterion"]
        self.optimizer = config["optimizer"]
        self.save_path = config["save_path"]
        self.save_interval = config.get("save_interval", 5)

        # Metrics tracking
        self.best_MSE = float("inf")
        self.best_epoch = -1
        self.metrics = {"train_loss": [], "val_MSE": []}

        # Create directories for saving outputs
        os.makedirs(self.save_path, exist_ok=True)
        os.makedirs(os.path.join(self.save_path, "validation_plots"), exist_ok=True)

    def train_one_epoch(self, epoch):
        """
        Train the model for one epoch using pixel-level MSE.

        Args:
            epoch (int): Current epoch number.
        """
        self.model.train()  # Set model to training mode
        running_loss = 0.0

        # Iterate through batches
        for batch_idx, batch in enumerate(tqdm(self.train_loader, desc=f"Training Epoch {epoch}")):
            images = batch["image"].to(self.device)  # Input images
            density_maps = batch["density_map"].to(self.device)  # Ground truth density maps

            # Forward pass
            predicted_maps = self.model(images)


            loss = self.train_criterion(predicted_maps, density_maps)

            # Backward pass and optimization
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

            # Clear CUDA cache periodically
            if batch_idx % 10 == 0:
                torch.cuda.empty_cache()

        avg_loss = running_loss / len(self.train_loader)
        self.metrics["train_loss"].append(avg_loss)
        print(f"Epoch {epoch} average density loss (MSE): {avg_loss:.4f}")

    def validate(self, epoch):
        """
        Validate the model and compute MSE for density maps.
        """
        self.model.eval()
        total_MSE = 0.0

        with torch.no_grad():
            # Iterate through validation data
            for batch_idx, batch in enumerate(tqdm(self.val_loader, desc=f"Validation Epoch {epoch}")):
                images = batch["image"].to(self.device)  # Input images
                density_maps = batch["density_map"].to(self.device)  # Ground truth density maps

                # Forward pass
                predicted_maps = self.model(images)

                # Compute pixel-level MSE
                MSE = self.val_criterion(predicted_maps, density_maps).item()
                total_MSE += MSE

                # Visualize predictions for the first batch
                if batch_idx == 0:
                    visualize_density_map(
                        image=images[0],
                        ground_truth=density_maps[0].cpu().numpy(),
                        prediction=predicted_maps[0].cpu().numpy(),
                        epoch=epoch,
                        save_path=os.path.join(self.save_path, "validation_plots",
                                               f"epoch_{epoch}_batch_{batch_idx}.png")
                    )

        avg_MSE = total_MSE / len(self.val_loader)
        self.metrics["val_MSE"].append(avg_MSE)
        print(f"Epoch {epoch} validation MSE: {avg_MSE:.4f}")

        # Save the best model based on validation MSE
        if avg_MSE < self.best_MSE:
            self.best_MSE = avg_MSE
            self.best_epoch = epoch
            torch.save(self.model.state_dict(), os.path.join(self.save_path, "best_model.pth"))
            print(f"Best model saved at epoch {epoch} with MSE: {avg_MSE:.4f}")

    def train(self):

        for epoch in range(1, self.config["epochs"] + 1):
            self.train_one_epoch(epoch)
            self.validate(epoch)

            # Save the model periodically
            if epoch % self.save_interval == 0:
                torch.save(self.model.state_dict(), os.path.join(self.save_path, f"unet_epoch_{epoch}.pth"))
                print(f"Model saved at epoch {epoch}")

        # Save metrics and plot training curve
        with open(os.path.join(self.save_path, "metrics.json"), "w") as f:
            json.dump(self.metrics, f, indent=4)

        plot_training_metrics(self.metrics, self.save_path)
        print(f"Best validation MSE: {self.best_MSE:.4f} at epoch {self.best_epoch}")



