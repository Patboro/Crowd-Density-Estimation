import matplotlib.pyplot as plt
import os

def visualize_density_map(image, ground_truth, prediction, epoch, save_path=None):
    """
    Visualize the original image, ground truth density map, and predicted density map.

    Args:
        image (torch.Tensor): The original input image (C, H, W).
        ground_truth (numpy.ndarray): The ground truth density map (H, W).
        prediction (numpy.ndarray): The predicted density map (H, W).
        epoch (int): Current epoch number (for title).
        save_path (str): Path to save the visualization (optional).
    """
    image = image.cpu().numpy().transpose(1, 2, 0)  # Convert (C, H, W) to (H, W, C)
    image = (image - image.min()) / (image.max() - image.min())

    ground_truth = ground_truth.squeeze()
    prediction = prediction.squeeze()

    plt.figure(figsize=(15, 5))

    # Display the original image
    plt.subplot(1, 3, 1)
    plt.title(f"Original Image (Epoch {epoch})")
    plt.imshow(image)
    plt.axis("off")

    # Ground truth density map
    plt.subplot(1, 3, 2)
    plt.title("Ground Truth Density Map")
    plt.imshow(ground_truth, cmap="jet")
    plt.axis("off")

    # Predicted density map
    plt.subplot(1, 3, 3)
    plt.title("Predicted Density Map")
    plt.imshow(prediction, cmap="jet")
    plt.axis("off")

    if save_path:
        plt.savefig(save_path)
    plt.show()



def plot_training_metrics(metrics, save_path):
    """
    Plot training loss and validation MSE.

    Args:
        metrics (dict): Dictionary containing "train_loss" and "val_MSE".
        save_path (str): Path to save the plot.
    """
    plt.figure(figsize=(12, 6))

    # Plot training loss
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(metrics["train_loss"]) + 1), metrics["train_loss"], label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()
    plt.grid()

    # Plot validation MSE
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(metrics["val_MSE"]) + 1), metrics["val_MSE"], label="Validation MSE", color="orange")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title("Validation MSE")
    plt.legend()
    plt.grid()

    plot_path = os.path.join(save_path, "training_metrics_plot.png")
    plt.tight_layout()
    plt.savefig(plot_path)
    print(f"Training plot saved at: {plot_path}")
    plt.show()


def visualize_side_by_side(image, density_map, title1="Image", title2="Density Map"):
    """
    Display an image and its corresponding density map side by side.

    Args:
        image (ndarray): Input image.
        density_map (ndarray): Corresponding density map.
        title1 (str): Title for the image.
        title2 (str): Title for the density map.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(image, cmap="gray")
    axes[0].set_title(title1)
    axes[0].axis("off")

    axes[1].imshow(density_map, cmap="jet")
    axes[1].set_title(title2)
    axes[1].axis("off")

    plt.show()

def visualize_overlay(image, density_map, alpha=0.6, title="Overlay"):
    """
    Overlay the density map on the image.

    Args:
        image (ndarray): Input image.
        density_map (ndarray): Corresponding density map.
        alpha (float): Transparency of the overlay.
        title (str): Title of the visualization.
    """
    plt.figure(figsize=(8, 8))
    plt.imshow(image, cmap="gray")
    plt.imshow(density_map, cmap="jet", alpha=alpha)
    plt.title(title)
    plt.axis("off")
    plt.show()

def visualize_with_points(image, points, title="Image with Points"):
    """
    Visualize an image with annotated points.

    Args:
        image (ndarray): Input image.
        points (ndarray): Array of points (x, y).
        title (str): Title for the visualization.
    """
    plt.figure(figsize=(8, 8))
    plt.imshow(image, cmap="gray")
    if points is not None:
        y_coords, x_coords = points[:, 1], points[:, 0]
        plt.scatter(x_coords, y_coords, c="red", s=10, label="Points")
    plt.title(title)
    plt.axis("off")
    plt.legend()
    plt.show()


