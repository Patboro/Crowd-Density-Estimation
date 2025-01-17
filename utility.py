import torch
import numpy as np



def generate_density_map(shape, points, kernel_size=50, sigma=5):
    """
    Generate a density map using Gaussian kernels for annotated points.

    Args:
        shape (tuple): Shape of the output density map (height, width).
        points (ndarray): Array of (x, y) coordinates for points (crowd locations).
        kernel_size (int): Size of the Gaussian kernel.
        sigma (int): Standard deviation of the Gaussian kernel.

    Returns:
        density_map (ndarray): Generated density map with normalized values.
    """
    # Initialize an empty density map with the specified shape
    density_map = np.zeros(shape, dtype=np.float32)

    # Create a Gaussian kernel
    ax = np.linspace(-(kernel_size // 2), kernel_size // 2, kernel_size)
    xx, yy = np.meshgrid(ax, ax)
    gauss_kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    gauss_kernel /= np.sum(gauss_kernel)  # Normalize the kernel so that its sum equals 1

    # Iterate through each point and apply the Gaussian kernel
    for point in points:
        x, y = int(point[0]), int(point[1])  # Convert coordinates to integers

        # Ensure the point is within the density map boundaries
        if 0 <= x < shape[1] and 0 <= y < shape[0]:
            # Define the region in the density map where the kernel will be applied
            x_start = max(0, x - kernel_size // 2)
            x_end = min(shape[1], x + kernel_size // 2 + 1)
            y_start = max(0, y - kernel_size // 2)
            y_end = min(shape[0], y + kernel_size // 2 + 1)

            # Define the corresponding region in the Gaussian kernel
            k_x_start = max(0, kernel_size // 2 - x)
            k_x_end = k_x_start + (x_end - x_start)
            k_y_start = max(0, kernel_size // 2 - y)
            k_y_end = k_y_start + (y_end - y_start)

            # Extract slices from the kernel and density map for the selected region
            kernel_slice = gauss_kernel[k_y_start:k_y_end, k_x_start:k_x_end]
            density_slice = density_map[y_start:y_end, x_start:x_end]

            # Adjust shapes in case the region goes beyond the boundaries
            min_y, min_x = (min(kernel_slice.shape[0], density_slice.shape[0]),
                            min(kernel_slice.shape[1], density_slice.shape[1]))
            kernel_slice = kernel_slice[:min_y, :min_x]
            density_slice = density_slice[:min_y, :min_x]

            # Add the Gaussian kernel to the selected region of the density map
            density_map[y_start:y_start + min_y, x_start:x_start + min_x] += kernel_slice

    # Normalize the density map to ensure its sum matches the number of points
    density_sum = np.sum(density_map)
    if density_sum > 0:
        density_map *= len(points) / density_sum  # Scale to match the total number of points

    return density_map

def weighted_mse_loss(predicted, target, weight=50.0, threshold = 0.7):
    """
    Compute a weighted MSE loss to prioritize hotspot regions in density maps.

    Args:
        predicted (torch.Tensor): Predicted density map.
        target (torch.Tensor): Ground truth density map.
        weight (float): Weight applied to hotspot regions.
        threshold (float): Threshold for identifying hotspot regions.

    Returns:
        torch.Tensor: Weighted MSE loss.
    """
    # Identify hotspot regions
    hotspot_mask = target > threshold  # Threshold to identify hotspots
    hotspot_weights = torch.ones_like(target)
    hotspot_weights[hotspot_mask] = weight

    # Calculate weighted MSE loss
    loss = ((predicted - target) ** 2) * hotspot_weights
    return loss.mean()






