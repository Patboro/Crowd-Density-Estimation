# Crowd-Density-Estimation

A comprehensive solution for crowd density estimation using U-Net-based segmentation. This project combines custom-built components for data processing, model training, and visualization to accurately estimate crowd density in images.

## Features
- **U-Net Architecture**: Utilizes an implementation of the U-Net architecture for semantic segmentation.
- **Modular Codebase**: Includes reusable components for data handling, training, visualization, and utility functions.
- **Custom Dataset Support**: Supports loading and processing custom datasets for training and evaluation.
- **Visualization Tools**: Built-in visualization functions to analyze model predictions and results.

## Repository Structure

```
.
├── LICENSE                 # License file
├── README.md               # Project description and usage guide
├── automated_training.py   # Script for automating training and evaluation
├── dataset_class.py        # Dataset loading and preprocessing
├── main.py                 # Main entry point for running the project
├── unet.py                 # Core U-Net implementation
├── unet_components.py      # U-Net building blocks (layers, modules)
├── unet_trainer.py         # Training loop and model checkpointing
├── utility.py              # Helper functions for general purposes
├── visualization.py        # Tools for visualizing predictions and results
```

