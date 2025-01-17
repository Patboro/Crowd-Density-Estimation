import torch
from torch import optim
from unet import Unet
from automated_training import AutomatedTraining
from utility import weighted_mse_loss
import torch.nn as nn

if __name__ == "__main__":
    if torch.cuda.is_available():
        print("Cleaning CUDA memory...")
        torch.cuda.empty_cache()

    base_config = {
        "root_path": dataset_path,
        "model": Unet(in_channels=3, num_classes=1),
        "train_criterion": weighted_mse_loss, # nn.MSELoss(),
        "val_criterion": nn.MSELoss(),
        "optimizer": optim.Adam,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "results_path": results_folder_path
    }

    # Initialize Automated Training
    automation = AutomatedTraining(base_config)

    # Choose mode: Single Training or Multiple Trainings
    mode = "single"

    if mode == "multiple":

        batch_sizes = [9, 12]
        epochs_list = [50, 100]
        learning_rates = [1e-6, 1e-5]

        automation.run_multiple_trainings(batch_sizes, epochs_list, learning_rates)

    elif mode == "single":

        batch_size = 6
        epochs = 200
        learning_rate = 1e-7
        automation.run_single_training(batch_size, epochs, learning_rate)
    else:
        print("Invalid mode selected. Use 'single' or 'multiple'.")