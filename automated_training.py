import os
import json
import matplotlib.pyplot as plt
from itertools import product
from datetime import datetime
from unet_trainer import UNetTrainer

class AutomatedTraining:
    """
    Automated training class for running multiple training sessions with varying hyperparameters.
    """

    def __init__(self, base_config):
        """
        Initialize the AutomatedTraining class.

        Args:
            base_config (dict): Base configuration for the training sessions.
        """
        self.base_config = base_config
        # Create a results directory if it doesn't exist
        self.results_path = base_config.get("results_path", "./training_results")
        os.makedirs(self.results_path, exist_ok=True)

    def run_multiple_trainings(self, batch_sizes, epochs_list, learning_rates):
        """
        Run multiple training sessions with different combinations of hyperparameters.

        Args:
            batch_sizes (list): List of batch sizes to test.
            epochs_list (list): List of epoch counts to test.
            learning_rates (list): List of learning rates to test.
        """
        # Generate all combinations of hyperparameters
        combinations = list(product(batch_sizes, epochs_list, learning_rates))
        results_summary = []

        for i, (batch_size, epochs, lr) in enumerate(combinations, 1):
            print(f"Running Training {i}/{len(combinations)}: "
                  f"Batch Size={batch_size}, Epochs={epochs}, Learning Rate={lr}")

            # Create a timestamped folder name
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            training_name = f"batch_{batch_size}_epochs_{epochs}_lr_{lr}_{timestamp}"
            training_path = os.path.join(self.results_path, training_name)
            os.makedirs(training_path, exist_ok=True)

            # Update the configuration for this training session
            config = self.base_config.copy()
            config.update({
                "batch_size": batch_size,
                "epochs": epochs,
                "optimizer": config["optimizer"](config["model"].parameters(), lr=lr),
                "save_path": training_path
            })


            trainer = UNetTrainer(config)
            trainer.train()

            # Log results for this training session
            results_summary.append({
                "training_name": training_name,
                "batch_size": batch_size,
                "epochs": epochs,
                "learning_rate": lr,
                "save_path": training_path
            })


        summary_path = os.path.join(self.results_path, "summary.json")
        with open(summary_path, "w") as f:
            json.dump(results_summary, f, indent=4)


        self.plot_summary(results_summary)

    def run_single_training(self, batch_size, epochs, learning_rate):
        """
        Run a single training session with specified parameters.

        Args:
            batch_size (int): Batch size to use.
            epochs (int): Number of epochs to train.
            learning_rate (float): Learning rate to use.
        """
        print(f"Running Single Training: Batch Size={batch_size}, Epochs={epochs}, Learning Rate={learning_rate}")


        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        training_name = f"single_batch_{batch_size}_epochs_{epochs}_lr_{learning_rate}_{timestamp}"
        training_path = os.path.join(self.results_path, training_name)
        os.makedirs(training_path, exist_ok=True)


        config = self.base_config.copy()
        config.update({
            "batch_size": batch_size,
            "epochs": epochs,
            "optimizer": config["optimizer"](config["model"].parameters(), lr=learning_rate),  # Pass lr dynamically
            "save_path": training_path
        })

        # Train the model
        trainer = UNetTrainer(config)
        trainer.train()

    def plot_summary(self, results_summary):
        """
        Generate a summary plot showing validation MSE trends for all training sessions.

        Args:
            results_summary (list): List of dictionaries containing training details.
        """
        plt.figure(figsize=(10, 6))
        for result in results_summary:
            training_path = result["save_path"]
            metrics_file = os.path.join(training_path, "metrics.json")

            if os.path.exists(metrics_file):
                with open(metrics_file, "r") as f:
                    metrics = json.load(f)

                epochs = range(1, len(metrics["val_MSE"]) + 1)
                val_MSE = metrics["val_MSE"]
                plt.plot(epochs, val_MSE, label=result["training_name"])

        plt.xlabel("Epoch")
        plt.ylabel("Validation wMSE")
        plt.title("Validation wMSE Across Trainings")
        plt.legend()
        plt.grid(True)

        plot_path = os.path.join(self.results_path, "training_summary.png")
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.show()


