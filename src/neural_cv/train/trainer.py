import json
from typing import Tuple
import torch
from torch.nn import Module, CrossEntropyLoss
from torch.utils.data import DataLoader
import torch.optim as opt
from loguru import logger
from pathlib import Path
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

import multiprocessing
multiprocessing.set_start_method("fork", force=True)

class Trainer:

    def __init__(self, model: Module, train_loader: DataLoader, validation_loader: DataLoader, test_loader: DataLoader, output_dir: str, num_epochs: int = 25, device = "cpu"):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.test_loader = test_loader
        self.criterion = CrossEntropyLoss()

        self.optimizer = opt.AdamW(params=self.model.parameters(), lr=3e-5)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=2  
        )
        self.num_epochs = num_epochs
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.history = {
            "train_loss": [],
            "val_loss": [],
            "val_accuracy": [],
            "val_f1": []
        }

        self.best_val_accuracy = 0.0
        self.best_val_loss = float("inf")
        self.patience = 3
        self.counter = 0

    
    def train_one_epoch(self) -> float:
        self.model.train()
        total_loss = 0
        total_samples = 0

        for x, y in self.train_loader:
            x, y = x.to(self.device), y.to(self.device)

            self.optimizer.zero_grad()

            logits = self.model(x)
            loss = self.criterion(logits, y)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * x.size(0)
            total_samples += x.size(0)

        return total_loss / total_samples
        
    def evaluate(self, loader: DataLoader):
        self.model.eval()

        total_loss = 0
        total_samples = 0
        correct = 0

        all_preds = []
        all_targets = []

        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)

                logits = self.model(x)
                loss = self.criterion(logits, y)

                total_loss += loss.item() * x.size(0)
                total_samples += x.size(0)

                preds = torch.argmax(logits, dim=1)

                correct += (preds == y).sum().item()

                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(y.cpu().numpy())

        avg_loss = total_loss / total_samples
        accuracy = correct / total_samples

        f1 = f1_score(all_targets, all_preds, average="macro")

        return avg_loss, accuracy, f1

        
    def train_validate(self):
        for epoch in range(self.num_epochs):

            train_loss = self.train_one_epoch()
            val_loss, val_accuracy, val_f1 = self.evaluate(self.validation_loader)

            self.scheduler.step(val_loss)

            # Tracking
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_accuracy"].append(val_accuracy)
            self.history["val_f1"].append(val_f1)

            logger.info(f"Epoch : {epoch}")
            logger.info(f"Train Loss : {train_loss:.4f}")
            logger.info(
                f"Validation Loss : {val_loss:.4f} ---- Accuracy : {val_accuracy:.4f} ---- F1 : {val_f1:.4f}"
            )
            improved = False

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                improved = True

            if val_accuracy > self.best_val_accuracy:
                self.best_val_accuracy = val_accuracy
                improved = True

            if improved:
                torch.save(self.model.state_dict(), self.output_dir / "best_model.pt")
                self.counter = 0
            else:
                self.counter += 1

            if self.counter >= self.patience:
                logger.info("Early stopping triggered")
                break

        self.model.load_state_dict(torch.load(self.output_dir / "best_model.pt"))

        test_loss, test_accuracy, test_f1 = self.evaluate(self.test_loader)

        logger.info(
    f"Test Loss : {test_loss:.4f} ---- Accuracy : {test_accuracy:.4f} ---- F1 : {test_f1:.4f}"
)

        results = {
            "test_loss": test_loss,
            "test_accuracy": test_accuracy,
            "best_val_accuracy": self.best_val_accuracy,
            "best_val_loss": self.best_val_loss,
            "history": self.history,
        }
        self.plot_training_curves()
        self.save_results(results)
        return test_loss, test_accuracy, test_f1
    
    def plot_training_curves(self):
        epochs = range(len(self.history["train_loss"]))

        plt.figure(figsize=(12, 5))

        # 🔹 Loss
        plt.subplot(1, 2, 1)
        plt.plot(epochs, self.history["train_loss"], label="Train Loss")
        plt.plot(epochs, self.history["val_loss"], label="Val Loss")
        plt.title("Loss (Overfitting detection)")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()

        # 🔹 Accuracy / F1
        plt.subplot(1, 2, 2)
        plt.plot(epochs, self.history["val_accuracy"], label="Val Accuracy")
        plt.plot(epochs, self.history["val_f1"], label="Val F1")
        plt.title("Validation Metrics")
        plt.xlabel("Epoch")
        plt.ylabel("Score")
        plt.legend()

        plt.tight_layout()

        # 🔥 sauvegarde
        plt.savefig(self.output_dir / "training_curves.png")
        plt.close()
    
    def save_results(self, results: dict):
        with open(self.output_dir / "results.json" , "w") as f:
            json.dump(results, f, indent=4)

            



            
        