import json
from typing import Tuple
import torch
from torch.nn import Module, CrossEntropyLoss
from torch.utils.data import DataLoader
import torch.optim as opt
from loguru import logger
from pathlib import Path


class Trainer:

    def __init__(self, model: Module, train_loader: DataLoader, validation_loader: DataLoader, test_loader: DataLoader, output_dir: str, num_epochs: int = 25, device = "cpu"):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.test_loader = test_loader
        self.criterion = CrossEntropyLoss()

        self.optimizer = opt.AdamW(params=self.model.parameters(), lr=3e-5)
        self.num_epochs = num_epochs
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.history = {
            "train_loss": [],
            "val_loss": [],
            "val_accuracy": [],
        }

        self.best_val_accuracy = 0.0

    
    def train_one_epoch(self) -> float:
        self.model.train(mode=True)
        total_loss = 0

        for x, y in self.train_loader:
            x, y = x.to(self.device), y.to(self.device)

            self.optimizer.zero_grad()

            logits = self.model(x)
            loss = self.criterion(logits, y)

            loss.backward()

            self.optimizer.step()

            total_loss += loss.item()




        return total_loss / len(self.train_loader)
        
    def evaluate(self, loader: DataLoader) -> Tuple[float, float]:
        self.model.eval()

        total = 0
        correct = 0
        total_loss = 0

        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)

                logits = self.model(x)
                loss = self.criterion(logits, y)

                total_loss += loss.item()

                predictions = torch.argmax(logits, dim=1)

                correct += (predictions == y).sum().item()
                total += y.size(0)
            
        accuracy = correct / total

        return total_loss / len(loader), accuracy


        
    def train_validate(self):
        for epoch in range(self.num_epochs):
            train_loss = self.train_one_epoch()
            validation_loss, validation_accuracy = self.evaluate(loader=self.validation_loader)

                        # Tracking
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(validation_loss)
            self.history["val_accuracy"].append(validation_accuracy)

            logger.info(f"Epoch : {epoch}")
            logger.info(f"Train Loss : {train_loss}")
            logger.info(f"Validation Loss : {validation_loss} ---- Validation Accuracy : {validation_accuracy}")

            if validation_accuracy > self.best_val_accuracy:
                self.best_val_accuracy = validation_accuracy
                torch.save(self.model.state_dict(), self.output_dir / "best_model.pt")

        self.model.load_state_dict(torch.load(self.output_dir / "best_model.pt"))

        test_loss, test_accuracy = self.evaluate(loader=self.test_loader)

        logger.info(f"Test Loss : {test_loss} ---- Final Test Accuracy : {test_accuracy}")

        results = {
            "test_loss": test_loss,
            "test_accuracy": test_accuracy,
            "history": self.history,
                   }
        
        self.save_results(results=results)
        return test_loss, test_accuracy
    
    def save_results(self, results: dict):
        with open(self.output_dir / "results.json" , "w") as f:
            json.dump(results, f, indent=4)

            



            
        