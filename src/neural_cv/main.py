from pathlib import Path
from neural_cv.train.trainer import Trainer
import torch
from datetime import datetime
from neural_cv.data.databuilder import DataBuilder
from neural_cv.model.cnn import ResNet
from neural_cv.model.vit import ViT

from loguru import logger


import numpy as np
import random


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    print("Hello from neural-networks-classification!")

    seeds = [42, 123, 999]

    all_acc = []
    all_f1 = []
    all_loss = []

    for seed in seeds:
        print(f"\n==== RUN WITH SEED {seed} ====")
        set_seed(seed)

        data_dir = "/content/drive/MyDrive/Deep_Learning/Data/seg_train"
        # data_dir = "/home/rabah/data/Paysages/seg_train"

        batch_size = 32
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        builder = DataBuilder(data_dir=data_dir, batch_size=batch_size)
        datasets = builder.build_dataset()

        mode = "texture"

        train_loader = datasets[mode]["train"]
        val_loader = datasets[mode]["validation"]
        test_loader = datasets[mode]["test"]

        model = ViT(num_classes=4).to(device)

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_name = f"{timestamp}_seed{seed}_ViT_{mode}"

        # output_dir = Path("/home/rabah/Projets/neural-networks-classification/results") /  "cnn" / run_name
        output_dir = Path("/content/drive/MyDrive/Deep_Learning/Data/results") / "cnn" / run_name
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            validation_loader=val_loader,
            test_loader=test_loader,
            num_epochs=25,
            output_dir=output_dir,
            device=device
        )

        test_loss, test_accuracy, test_f1 = trainer.train_validate()

        all_loss.append(test_loss)
        all_acc.append(test_accuracy)
        all_f1.append(test_f1)

    logger.info("\n===== FINAL RESULTS =====")
    logger.info(f"Accuracy: {np.mean(all_acc):.4f} ± {np.std(all_acc):.4f}")
    logger.info(f"F1-score: {np.mean(all_f1):.4f} ± {np.std(all_f1):.4f}")


if __name__ == "__main__":
    main()