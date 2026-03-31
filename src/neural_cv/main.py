from pathlib import Path
from neural_cv.train.trainer import Trainer
import torch
from datetime import datetime
from neural_cv.data.databuilder import DataBuilder
from neural_cv.model.cnn import ResNet
from neural_cv.model.vit import ViT


def main():
    print("Hello from neural-networks-classification!")

    data_dir = "/home/rabah/data/Paysages/seg_train"

    batch_size = 32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 🔹 1. Data
    builder = DataBuilder(data_dir=data_dir, batch_size=batch_size, sample_ratio=0.1)
    datasets = builder.build_dataset()

    mode = "texture"

    train_loader = datasets[mode]["train"]   
    val_loader = datasets[mode]["validation"]
    test_loader = datasets[mode]["test"]

    # 🔹 2. Model
    model = ViT(num_classes=4).to(device)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_name = model.__class__.__name__ 

    run_name = f"{timestamp}_{model_name}_{mode}"

    output_dir = Path("/home/rabah/Projets/neural-networks-classification/results") / run_name

    trainer = Trainer(model=model, train_loader=train_loader, validation_loader=val_loader, test_loader=test_loader, num_epochs=2, output_dir=output_dir,  device=device)

    test_loss, test_accuracy = trainer.train_validate()

    print("\nFinal Test Results")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")


if __name__ == "__main__":
    main()

