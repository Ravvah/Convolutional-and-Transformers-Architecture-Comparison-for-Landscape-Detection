from neural_cv.train.trainer import Trainer
import torch
from neural_cv.data.databuilder import DataBuilder
from neural_cv.model.cnn import ResNet


def main():
    print("Hello from neural-networks-classification!")

    data_dir = "/home/rabah/data/Paysages/seg_train"
    output_dir = "/home/rabah/Projets/neural-networks-classification/results"
    batch_size = 32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 🔹 1. Data
    builder = DataBuilder(data_dir=data_dir, batch_size=batch_size)
    datasets = builder.build_dataset()

    train_loader = datasets["texture"]["train"]   
    val_loader = datasets["texture"]["validation"]
    test_loader = datasets["texture"]["test"]

    # 🔹 2. Model
    model = ResNet(num_classes=4).to(device)

    trainer = Trainer(model=model, train_loader=train_loader, validation_loader=val_loader, test_loader=test_loader, num_epochs=2, output_dir=output_dir,  device=device)

    test_loss, test_accuracy = trainer.train_validate()

    print("\nFinal Test Results")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")


if __name__ == "__main__":
    main()

