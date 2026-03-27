from typing import Dict, List, Tuple
import torch
from torch import Tensor
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from matplotlib import pyplot as plt
import random

import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder


class Processor:
    def __init__(self, mode):
        self.mode = mode

        if mode == "texture":
            self.transform = transforms.Compose(transforms=[
                transforms.Resize(size=224),
                transforms.RandomCrop(size=64),
                transforms.Resize(size=224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
            ])
        
        elif mode == "global":
            self.transform = transforms.Compose(transforms=[
                transforms.Resize(size=224),
                transforms.GaussianBlur(kernel_size=5),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
            ])
        else :
            raise ValueError(f"mode spécifié inconnu : {mode}")
        
    def __call__(self, image) -> transforms.Compose:
        return self.transform(image)
    
    def create_datasets(data_dir: str) -> Tuple[ImageFolder, ImageFolder]:
        texture_transform = Processor(mode="texture")
        global_transform = Processor(mode="global")

        dataset_texture = ImageFolder(data_dir, transform=texture_transform)
        dataset_global = ImageFolder(data_dir, transform=global_transform)

        return dataset_texture, dataset_global
    
    def get_datasets_frequencies(self, datasets: List[ImageFolder]) -> Dict[str, float]:
        results = {}
        for dataset in datasets:
            dataset_scores = []
            for image, label in dataset:
                image_np = image.cpu().numpy()
                gray = image_np.mean(axis=0)
                fft = np.fft.fft2(gray)
                fft_shift = np.fft.fftshift(fft)
                energy = np.abs(fft_shift) ** 2
                total_energy = energy.sum()

                H, W = gray.shape
                center_x, center_y = H // 2, W // 2
                y, x = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
                dist = np.sqrt((y - center_x)**2 + (x - center_y)**2)

                threshold = min(H, W) * 0.25
                mask = dist > threshold
                high_freq_energy = energy[mask].sum()

                ratio = high_freq_energy / (total_energy + 1e-8)
                dataset_scores.append(ratio)
            mean_score = np.mean(dataset_scores)

        results[label] = mean_score

    def _denormalize_image(self, image: Tensor) -> Tensor:
        mean = torch.tensor(data=[0.485, 0.456, 0.406]).cpu().view(3, 1, 1)
        std = torch.tensor(data=[0.229, 0.224, 0.225]).cpu().view(3, 1, 1)

        image = image * std + mean

        return image
    
    def plot_processing_example(self, original_data_dir: str, texture_dataset, global_dataset) -> None:
        original_dataset = ImageFolder(root=original_data_dir, transform=transforms.ToTensor())

        random_index = random.randint(a=0, b=len(original_dataset) - 1)

        original_image, original_label = original_dataset[random_index]
        texture_image, _ = texture_dataset[random_index]
        global_image, _ = global_dataset[random_index]

        # original_image = self._denormalize_image(original_image)
        texture_image = self._denormalize_image(texture_image)
        global_image = self._denormalize_image(global_image)

        original_image = original_image.permute(1, 2, 0).cpu().numpy()
        texture_image = texture_image.permute(1, 2, 0).cpu().numpy()
        global_image = global_image.permute(1, 2, 0).cpu().numpy()
        
        fig, axes = plt.subplots(ncols=3, nrows=1)
        axes[0].imshow(original_image)
        axes[1].imshow(texture_image)
        axes[2].imshow(global_image)
        axes[0].set_title("Original")
        axes[1].set_title("Image after texture transformation")
        axes[2].set_title("Image after global transformation")

        fig.tight_layout()
        plt.show()





if __name__ == "__main__":
    data_dir = "/home/rabah/data/Paysages/seg_train"
    processor = Processor(mode="texture")
    texture_dataset, global_dataset = Processor.create_datasets(data_dir=data_dir)
    processor.plot_processing_example(original_data_dir=data_dir, texture_dataset=texture_dataset, global_dataset=global_dataset)

        