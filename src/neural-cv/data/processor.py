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
    def __init__(
        self,
        mode: str,
        use_patch_shuffle: bool = False,
        use_high_pass_filter: bool = False,
        use_random_erasing: bool = False
    ):
        self.mode = mode
        self.use_patch_shuffle = use_patch_shuffle
        self.use_high_pass_filter = use_high_pass_filter
        self.use_random_erasing = use_random_erasing

        if mode == "texture":
            transforms_list = [
                transforms.Resize(256),
                transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
                transforms.RandomGrayscale(p=0.5),
                transforms.ColorJitter(
                    brightness=0.3,
                    contrast=0.3,
                    saturation=0.1,
                    hue=0.05
                ),
                transforms.ToTensor(),
            ]

        elif mode == "global":
            transforms_list = [
                transforms.Resize(256),
                transforms.RandomResizedCrop(
                    size=224,
                    scale=(0.80, 1.0),
                    ratio=(0.95, 1.05)
                ),
                transforms.Resize(112, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.Resize(224, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.GaussianBlur(
                    kernel_size=7,
                    sigma=(0.8, 1.5)
                ),
                transforms.ColorJitter(
                    brightness=0.08,
                    contrast=0.05,
                    saturation=0.05,
                    hue=0.02
                ),
                transforms.ToTensor(),
            ]

        else:
            raise ValueError(f"mode inconnu : {mode}")


        if self.use_high_pass_filter:
            transforms_list.append(
                transforms.Lambda(lambda x: self._high_pass_filter(x, alpha=2.0))
            )

        if self.use_patch_shuffle:
            transforms_list.append(
                transforms.Lambda(lambda x: self._patch_shuffle(x, patch_size=32))
            )
        
        if self.use_random_erasing:
            transforms_list.append(
                transforms.RandomErasing(p=0.5,
                            scale=(0.02, 0.2),
                            ratio=(0.3, 3.3))
            )

        transforms_list.append(
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        )

        self.transform = transforms.Compose(transforms_list)

    def __call__(self, image):
        return self.transform(image)
    
    @staticmethod
    def create_datasets(data_dir: str) -> Tuple[ImageFolder, ImageFolder]:
        texture_transform = Processor(mode="texture")
        global_transform = Processor(mode="global")

        dataset_texture = ImageFolder(data_dir, transform=texture_transform)
        dataset_global = ImageFolder(data_dir, transform=global_transform)

        return dataset_texture, dataset_global
    
    def get_datasets_frequencies(self, datasets: Dict[ImageFolder]) -> Dict[str, float]:
        results = {}
        for name, dataset in datasets.items():
            dataset_scores = []
            for image, _ in dataset:
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
            results[name] = mean_score
        return results

    def _denormalize_image(self, image: Tensor) -> Tensor:
        mean = torch.tensor(data=[0.485, 0.456, 0.406]).cpu().view(3, 1, 1)
        std = torch.tensor(data=[0.229, 0.224, 0.225]).cpu().view(3, 1, 1)

        image = image * std + mean

        return image
    
    def _patch_shuffle(self, image: Tensor, patch_size=32) -> Tensor:
        C, H, W = image.shape

        # nombre de patches
        n_h = H // patch_size
        n_w = W // patch_size

        # reshape en grille
        image = image.view(C, n_h, patch_size, n_w, patch_size)
        image = image.permute(1, 3, 0, 2, 4)  # (n_h, n_w, C, p, p)

        # flatten patches
        patches = image.reshape(-1, C, patch_size, patch_size)

        # shuffle
        idx = torch.randperm(patches.size(0))
        patches = patches[idx]

        # reconstruction
        patches = patches.view(n_h, n_w, C, patch_size, patch_size)
        image = patches.permute(2, 0, 3, 1, 4).contiguous()

        return image.view(C, H, W)
    
    def _high_pass_filter(self, image: Tensor, alpha=0.2) -> Tensor:
        blur = transforms.GaussianBlur(kernel_size=11, sigma=(0.3, 5.0))(image)
        image_hp = image - blur

        image_hp = image_hp * alpha

        image_hp = (image_hp - image_hp.min()) / (image_hp.max() - image_hp.min() + 1e-8)

        return image_hp

    def plot_processing_example(self, original_data_dir: str, texture_dataset: ImageFolder, global_dataset: ImageFolder) -> None:
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
    print("Not runnable yet !")
    # data_dir = "/home/rabah/data/Paysages/seg_train"
    # processor = Processor(mode="texture")
    # texture_dataset, global_dataset = Processor.create_datasets(data_dir=data_dir)
    # processor.plot_processing_example(original_data_dir=data_dir, texture_dataset=texture_dataset, global_dataset=global_dataset)
    # # frequencies = processor.get_datasets_frequencies({"texture": texture_dataset, "global": global_dataset})
    # print(frequencies)