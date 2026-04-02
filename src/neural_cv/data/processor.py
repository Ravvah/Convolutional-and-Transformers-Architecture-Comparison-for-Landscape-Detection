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
    """
    Class for data tranqsformations
    """
    def __init__(
        self,
        mode: str,
        use_patch_shuffle: bool = False,
        use_high_pass_filter: bool = False,
        use_local_contrast_normalization: bool = False,
        use_random_erasing: bool = False,
        use_random_patch_drop: bool = False,
        use_local_patch_shuffle: bool = False

    ):
        self.mode = mode
        self.use_patch_shuffle = use_patch_shuffle
        self.use_high_pass_filter = use_high_pass_filter
        self.use_local_contrast_normalization = use_local_contrast_normalization
        self.use_random_erasing = use_random_erasing
        self.use_random_patch_drop = use_random_patch_drop
        self.use_local_patch_shuffle = use_local_patch_shuffle
        self.transforms_list = []

        if mode == "base":
            self.transforms_list = [
                transforms.Resize(224),
                transforms.ToTensor()
            ]

        elif mode == "texture":
            self.transforms_list = [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.Grayscale(num_output_channels=3),
                # transforms.ColorJitter(
                #     brightness=0.3,
                #     contrast=0.3,
                #     saturation=0.1,
                #     hue=0.05
                # ),
                transforms.ToTensor(),
            ]

        elif mode == "global":
            self.transforms_list = [
                transforms.Resize(256),
                transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
                transforms.Resize(102, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.Resize(224, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.Grayscale(num_output_channels=3),

                transforms.GaussianBlur(kernel_size=9, sigma=(1.5, 2.0)),

                # transforms.ColorJitter(
                #     brightness=0.08,
                #     contrast=0.05,
                #     saturation=0.05,
                #     hue=0.02
                # ),
                transforms.ToTensor(),
            ]

        else:
            raise ValueError(f"mode inconnu : {mode}")


        if self.use_high_pass_filter:
            self.transforms_list.append(
                transforms.Lambda(lambda x: self._high_pass_filter(x, alpha=20.0))
            )
        
        if self.use_local_contrast_normalization:
            self.transforms_list.append(
                transforms.Lambda(lambda x: self.local_contrast_norm_v2(x)),

            )

        if self.use_patch_shuffle:
            self.transforms_list.append(
                transforms.Lambda(lambda x: self._global_patch_shuffle_partial(x, patch_size=8, shuffle_ratio=0.57))
            )
        
        if self.use_local_contrast_normalization:
            self.transforms_list.extend([
                transforms.Lambda(lambda x: self.local_contrast_norm_v2(x)),
                transforms.Lambda(lambda x: x - 0.3 * transforms.GaussianBlur(21, sigma=5.0)(x))
            ]
            )
        
        if self.use_random_erasing:
            self.transforms_list.append(
                transforms.RandomErasing(p=0.5,
                            scale=(0.02, 0.2),
                            ratio=(0.3, 3.3))
            )

        if self.use_random_patch_drop:
            self.transforms_list.append(
                transforms.Lambda(lambda x: self.random_patch_drop_single(x))
                )
        if self.use_local_patch_shuffle:
            self.transforms_list.append(
                transforms.Lambda(lambda x: self.local_patch_permutation(x, patch_size=8, permute_prob=0.20)),

            )

        self.transforms_list.append(
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
    
        )

        self.transform = transforms.Compose(self.transforms_list)

    def __call__(self, image):
        return self.transform(image)
    
    @staticmethod
    def create_datasets(data_dir: str) -> Tuple[ImageFolder, ImageFolder]:
        texture_transform = Processor(mode="texture")
        global_transform = Processor(mode="global")

        dataset_texture = ImageFolder(data_dir, transform=texture_transform)
        dataset_global = ImageFolder(data_dir, transform=global_transform)

        return dataset_texture, dataset_global
    
    # def get_datasets_frequencies(self, datasets: Dict[str, ImageFolder]) -> Dict[str, float]:
    #     results = {}
    #     for name, dataset in datasets.items():
    #         dataset_scores = []
    #         for image, _ in dataset:
    #             image_np = image.cpu().numpy()
    #             gray = image_np.mean(axis=0)
    #             fft = np.fft.fft2(gray)
    #             fft_shift = np.fft.fftshift(fft)
    #             energy = np.abs(fft_shift) ** 2
    #             total_energy = energy.sum()

    #             H, W = gray.shape
    #             center_x, center_y = H // 2, W // 2
    #             y, x = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    #             dist = np.sqrt((y - center_x)**2 + (x - center_y)**2)

    #             threshold = min(H, W) * 0.25
    #             mask = dist > threshold
    #             high_freq_energy = energy[mask].sum()

    #             ratio = high_freq_energy / (total_energy + 1e-8)
    #             dataset_scores.append(ratio)

    #         mean_score = np.mean(dataset_scores)
    #         results[name] = mean_score
    #     return results

    def compute_texture_score(self, image: torch.Tensor) -> float:
        img = image.cpu().numpy()
        gray = img.mean(axis=0)

        fft = np.fft.fft2(gray)
        fft_shift = np.fft.fftshift(fft)
        energy = np.abs(fft_shift) ** 2

        H, W = gray.shape
        cx, cy = H // 2, W // 2

        y, x = np.ogrid[:H, :W]
        dist = np.sqrt((x - cx)**2 + (y - cy)**2)

        r_low = int(min(H, W) * 0.05)
        r_high = int(min(H, W) * 0.25)

        mask = (dist > r_low) & (dist < r_high)

        band_energy = energy[mask].sum()
        total_energy = energy.sum()

        return band_energy / (total_energy + 1e-8)

    def _denormalize_image(self, image: Tensor) -> Tensor:
        mean = torch.tensor(data=[0.485, 0.456, 0.406]).cpu().view(3, 1, 1)
        std = torch.tensor(data=[0.229, 0.224, 0.225]).cpu().view(3, 1, 1)

        image = image * std + mean

        return image
    
    def _global_patch_shuffle_partial(self, image: Tensor, patch_size=32, shuffle_ratio=0.6):
        C, H, W = image.shape

        n_h = H // patch_size
        n_w = W // patch_size

        image = image.view(C, n_h, patch_size, n_w, patch_size)
        image = image.permute(1, 3, 0, 2, 4)

        patches = image.reshape(-1, C, patch_size, patch_size)

        num_patches = patches.size(0)
        num_shuffle = int(shuffle_ratio * num_patches)

        shuffle_indices = torch.randperm(num_patches)[:num_shuffle]

        permuted = shuffle_indices[torch.randperm(num_shuffle)]

        patches[shuffle_indices] = patches[permuted]

        patches = patches.view(n_h, n_w, C, patch_size, patch_size)
        image = patches.permute(2, 0, 3, 1, 4).contiguous()

        return image.view(C, H, W)
    
    def local_patch_permutation(self, x, patch_size=16, permute_prob=0.2):
        C, H, W = x.shape

        for i in range(0, H, patch_size):
            for j in range(0, W, patch_size):
                if np.random.rand() < permute_prob:
                    i2 = np.random.randint(0, H // patch_size) * patch_size
                    j2 = np.random.randint(0, W // patch_size) * patch_size

                    patch1 = x[:, i:i+patch_size, j:j+patch_size].clone()
                    patch2 = x[:, i2:i2+patch_size, j2:j2+patch_size].clone()

                    x[:, i:i+patch_size, j:j+patch_size] = patch2
                    x[:, i2:i2+patch_size, j2:j2+patch_size] = patch1

        return x
    
    def _high_pass_filter(self, image: Tensor, alpha=0.2) -> Tensor:
        blur1 = transforms.GaussianBlur(7, sigma=1.0)(image)
        blur2 = transforms.GaussianBlur(9, sigma=2.5)(image)
        blur_global = transforms.GaussianBlur(15, sigma=5.0)(image)

        detail1 = image - blur1
        detail2 = image - blur2

        enhanced = image + 1.2 * (detail1 + 0.5 * detail2)

        enhanced = enhanced - 0.3 * blur_global

        enhanced = torch.clamp(enhanced, 0.0, 1.0)
        enhanced = enhanced ** 0.8

        return enhanced
    
    def local_contrast_norm_v2(self, image: Tensor):
        mean = image.mean(dim=(1,2), keepdim=True)
        std = image.std(dim=(1,2), keepdim=True)

        out = (image - mean) / (std + 1e-8)

        out = torch.sign(out) * torch.sqrt(torch.abs(out))

        return out
    def random_patch_drop_single(self, x, drop_prob=0.5, patch_size=24):
        C, H, W = x.shape

        n_patches_h = H // patch_size
        n_patches_w = W // patch_size

        for _ in range(int(drop_prob * n_patches_h * n_patches_w)):
            i = np.random.randint(0, n_patches_h) * patch_size
            j = np.random.randint(0, n_patches_w) * patch_size

            x[:, i:i+patch_size, j:j+patch_size] = 0

        return x
    
    def random_multi_patch_drop_single(self, x, drop_prob=0.35, patch_size=16):
        C, H, W = x.shape
        n_h = H // patch_size
        n_w = W // patch_size
        n_total = n_h * n_w
        n_drop = int(drop_prob * n_total)

        coords = [(i, j) for i in range(n_h) for j in range(n_w)]
        random.shuffle(coords)

        for i, j in coords[:n_drop]:
            y0 = i * patch_size
            x0 = j * patch_size
            x[:, y0:y0+patch_size, x0:x0+patch_size] = 0.0

        return x
    
    def compute_fft_spectrum(self, image: torch.Tensor) -> np.ndarray:
        img = image.detach().cpu().numpy()
        gray = img.mean(axis=0)

        fft = np.fft.fft2(gray)
        fft_shift = np.fft.fftshift(fft)

        spectrum = np.log(1 + np.abs(fft_shift))
        return spectrum

    def plot_deterministic_transformations(self, image_pil):
        """
        Visualize transformations
        
        :param self: Description
        :param image_pil: Description
        """
        transforms_to_apply = {
            "Original": lambda x: x,
            # "Resize": transforms.Resize(224),
            # "CenterCrop": transforms.CenterCrop(224),
            # "Grayscale": transforms.Grayscale(num_output_channels=3),
            # "ColorJitter": transforms.ColorJitter(
            #     brightness=0.3, contrast=0.3, saturation=0.1, hue=0.05),
            "Texture Transformation": transforms.Compose([
                # transforms.ColorJitter(
                #     brightness=0.5,
                #     contrast=0.9,
                #     saturation=0.1,
                #     hue=0.05
                # ),
                transforms.Resize(224),
                # transforms.CenterCrop(224),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                # transforms.Lambda(lambda x: torch.clamp(x, 0, 1)),
                transforms.Lambda(lambda x: self._high_pass_filter(x, alpha=35.0)),
                transforms.Lambda(lambda x: self._global_patch_shuffle_partial(x, patch_size=8, shuffle_ratio=0.4)),
                transforms.Lambda(lambda x: self.local_contrast_norm_v2(x)),
                transforms.Lambda(lambda x: x - 0.6 * transforms.GaussianBlur(21, sigma=5.0)(x))]),
                
            "Pipeline_2": transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
    transforms.Resize(115, interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.Resize(224, interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.Grayscale(num_output_channels=3),
    transforms.GaussianBlur(kernel_size=7, sigma=(1.5, 2.0)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: self.local_patch_permutation(x, patch_size=8, permute_prob=0.15)),
    transforms.Lambda(lambda x: self.random_patch_drop_single(x, drop_prob=0.1, patch_size=8)),

])
        }

        fig, axes = plt.subplots(3, len(transforms_to_apply), figsize=(15, 9))
        for i, (name, transform) in enumerate(transforms_to_apply.items()):
            img = transform(image_pil)

            if isinstance(img, torch.Tensor):
                score = self.compute_texture_score(img)

                img_np = img.permute(1, 2, 0).cpu().numpy()
            else:
                img_tensor = transforms.ToTensor()(img)
                score = self.compute_texture_score(img_tensor)

                img_np = np.array(img)

            # image
            axes[0, i].imshow(img_np)
            axes[0, i].set_title(name)
            axes[0, i].axis("off")

            # score
            axes[1, i].text(0.5, 0.5, f"{score:.3f}", ha="center", va="center", fontsize=12)
            axes[1, i].set_title("Texture score")
            axes[1, i].axis("off")

            spectrum = self.compute_fft_spectrum(img if isinstance(img, torch.Tensor) else img_tensor)

            axes[2, i].imshow(spectrum, cmap='inferno')
            axes[2, i].set_title("FFT spectrum")
            axes[2, i].set_xlabel("Frequency")
            axes[2, i].set_ylabel("Frequency")

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    data_dir = "/home/rabah/data/Paysages/seg_train"
    processor = Processor(mode="global", use_patch_shuffle=True)
    # processor.plot_processing_example(original_data_dir=data_dir)
    dataset = ImageFolder(data_dir, transform=None)
    image, label = dataset[502]
    # print(image)

    processor.plot_deterministic_transformations(image)

