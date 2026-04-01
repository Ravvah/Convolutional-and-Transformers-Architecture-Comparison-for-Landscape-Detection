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
        self.transforms_list = []

        if mode == "texture":
            self.transforms_list = [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.Grayscale(num_output_channels=3),
                transforms.ColorJitter(
                    brightness=0.3,
                    contrast=0.3,
                    saturation=0.1,
                    hue=0.05
                ),
                transforms.ToTensor(),
            ]

        elif mode == "global":
            self.transforms_list = [
                transforms.Resize(256),
                transforms.CenterCrop(size=224),
                transforms.Resize(112, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.Resize(224, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.GaussianBlur(
                    kernel_size=11,
                    sigma=(2.0, 3.0)
                ),
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
                transforms.Lambda(lambda x: self._high_pass_filter(x, alpha=2.0))
            )

        if self.use_patch_shuffle:
            self.transforms_list.append(
                transforms.Lambda(lambda x: self._patch_shuffle_partial(x, patch_size=32, shuffle_ratio=0.4))
            )
        
        if self.use_random_erasing:
            self.transforms_list.append(
                transforms.RandomErasing(p=0.5,
                            scale=(0.02, 0.2),
                            ratio=(0.3, 3.3))
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
    
    def get_datasets_frequencies(self, datasets: Dict[str, ImageFolder]) -> Dict[str, float]:
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
    
    def _patch_shuffle_partial(self, image: Tensor, patch_size=32, shuffle_ratio=0.6):
        C, H, W = image.shape

        n_h = H // patch_size
        n_w = W // patch_size

        image = image.view(C, n_h, patch_size, n_w, patch_size)
        image = image.permute(1, 3, 0, 2, 4)  # (n_h, n_w, C, p, p)

        patches = image.reshape(-1, C, patch_size, patch_size)

        num_patches = patches.size(0)
        num_shuffle = int(shuffle_ratio * num_patches)

        # indices à mélanger
        shuffle_indices = torch.randperm(num_patches)[:num_shuffle]

        # copie pour éviter overwrite
        patches_shuffled = patches.clone()

        permuted = shuffle_indices[torch.randperm(num_shuffle)]

        patches_shuffled[shuffle_indices] = patches[permuted]

        # reconstruction
        patches_shuffled = patches_shuffled.view(n_h, n_w, C, patch_size, patch_size)
        image = patches_shuffled.permute(2, 0, 3, 1, 4).contiguous()

        return image.view(C, H, W)
    
    def _high_pass_filter(self, image: Tensor, alpha=0.2) -> Tensor:
        # blur = transforms.GaussianBlur(kernel_size=7, sigma=(0.5, 1.0))(image)
        # image_hp = image - blur
        

        # image_hp = image_hp * alpha
        # image_hp = torch.abs(image_hp)


        # image_hp = (image_hp - image_hp.min()) / (image_hp.max() - image_hp.min() + 1e-8)

        # image_hp = image_hp ** 0.3

        # return image_hp
        blur1 = transforms.GaussianBlur(7, sigma=1.0)(image)
        blur2 = transforms.GaussianBlur(9, sigma=2.5)(image)
        blur_global = transforms.GaussianBlur(15, sigma=5.0)(image)

        detail1 = image - blur1
        detail2 = image - blur2

        enhanced = image + 1.2 * (detail1 + 0.5 * detail2)

        # 🔥 suppression structure globale
        enhanced = enhanced - 0.3 * blur_global

        # 🔥 non-linéarité texture
        enhanced = torch.clamp(enhanced, 0.0, 1.0)
        enhanced = enhanced ** 0.8

        return enhanced
    
    def local_contrast_norm(self, image):
        mean = image.mean(dim=(1,2), keepdim=True)
        std = image.std(dim=(1,2), keepdim=True)
        return (image - mean) / (std + 1e-8)
    
    def compute_fft_image(self, image_tensor: Tensor) -> np.ndarray:
        image_np = image_tensor.mean(dim=0).cpu().numpy()
        fft = np.fft.fft2(image_np)
        fft_shift = np.fft.fftshift(fft)
        magnitude = np.log(1 + np.abs(fft_shift))
        return magnitude

    def plot_processing_example(self, original_data_dir: str, n_samples: int = 4):
        original_dataset = ImageFolder(root=original_data_dir, transform=None)

        index = random.randint(0, len(original_dataset) - 1)
        image_pil, label = original_dataset[index]

        texture_processor = Processor(mode="texture")
        global_processor = Processor(mode="global", use_patch_shuffle=True)

        fig, axes = plt.subplots(2, n_samples + 1, figsize=(15, 6))

        # original
        axes[0, 0].imshow(np.array(image_pil))
        axes[0, 0].set_title("Original")

        for i in range(n_samples):
            texture_image = texture_processor(image_pil)
            texture_image = self._denormalize_image(texture_image)
            texture_np = texture_image.permute(1, 2, 0).cpu().numpy()

            axes[0, i+1].imshow(texture_np)
            axes[0, i+1].set_title(f"Texture {i}")

        for i in range(n_samples):
            global_image = global_processor(image_pil)
            global_image = self._denormalize_image(global_image)
            global_np = global_image.permute(1, 2, 0).cpu().numpy()

            axes[1, i+1].imshow(global_np)
            axes[1, i+1].set_title(f"Global {i}")

        axes[1, 0].axis("off")

        plt.tight_layout()
        plt.show()

    def compare_pipelines(self, image_pil):
        texture_processor = Processor(
            mode="texture",
            use_patch_shuffle=True,
            use_high_pass_filter=True
        )

        global_processor = Processor(
            mode="global"
        )

        def apply_pipeline(processor, image):
            outputs = []
            current = image

            outputs.append(("Original", np.array(current)))

            for transform in processor.transforms_list:
                current = transform(current)

                name = transform.__class__.__name__

                if isinstance(current, torch.Tensor):
                    img = current.clone()

                    # On dénormalise seulement si on vient de passer Normalize
                    if isinstance(transform, transforms.Normalize):
                        img = processor._denormalize_image(img)

                    img = img.permute(1, 2, 0).cpu().numpy()
                    img = np.clip(img, 0, 1)
                else:
                    img = np.array(current)

                outputs.append((name, img))

            return outputs

        texture_outputs = apply_pipeline(texture_processor, image_pil)
        global_outputs = apply_pipeline(global_processor, image_pil)

        n = max(len(texture_outputs), len(global_outputs))

        fig, axes = plt.subplots(2, n, figsize=(4*n, 8))

        for i in range(n):
            if i < len(texture_outputs):
                name, img = texture_outputs[i]
                axes[0, i].imshow(img)
                axes[0, i].set_title(f"Texture: {name}")
            axes[0, i].axis("off")

            if i < len(global_outputs):
                name, img = global_outputs[i]
                axes[1, i].imshow(img)
                axes[1, i].set_title(f"Global: {name}")
            axes[1, i].axis("off")

        plt.tight_layout()
        plt.show()

    def plot_deterministic_transformations(self, image_pil):

        transforms_to_apply = {
            "Original": lambda x: x,
            "Resize": transforms.Resize(224),
            "CenterCrop": transforms.CenterCrop(224),
            "Grayscale": transforms.Grayscale(num_output_channels=3),
            "ColorJitter": transforms.ColorJitter(
                brightness=0.3, contrast=0.3, saturation=0.1, hue=0.05
            ),
            "High Pass Filter": transforms.Compose([
                # transforms.ColorJitter(
                #     brightness=0.5,
                #     contrast=0.9,
                #     saturation=0.1,
                #     hue=0.05
                # ),
                transforms.Resize(224),
                # transforms.CenterCrop(224),
                # transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                # transforms.Lambda(lambda x: torch.clamp(x, 0, 1))
                transforms.Lambda(lambda x: self._high_pass_filter(x, alpha=5.0)),
                transforms.Lambda(lambda x: self.local_contrast_norm(x)),

                ]
                )
        }

        fig, axes = plt.subplots(1, len(transforms_to_apply), figsize=(15, 4))

        for i, (name, transform) in enumerate(transforms_to_apply.items()):
            img = transform(image_pil)

            if isinstance(img, torch.Tensor):
                img = img.permute(1, 2, 0).numpy()
            else:
                img = np.array(img)

            axes[i].imshow(img)
            axes[i].set_title(name)
            axes[i].axis("off")

        plt.tight_layout()
        plt.show()



if __name__ == "__main__":
    data_dir = "/home/rabah/data/Paysages/seg_train"
    processor = Processor(mode="global", use_patch_shuffle=True)
    # processor.plot_processing_example(original_data_dir=data_dir)
    dataset = ImageFolder(data_dir, transform=None)
    image, _ = dataset[0]

    processor.plot_deterministic_transformations(image)

    # processor.compare_pipelines(image)
    # processor = Processor(mode="texture")
    # texture_dataset, global_dataset = Processor.create_datasets(data_dir=data_dir)
    # processor.plot_processing_example(original_data_dir=data_dir, texture_dataset=texture_dataset, global_dataset=global_dataset)
    # # frequencies = processor.get_datasets_frequencies({"texture": texture_dataset, "global": global_dataset})
    # print(frequencies)