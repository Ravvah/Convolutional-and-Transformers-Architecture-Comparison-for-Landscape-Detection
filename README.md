# Convolutional and Transformers Architecture Comparison for Landscape Detection

## Project goal
This project aims to evaluate and compare convolutional neural networks (CNNs) and transformer-based vision architectures for the task of landscape detection (semantic segmentation or classification of landscape features). The goal is to determine which family of models delivers better accuracy, robustness, and efficiency for typical landscape imagery, and to provide practical guidance on trade-offs (performance vs. compute, generalization, and inference speed).

## Approaches

- Data and preprocessing
  - Use curated landscape image datasets (satellite, drone, or ground photos) split into train/val/test. Apply standard preprocessing: resizing, normalization, and data augmentation (flips, rotations, color jitter).

- Model families compared
  - Convolutional architectures: classical and modern CNN backbones and segmentation heads (e.g., ResNet, EfficientNet, U-Net, DeepLab variants). Evaluate parameter counts, latency, and feature locality strengths.
  - Transformer-based architectures: Vision Transformer (ViT), Swin Transformer, and hybrid conv-transformer models. Evaluate benefits for long-range context, attention-driven feature learning, and scaling behavior.

- Training and evaluation
  - Train under matched conditions (same training schedule, augmentation pipeline, and compute budget) to make comparisons fair.
  - Metrics: classification accuracy or mean Intersection-over-Union (mIoU), F1/precision/recall for segmentation, plus runtime (inference latency) and model size (parameters, FLOPs).
  - Experiments: ablation studies on patch size / receptive field, pretraining (ImageNet / self-supervised), effect of augmentations, and transfer to different landscape domains.
