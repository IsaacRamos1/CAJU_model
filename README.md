# Hybrid CNN Training Pipeline with Attention

This repository provides a PyTorch-based training pipeline for multi-class image classification with CAJU model a custom hybrid architecture that combines VGG16 and MobileNetV3 with a lightweight self-attention mechanism. The code supports several standard CNN backbones and early stopping.

---

## 1. Overview

The core components are:

- **`CombinedVGG16MobileNetV3`**: hybrid CNN that fuses VGG16 and MobileNetV3 feature maps, followed by a Light Self-Attention block and an MLP classifier.
- **`CNNTrainer`**: high-level training and evaluation wrapper for multiple backbones.
- **`LightSelfAttention`**: simple self-attention module applied over spatially flattened feature maps.
- Integrated **early stopping** (via a custom `EarlyStopping` class), you can use your own earlystop class.
- Support for multi-class metrics: **accuracy, precision, sensitivity, F1-score**.

The code is intended for scientific workflow in computer vision, especially when comparing different CNN backbones under a common training pipeline.

---

## 2. Implemented Architectures

The trainer can supports the following `model_name` options:

- `vgg16`       (VGG16 used)
- `mobilenetv3` (MobileNetV3-Large used)
- `vgg16_mobilenetv3` (CAJU model with attention block)
  You can load other models for new aproaches.

All architectures are initialized with ImageNet-pretrained weights from `torchvision` the first five epochs.
