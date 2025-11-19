# Hybrid CNN Training Pipeline with Attention

This repository provides a PyTorch-based training pipeline for multi-class image classification with CAJU model a custom hybrid architecture that combines VGG16 and MobileNetV3 with a lightweight self-attention mechanism. The code supports several standard CNN backbones, early stopping, Weights & Biases logging, and automatic plotting of training curves.

---

## 1. Overview

The core components are:

- **`CombinedVGG16MobileNetV3`**: hybrid CNN that fuses VGG16 and MobileNetV3 feature maps, followed by a Light Self-Attention block and an MLP classifier.
- **`CNNTrainer`**: high-level training and evaluation wrapper for multiple backbones.
- **`LightSelfAttention`**: simple self-attention module applied over spatially flattened feature maps.
- Integrated **early stopping** (via a custom `EarlyStopping` class), you can use your own earlystop class.
- Support for multi-class metrics: **accuracy, precision, recall, F1-score, ROC–AUC (OvR, macro)**.

The code is intended for scientific/experimental workflows in computer vision, especially when comparing different CNN backbones under a common training pipeline.

---

## 2. Implemented Architectures

The trainer supports the following `model_name` options:

- `resnet18`
- `resnet50`
- `resnet101`
- `vgg16`
- `mobilenetv3` (MobileNetV3-Large)
- `densenet121`
- `densenet161`
- `inceptionv3`
- `efficientnet_b0`
- `efficientnet_b3`
- `mobilenetv4_hybrid` (via `timm`)
- `vgg16_mobilenetv3` (custom hybrid with attention)

All architectures are initialized with ImageNet-pretrained weights from `torchvision` or `timm`. After loading, **all convolutional and fully connected layers are reinitialized using Xavier uniform initialization**, via:

```python
def initialize_weights_xavier(module):
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
        init.xavier_uniform_(module.weight)
        if module.bias is not None:
            init.zeros_(module.bias)
