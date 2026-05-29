# CRCD: Cross-Resolution Contrastive Dynamics for Vehicle Re-Identification

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A lightweight vehicle re-identification model designed for edge deployment under cross-resolution conditions. CRCD achieves **21.64% mAP** and **28.37% Rank-1** accuracy on VeRi-776 while running at **2.54ms** on CPU with only **1.22M** parameters.

## Problem

In real-world MTMC (Multi-Target Multi-Camera) tracking systems, query images from long-range cameras capture vehicles at **32×32 pixels**, while gallery databases store **224×224** reference signatures. Standard Re-ID models collapse under this resolution mismatch—ResNet-50 achieves only **7.53% mAP** without explicit cross-resolution adaptation.

## Solution

CRCD couples a lightweight MobileNetV3-Small backbone with **supervised contrastive loss** applied across the LR-HR resolution boundary. The contrastive objective directly pulls same-identity cross-resolution embeddings together while pushing apart different identities.

### Key Features

- **Cross-Resolution Contrastive Loss**: Direct gradient pathways between LR and HR embeddings of the same vehicle
- **Multi-Task Objective**: Combines cross-entropy (label smoothing), supervised contrastive (τ=0.07), and batch-hard triplet loss (margin=0.3)
- **Edge-First Design**: 1.22M parameters, 110M FLOPs, 2.54ms CPU inference
- **Reproducible**: Fixed seeds (42) and deterministic evaluation pipeline

## Architecture

```
Input (224×224)
    ↓
MobileNetV3-Small backbone
    ↓
Global Average Pooling (576D)
    ↓
Linear → BatchNorm → ReLU → Dropout(0.2)
    ↓
L2 Normalization → 256D embedding
```

## Results

| Model | mAP | Rank-1 | Params | FLOPs | CPU Latency |
|-------|-----|--------|--------|-------|--------------|
| ResNet-50 (fine-tuned) | 7.53% | 9.54% | 24.69M | 8,261M | 39.6ms |
| OSNet-x1.0 (fine-tuned) | 15.16% | 19.73% | 2.46M | 3,103M | 29.9ms |
| MobileNetV3-L (fine-tuned) | 17.19% | 24.31% | 4.94M | 440M | 7.8ms |
| ViT-B/16 (fine-tuned) | 14.58% | 19.07% | 86.6M | 24,034M | 106.5ms |
| **CRCD (ours)** | **21.64%** | **28.37%** | **1.22M** | **110M** | **2.54ms** |

CRCD improves over fine-tuned MobileNetV3-L by **+4.45 mAP points (+25.9% relative)** while being **4× smaller** and **3.1× faster**.

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/vehicle-reid-crcd
cd vehicle-reid-crcd
pip install torch torchvision numpy tqdm pillow
```

## Dataset

Download VeRi-776 from the [official source](https://github.com/JDAI-CV/VeRidataset) and place it in the expected structure:

```
VeRi_reid/
├── image_train/
├── image_query/
├── image_test/
└── train_test_split/
```

## Usage

### Training

```bash
python train_crcd.py \
    --data-root ./VeRi_reid \
    --epochs 25 \
    --batch-size-train 64 \
    --device cuda \
    --amp
```


## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--embedding-dim` | 256 | Output embedding dimension |
| `--temperature` | 0.07 | Contrastive loss temperature |
| `--triplet-margin` | 0.3 | Batch-hard triplet margin |
| `--id-weight` | 1.0 | Cross-entropy loss weight |
| `--contrastive-weight` | 1.0 | Contrastive loss weight |
| `--triplet-weight` | 1.0 | Triplet loss weight |
| `--lr` | 3e-4 | Learning rate |
| `--early-stop-patience` | 7 | Patience for early stopping |


## Citation

If you use this code in your research, please cite:

```bibtex
@misc{elmilouay2025crcd,
  title={CRCD: Cross-Resolution Contrastive Dynamics for Vehicle Re-Identification},
  author={Elmilouay, Saad},
  year={2025},
  howpublished={GitHub},
}
```

## License

MIT

## Acknowledgments

- VeRi-776 dataset by Liu et al.
- MobileNetV3 by Howard et al.
- Supervised Contrastive Learning by Khosla et al.

