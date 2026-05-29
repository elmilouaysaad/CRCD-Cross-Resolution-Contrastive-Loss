from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


def plot_training_curves(history: List[Dict[str, float]], out_path: Path) -> None:
    if not history:
        return

    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_map = [h["val_mAP"] for h in history]
    val_r1 = [h["val_Rank-1"] for h in history]
    train_contrastive = [h["train_contrastive"] for h in history]

    plt.figure(figsize=(12, 5))

    ax1 = plt.subplot(1, 3, 1)
    ax1.plot(epochs, train_loss, marker="o", linewidth=2)
    ax1.set_title("CRCD Train Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(True, alpha=0.3)

    ax2 = plt.subplot(1, 3, 2)
    ax2.plot(epochs, train_contrastive, marker="o", linewidth=2, color="#d62728")
    ax2.set_title("Cross-Res Contrastive Loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.grid(True, alpha=0.3)

    ax3 = plt.subplot(1, 3, 3)
    ax3.plot(epochs, val_map, marker="o", linewidth=2, label="mAP")
    ax3.plot(epochs, val_r1, marker="o", linewidth=2, label="Rank-1")
    ax3.set_title("CRCD Validation Metrics")
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Score")
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=220)
    plt.close()


def plot_cmc(cmc: np.ndarray, out_path: Path, topk: int = 50) -> None:
    if cmc.size == 0:
        return

    k = min(topk, int(cmc.size))
    ranks = np.arange(1, k + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(ranks, cmc[:k], linewidth=2)
    plt.title("CRCD CMC Curve")
    plt.xlabel("Rank")
    plt.ylabel("Retrieval Rate")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=220)
    plt.close()


def plot_first_positive_rank_hist(first_pos_ranks: List[int], out_path: Path) -> None:
    if not first_pos_ranks:
        return

    plt.figure(figsize=(8, 5))
    plt.hist(first_pos_ranks, bins=40, alpha=0.9, color="#1f77b4")
    plt.title("First Positive Rank Distribution")
    plt.xlabel("First Positive Rank")
    plt.ylabel("Count")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=220)
    plt.close()
