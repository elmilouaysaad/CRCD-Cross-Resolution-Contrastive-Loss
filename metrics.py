from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader


@torch.no_grad()
def extract_embeddings(model, data_loader: DataLoader, device: torch.device):
    model.eval()
    feats: List[np.ndarray] = []
    pids: List[int] = []
    camids: List[int] = []
    paths: List[str] = []
    pid_strs: List[str] = []

    for images, pid_labels, cids, pths, pids_s in data_loader:
        images = images.to(device, non_blocking=True)
        out = model(images, training_outputs=False)
        feats.append(out.embedding.cpu().numpy())
        pids.extend(pid_labels.numpy().tolist())
        camids.extend(cids.numpy().tolist())
        paths.extend(list(pths))
        pid_strs.extend(list(pids_s))

    if feats:
        feat_arr = np.concatenate(feats, axis=0).astype(np.float32)
    else:
        feat_arr = np.zeros((0, model.embedding_dim), dtype=np.float32)

    return {
        "embeddings": feat_arr,
        "pids": np.asarray(pids, dtype=np.int64),
        "camids": np.asarray(camids, dtype=np.int64),
        "paths": paths,
        "pid_strs": pid_strs,
    }


def evaluate_rank_map(
    qf: np.ndarray,
    q_pids: np.ndarray,
    q_camids: np.ndarray,
    gf: np.ndarray,
    g_pids: np.ndarray,
    g_camids: np.ndarray,
    max_rank: int = 50,
) -> Tuple[Dict[str, float], np.ndarray, List[int]]:
    if qf.size == 0 or gf.size == 0:
        empty = {
            "mAP": 0.0,
            "Rank-1": 0.0,
            "Rank-5": 0.0,
            "Rank-10": 0.0,
            "mean_first_pos_rank": 0.0,
        }
        return empty, np.zeros((max_rank,), dtype=np.float64), []

    qf = qf.astype(np.float32, copy=False)
    gf = gf.astype(np.float32, copy=False)

    cmc = np.zeros((max_rank,), dtype=np.float64)
    ap_list: List[float] = []
    first_pos_ranks: List[int] = []

    valid_q = 0
    gf_sq = np.sum(gf * gf, axis=1)

    for i in range(qf.shape[0]):
        q = qf[i : i + 1]
        dist = np.sum(q * q, axis=1, keepdims=True) + gf_sq[None, :] - 2.0 * (q @ gf.T)
        order = np.argsort(dist[0])

        remove = (g_pids[order] == q_pids[i]) & (g_camids[order] == q_camids[i])
        order = order[~remove]

        matches = (g_pids[order] == q_pids[i]).astype(np.int32)
        if matches.sum() == 0:
            continue

        valid_q += 1
        first = int(np.where(matches == 1)[0][0]) + 1
        first_pos_ranks.append(first)

        cmc[min(first - 1, max_rank - 1) :] += 1

        precisions = np.cumsum(matches) / (np.arange(matches.size) + 1)
        ap = float((precisions * matches).sum() / matches.sum())
        ap_list.append(ap)

    if valid_q == 0:
        empty = {
            "mAP": 0.0,
            "Rank-1": 0.0,
            "Rank-5": 0.0,
            "Rank-10": 0.0,
            "mean_first_pos_rank": 0.0,
        }
        return empty, np.zeros((max_rank,), dtype=np.float64), []

    cmc = cmc / valid_q
    metrics = {
        "mAP": float(np.mean(ap_list)),
        "Rank-1": float(cmc[0]) if max_rank >= 1 else 0.0,
        "Rank-5": float(cmc[4]) if max_rank >= 5 else float(cmc[-1]),
        "Rank-10": float(cmc[9]) if max_rank >= 10 else float(cmc[-1]),
        "mean_first_pos_rank": float(np.mean(first_pos_ranks)) if first_pos_ranks else 0.0,
    }

    return metrics, cmc, first_pos_ranks
