#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from tqdm.auto import tqdm

from dataloaders import build_loaders
from metrics import evaluate_rank_map, extract_embeddings
from model import CRCDLoss, CRCDModel
from visualize import plot_cmc, plot_first_positive_rank_hist, plot_training_curves


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train and benchmark CRCD end-to-end")
    p.add_argument("--data-root", type=Path, default=Path("VeRi_reid"))
    p.add_argument("--output-dir", type=Path, default=Path("models") / "crcd" / "outputs")
    p.add_argument("--checkpoint-dir", type=Path, default=Path("models") / "crcd" / "checkpoints")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch-size-train", type=int, default=64)
    p.add_argument("--batch-size-eval", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=4)

    p.add_argument("--train-pids-per-batch", type=int, default=16)
    p.add_argument("--train-instances", type=int, default=4)
    p.add_argument("--image-size-hr", type=int, default=224)
    p.add_argument("--image-size-lr", type=int, default=32)

    p.add_argument("--embedding-dim", type=int, default=256)
    p.add_argument("--dropout-p", type=float, default=0.2)
    p.add_argument("--temperature", type=float, default=0.07)

    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--triplet-margin", type=float, default=0.3)

    p.add_argument("--id-weight", "--cross-entropy-weight", dest="id_weight", type=float, default=1.0)
    p.add_argument("--contrastive-weight", type=float, default=1.0)
    p.add_argument("--triplet-weight", type=float, default=1.0)

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--early-stop-patience", type=int, default=7)
    p.add_argument("--early-stop-min-delta", type=float, default=1e-4)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--no-progress", action="store_true")
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_jsonable(v: Any) -> Any:
    if isinstance(v, Path):
        return str(v)
    if isinstance(v, dict):
        return {str(k): to_jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [to_jsonable(x) for x in v]
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    return v


def save_checkpoint(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def resolve_repo_path(path_value: Path, repo_root: Path) -> Path:
    if path_value.is_absolute():
        return path_value
    if path_value.exists():
        return path_value
    return repo_root / path_value


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    repo_root = Path(__file__).resolve().parents[2]
    args.data_root = resolve_repo_path(args.data_root, repo_root)
    args.output_dir = resolve_repo_path(args.output_dir, repo_root)
    args.checkpoint_dir = resolve_repo_path(args.checkpoint_dir, repo_root)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")

    loaders = build_loaders(
        data_root=args.data_root,
        batch_size_train=args.batch_size_train,
        batch_size_eval=args.batch_size_eval,
        num_workers=args.num_workers,
        train_pids_per_batch=args.train_pids_per_batch,
        train_instances=args.train_instances,
        image_size_hr=args.image_size_hr,
        image_size_lr=args.image_size_lr,
    )

    train_loader = loaders["train_loader"]
    query_loader = loaders["query_loader"]
    gallery_loader = loaders["gallery_loader"]
    train_pid_map = loaders["train_pid_map"]

    model = CRCDModel(
        num_classes=len(train_pid_map),
        image_size=args.image_size_hr,
        embedding_dim=args.embedding_dim,
        pretrained=True,
        dropout_p=args.dropout_p,
    ).to(device)

    criterion = CRCDLoss(
        temperature=args.temperature,
        triplet_margin=args.triplet_margin,
        id_weight=args.id_weight,
        contrastive_weight=args.contrastive_weight,
        triplet_weight=args.triplet_weight,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=bool(args.amp and device.type == "cuda"))

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    latest_ckpt = args.checkpoint_dir / "crcd_latest.pt"
    best_ckpt = args.checkpoint_dir / "crcd_best.pt"

    start_epoch = 1
    best_map = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    stopped_early = False
    stop_epoch = args.epochs
    history: List[Dict[str, float]] = []

    if args.resume and latest_ckpt.exists():
        ckpt = torch.load(latest_ckpt, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        if "scaler_state" in ckpt and args.amp and device.type == "cuda":
            scaler.load_state_dict(ckpt["scaler_state"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_map = float(ckpt.get("best_map", -1.0))
        best_epoch = int(ckpt.get("best_epoch", 0))
        epochs_without_improvement = int(ckpt.get("epochs_without_improvement", 0))
        history = list(ckpt.get("history", []))

    show_progress = not args.no_progress

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        running_loss = 0.0
        running_id = 0.0
        running_ctr = 0.0
        running_tri = 0.0
        running_steps = 0

        iterator = train_loader if not show_progress else tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}", leave=False)
        for images_lr, images_hr, pid_labels in iterator:
            images_lr = images_lr.to(device, non_blocking=True)
            images_hr = images_hr.to(device, non_blocking=True)
            pid_labels = pid_labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=bool(args.amp and device.type == "cuda")):
                x = torch.cat([images_lr, images_hr], dim=0)
                out = model(x, training_outputs=True)
                z_lr, z_hr = torch.split(out.embedding, [images_lr.size(0), images_hr.size(0)], dim=0)
                logits_lr, logits_hr = torch.split(out.id_logits, [images_lr.size(0), images_hr.size(0)], dim=0)

                losses = criterion(
                    z_lr=z_lr,
                    z_hr=z_hr,
                    logits_lr=logits_lr,
                    logits_hr=logits_hr,
                    labels=pid_labels,
                )
                loss = losses["loss"]

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += float(loss.item())
            running_id += float(losses["loss_id"].item())
            running_ctr += float(losses["loss_cross_res_contrastive"].item())
            running_tri += float(losses["loss_triplet"].item())
            running_steps += 1

        train_loss = running_loss / max(1, running_steps)
        train_id = running_id / max(1, running_steps)
        train_ctr = running_ctr / max(1, running_steps)
        train_tri = running_tri / max(1, running_steps)

        model.eval()
        q = extract_embeddings(model, query_loader, device)
        g = extract_embeddings(model, gallery_loader, device)
        metrics, cmc, first_pos = evaluate_rank_map(
            qf=q["embeddings"],
            q_pids=q["pids"],
            q_camids=q["camids"],
            gf=g["embeddings"],
            g_pids=g["pids"],
            g_camids=g["camids"],
            max_rank=50,
        )

        row = {
            "epoch": float(epoch),
            "train_loss": float(train_loss),
            "train_id": float(train_id),
            "train_contrastive": float(train_ctr),
            "train_triplet": float(train_tri),
            "val_mAP": float(metrics["mAP"]),
            "val_Rank-1": float(metrics["Rank-1"]),
            "val_Rank-5": float(metrics["Rank-5"]),
            "val_Rank-10": float(metrics["Rank-10"]),
            "val_mean_first_pos_rank": float(metrics["mean_first_pos_rank"]),
        }
        history.append(row)

        payload = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scaler_state": scaler.state_dict() if args.amp and device.type == "cuda" else None,
            "best_map": best_map,
            "best_epoch": best_epoch,
            "epochs_without_improvement": epochs_without_improvement,
            "history": history,
            "args": to_jsonable(vars(args)),
        }

        improved = metrics["mAP"] > (best_map + args.early_stop_min_delta)
        if improved:
            best_map = float(metrics["mAP"])
            best_epoch = epoch
            epochs_without_improvement = 0
            payload["best_map"] = best_map
            payload["best_epoch"] = best_epoch
            payload["epochs_without_improvement"] = epochs_without_improvement
            save_checkpoint(best_ckpt, payload)
        else:
            epochs_without_improvement += 1
            payload["epochs_without_improvement"] = epochs_without_improvement

        save_checkpoint(latest_ckpt, payload)

        if show_progress:
            print(
                f"[epoch {epoch:02d}] loss={train_loss:.4f} ctr={train_ctr:.4f} "
                f"mAP={metrics['mAP']:.4f} R1={metrics['Rank-1']:.4f}"
            )

        if args.early_stop_patience > 0 and epochs_without_improvement >= args.early_stop_patience:
            stopped_early = True
            stop_epoch = epoch
            if show_progress:
                print(
                    f"Early stopping at epoch {epoch}: "
                    f"no mAP improvement greater than {args.early_stop_min_delta} "
                    f"for {args.early_stop_patience} epoch(s)."
                )
            break

    if best_ckpt.exists():
        ckpt = torch.load(best_ckpt, map_location=device)
        model.load_state_dict(ckpt["model_state"])

    model.eval()
    q = extract_embeddings(model, query_loader, device)
    g = extract_embeddings(model, gallery_loader, device)
    metrics, cmc, first_pos = evaluate_rank_map(
        qf=q["embeddings"],
        q_pids=q["pids"],
        q_camids=q["camids"],
        gf=g["embeddings"],
        g_pids=g["pids"],
        g_camids=g["camids"],
        max_rank=50,
    )

    query_npz = args.output_dir / "crcd_query_embeddings.npz"
    gallery_npz = args.output_dir / "crcd_gallery_embeddings.npz"
    np.savez_compressed(query_npz, **q)
    np.savez_compressed(gallery_npz, **g)

    train_plot = args.output_dir / "crcd_training_curves.png"
    cmc_plot = args.output_dir / "crcd_cmc_curve.png"
    rank_plot = args.output_dir / "crcd_first_positive_rank_hist.png"

    plot_training_curves(history, train_plot)
    plot_cmc(cmc, cmc_plot)
    plot_first_positive_rank_hist(first_pos, rank_plot)

    summary = {
        "meta": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "script": "train_crcd.py",
            "model": "CrossResolutionContrastiveDistillation",
            "device": str(device),
            "data_root": str(args.data_root),
            "settings": to_jsonable(vars(args)),
        },
        "dataset": {
            "train": int(loaders["counts"]["train"]),
            "query": int(loaders["counts"]["query"]),
            "gallery": int(loaders["counts"]["gallery"]),
        },
        "metrics": {
            "mAP": float(metrics["mAP"]),
            "Rank-1": float(metrics["Rank-1"]),
            "Rank-5": float(metrics["Rank-5"]),
            "Rank-10": float(metrics["Rank-10"]),
            "mean_first_pos_rank": float(metrics["mean_first_pos_rank"]),
        },
        "best_monitor": float(best_map),
        "best_epoch": int(best_epoch),
        "early_stopping": {
            "enabled": bool(args.early_stop_patience > 0),
            "patience": int(args.early_stop_patience),
            "min_delta": float(args.early_stop_min_delta),
            "stopped_early": bool(stopped_early),
            "stop_epoch": int(stop_epoch),
            "epochs_without_improvement": int(epochs_without_improvement),
        },
        "history": history,
        "artifacts": {
            "best_checkpoint": str(best_ckpt),
            "latest_checkpoint": str(latest_ckpt),
            "query_embeddings": str(query_npz),
            "gallery_embeddings": str(gallery_npz),
            "training_curves": str(train_plot),
            "cmc_curve": str(cmc_plot),
            "rank_histogram": str(rank_plot),
        },
    }

    summary_path = args.output_dir / "crcd_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved CRCD summary: {summary_path}")
    print(
        f"mAP={metrics['mAP']:.4f}, Rank-1={metrics['Rank-1']:.4f}, "
        f"Rank-5={metrics['Rank-5']:.4f}, Rank-10={metrics['Rank-10']:.4f}"
    )


if __name__ == "__main__":
    main()
