from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v3_small


@dataclass
class CRCDOutput:
    embedding: torch.Tensor
    id_logits: Optional[torch.Tensor]


class CRCDModel(nn.Module):
    """Cross-Resolution Contrastive Distillation model with shared encoder."""

    def __init__(
        self,
        num_classes: int,
        image_size: int = 224,
        embedding_dim: int = 256,
        pretrained: bool = True,
        dropout_p: float = 0.2,
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.image_size = int(image_size)
        self.embedding_dim = int(embedding_dim)

        weights = "DEFAULT" if pretrained else None
        base = mobilenet_v3_small(weights=weights)

        self.backbone = base.features
        self.pool = nn.AdaptiveAvgPool2d(1)

        feat_dim = self._infer_backbone_dim()
        self.embedding_head = nn.Sequential(
            nn.Linear(feat_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
        )
        self.id_classifier = nn.Linear(embedding_dim, num_classes, bias=False)

    def _infer_backbone_dim(self) -> int:
        was_training = self.backbone.training
        self.backbone.eval()
        with torch.no_grad():
            x = torch.zeros(1, 3, self.image_size, self.image_size)
            f = self.backbone(x)
            f = self.pool(f).flatten(1)
            dim = int(f.shape[1])
        self.backbone.train(was_training)
        return dim

    def forward(self, x: torch.Tensor, training_outputs: bool = True) -> CRCDOutput:
        f = self.backbone(x)
        f = self.pool(f).flatten(1)

        embedding = self.embedding_head(f)
        embedding = F.normalize(embedding, p=2, dim=1)

        if not training_outputs:
            return CRCDOutput(embedding=embedding, id_logits=None)

        id_logits = self.id_classifier(embedding)
        return CRCDOutput(embedding=embedding, id_logits=id_logits)


class BatchHardTripletLoss(nn.Module):
    def __init__(self, margin: float = 0.3) -> None:
        super().__init__()
        self.ranking = nn.MarginRankingLoss(margin=margin)

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if embeddings.size(0) < 2:
            return embeddings.new_tensor(0.0)

        sq = embeddings.pow(2).sum(dim=1, keepdim=True)
        dist = (sq + sq.t() - 2.0 * embeddings @ embeddings.t()).clamp_min(1e-12).sqrt()

        labels = labels.view(-1)
        pos = labels.unsqueeze(1).eq(labels.unsqueeze(0))
        neg = ~pos

        eye = torch.eye(pos.size(0), device=pos.device, dtype=torch.bool)
        pos = pos & ~eye

        hardest_pos = torch.where(pos, dist, torch.full_like(dist, -1e6)).max(dim=1).values
        hardest_neg = torch.where(neg, dist, torch.full_like(dist, 1e6)).min(dim=1).values

        valid = (hardest_pos > -1e5) & (hardest_neg < 1e5)
        if valid.sum() == 0:
            return embeddings.new_tensor(0.0)

        target = torch.ones_like(hardest_neg[valid])
        return self.ranking(hardest_neg[valid], hardest_pos[valid], target)


class CrossResolutionContrastiveLoss(nn.Module):
    """Supervised contrastive loss computed only across LR<->HR pairs."""

    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        self.temperature = float(temperature)

    def _directional_loss(self, sim: torch.Tensor, labels_a: torch.Tensor, labels_b: torch.Tensor) -> torch.Tensor:
        positives = labels_a.unsqueeze(1).eq(labels_b.unsqueeze(0))
        if not positives.any():
            return sim.new_tensor(0.0)

        log_probs = sim - torch.logsumexp(sim, dim=1, keepdim=True)

        per_anchor: list[torch.Tensor] = []
        for i in range(sim.size(0)):
            pos_mask = positives[i]
            if pos_mask.any():
                per_anchor.append(-log_probs[i, pos_mask].mean())

        if not per_anchor:
            return sim.new_tensor(0.0)

        return torch.stack(per_anchor).mean()

    def forward(self, z_lr: torch.Tensor, z_hr: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        labels = labels.view(-1).long()
        sim = (z_lr @ z_hr.t()) / max(self.temperature, 1e-6)
        loss_lr_to_hr = self._directional_loss(sim, labels, labels)
        loss_hr_to_lr = self._directional_loss(sim.t(), labels, labels)
        return 0.5 * (loss_lr_to_hr + loss_hr_to_lr)


class CRCDLoss(nn.Module):
    def __init__(
        self,
        temperature: float = 0.07,
        triplet_margin: float = 0.3,
        id_weight: float = 1.0,
        contrastive_weight: float = 1.0,
        triplet_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.id_ce = nn.CrossEntropyLoss()
        self.cross_res_contrastive = CrossResolutionContrastiveLoss(temperature=temperature)
        self.triplet = BatchHardTripletLoss(margin=triplet_margin)

        self.id_weight = float(id_weight)
        self.contrastive_weight = float(contrastive_weight)
        self.triplet_weight = float(triplet_weight)

    def forward(
        self,
        z_lr: torch.Tensor,
        z_hr: torch.Tensor,
        logits_lr: torch.Tensor,
        logits_hr: torch.Tensor,
        labels: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        labels = labels.view(-1).long()

        loss_id = 0.5 * (self.id_ce(logits_lr, labels) + self.id_ce(logits_hr, labels))
        loss_contrastive = self.cross_res_contrastive(z_lr, z_hr, labels)

        z_all = torch.cat([z_lr, z_hr], dim=0)
        y_all = labels.repeat(2)
        loss_triplet = self.triplet(z_all, y_all)

        total = (
            self.id_weight * loss_id
            + self.contrastive_weight * loss_contrastive
            + self.triplet_weight * loss_triplet
        )

        return {
            "loss": total,
            "loss_id": loss_id.detach(),
            "loss_cross_res_contrastive": loss_contrastive.detach(),
            "loss_triplet": loss_triplet.detach(),
        }
