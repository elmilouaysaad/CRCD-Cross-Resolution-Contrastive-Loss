from __future__ import annotations

import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision import transforms


CAM_RE = re.compile(r"_(c\d{3})_", re.IGNORECASE)
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass
class Sample:
    path: Path
    pid_str: str
    pid_label: int
    camid: int


class CRCDTrainDataset(Dataset):
    """Each sample returns an LR view and HR view of the same identity image."""

    def __init__(
        self,
        raw_samples: Sequence[Tuple[Path, str, int]],
        pid_map: Dict[str, int],
        tf_lr: transforms.Compose,
        tf_hr: transforms.Compose,
    ) -> None:
        self.samples: List[Sample] = []
        self.tf_lr = tf_lr
        self.tf_hr = tf_hr

        for path, pid_str, camid in raw_samples:
            if pid_str not in pid_map:
                continue
            self.samples.append(Sample(path=path, pid_str=pid_str, pid_label=pid_map[pid_str], camid=camid))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        with Image.open(s.path) as img:
            img = img.convert("RGB")
        x_lr = self.tf_lr(img)
        x_hr = self.tf_hr(img)
        return x_lr, x_hr, s.pid_label


class CRCDEvalDataset(Dataset):
    def __init__(
        self,
        raw_samples: Sequence[Tuple[Path, str, int]],
        pid_map: Dict[str, int],
        tf: transforms.Compose,
    ) -> None:
        self.samples: List[Sample] = []
        self.tf = tf

        for path, pid_str, camid in raw_samples:
            if pid_str not in pid_map:
                continue
            self.samples.append(Sample(path=path, pid_str=pid_str, pid_label=pid_map[pid_str], camid=camid))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        with Image.open(s.path) as img:
            img = img.convert("RGB")
        x = self.tf(img)
        return x, s.pid_label, s.camid, str(s.path), s.pid_str


class RandomIdentitySampler(Sampler[int]):
    """Sample P identities x K instances so each identity appears every batch."""

    def __init__(self, dataset: CRCDTrainDataset, num_pids_per_batch: int, num_instances: int) -> None:
        self.dataset = dataset
        self.num_pids_per_batch = num_pids_per_batch
        self.num_instances = num_instances

        self.index_dic: Dict[int, List[int]] = {}
        for idx, sample in enumerate(dataset.samples):
            self.index_dic.setdefault(sample.pid_label, []).append(idx)
        self.pids = list(self.index_dic.keys())

        if len(self.pids) < self.num_pids_per_batch:
            raise ValueError(
                f"Not enough identities ({len(self.pids)}) for sampler setting "
                f"num_pids_per_batch={self.num_pids_per_batch}."
            )

        self.length = len(self.pids) * self.num_instances

    def __iter__(self) -> Iterable[int]:
        batch_idxs: List[int] = []
        pid_pool = self.pids.copy()
        random.shuffle(pid_pool)

        while len(pid_pool) >= self.num_pids_per_batch:
            selected = [pid_pool.pop() for _ in range(self.num_pids_per_batch)]
            for pid in selected:
                idxs = self.index_dic[pid]
                if len(idxs) >= self.num_instances:
                    sampled = random.sample(idxs, self.num_instances)
                else:
                    sampled = np.random.choice(idxs, self.num_instances, replace=True).tolist()
                batch_idxs.extend(sampled)

        return iter(batch_idxs)

    def __len__(self) -> int:
        return self.length


def parse_camid_from_name(name: str) -> int:
    m = CAM_RE.search(name)
    if not m:
        return -1
    return int(m.group(1)[1:])


def collect_split_samples(split_dir: Path) -> List[Tuple[Path, str, int]]:
    if not split_dir.exists():
        raise FileNotFoundError(f"Missing split directory: {split_dir}")

    paths = [p for p in split_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS]
    paths.sort()

    out: List[Tuple[Path, str, int]] = []
    for p in paths:
        out.append((p, p.parent.name, parse_camid_from_name(p.name)))
    return out


def build_pid_map(samples: Sequence[Tuple[Path, str, int]]) -> Dict[str, int]:
    pids = sorted({pid for _, pid, _ in samples})
    return {pid: i for i, pid in enumerate(pids)}


def build_transforms(image_size_hr: int = 224, image_size_lr: int = 32):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    tf_train_lr = transforms.Compose(
        [
            transforms.Resize((image_size_lr, image_size_lr)),
            transforms.Resize((image_size_hr, image_size_hr)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            normalize,
        ]
    )
    tf_train_hr = transforms.Compose(
        [
            transforms.Resize((image_size_hr, image_size_hr)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            normalize,
        ]
    )

    tf_eval_query = transforms.Compose(
        [
            transforms.Resize((image_size_lr, image_size_lr)),
            transforms.Resize((image_size_hr, image_size_hr)),
            transforms.ToTensor(),
            normalize,
        ]
    )
    tf_eval_gallery = transforms.Compose(
        [
            transforms.Resize((image_size_hr, image_size_hr)),
            transforms.ToTensor(),
            normalize,
        ]
    )

    return tf_train_lr, tf_train_hr, tf_eval_query, tf_eval_gallery


def build_loaders(
    data_root: Path,
    batch_size_train: int,
    batch_size_eval: int,
    num_workers: int,
    train_pids_per_batch: int,
    train_instances: int,
    image_size_hr: int = 224,
    image_size_lr: int = 32,
):
    train_raw = collect_split_samples(data_root / "train")
    query_raw = collect_split_samples(data_root / "query")
    gallery_raw = collect_split_samples(data_root / "gallery")

    train_pid_map = build_pid_map(train_raw)
    eval_pid_map = build_pid_map(list(query_raw) + list(gallery_raw))

    tf_train_lr, tf_train_hr, tf_eval_query, tf_eval_gallery = build_transforms(
        image_size_hr=image_size_hr,
        image_size_lr=image_size_lr,
    )

    train_ds = CRCDTrainDataset(train_raw, train_pid_map, tf_train_lr, tf_train_hr)
    query_ds = CRCDEvalDataset(query_raw, eval_pid_map, tf_eval_query)
    gallery_ds = CRCDEvalDataset(gallery_raw, eval_pid_map, tf_eval_gallery)

    sampler = RandomIdentitySampler(
        train_ds,
        num_pids_per_batch=train_pids_per_batch,
        num_instances=train_instances,
    )

    expected_bs = train_pids_per_batch * train_instances
    if batch_size_train != expected_bs:
        batch_size_train = expected_bs

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size_train,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )

    query_loader = DataLoader(
        query_ds,
        batch_size=batch_size_eval,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    gallery_loader = DataLoader(
        gallery_ds,
        batch_size=batch_size_eval,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return {
        "train_loader": train_loader,
        "query_loader": query_loader,
        "gallery_loader": gallery_loader,
        "train_pid_map": train_pid_map,
        "eval_pid_map": eval_pid_map,
        "counts": {
            "train": len(train_ds),
            "query": len(query_ds),
            "gallery": len(gallery_ds),
        },
    }
