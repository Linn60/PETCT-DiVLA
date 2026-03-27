import os
import sys
import json
import csv
import gc
import random
import argparse
import math
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve,
)

from src.model.CLIP import PETCTCLIPConfig, PETCTCLIP
from src.dataloader.dataloader import PETCT_REGIONS, PETCT_REGION_TO_IDX


# ---------------------------------------------------------------------------
# Label loading (CT abnormal binary: 1=abnormal, 0=normal, flipped from CT_1)
# ---------------------------------------------------------------------------

def _load_labels_with_ct1(labels_path):
    df = pd.read_excel(labels_path)
    num_regions = len(PETCT_REGIONS)
    labels_dict = {}
    for exam_id, group in df.groupby('检查号'):
        exam_id = str(exam_id).strip()
        pet_labels = np.full(num_regions, -1, dtype=np.int64)
        ct_labels = np.zeros(num_regions, dtype=np.float32)
        for _, row in group.iterrows():
            region = str(row['解剖区域']).strip()
            if region not in PETCT_REGION_TO_IDX:
                continue
            ridx = PETCT_REGION_TO_IDX[region]
            pet_val = row['PET摄取分级']
            if pd.notna(pet_val):
                try:
                    v = int(float(pet_val))
                    if 1 <= v <= 5:
                        pet_labels[ridx] = v - 1
                except (ValueError, OverflowError):
                    pass
            ct_labels[ridx] = 1.0 - float(row['CT_1'])
        labels_dict[exam_id] = (pet_labels, ct_labels)
    return labels_dict


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ClassificationDataset(Dataset):
    def __init__(self, image_root, labels_path, organ_labels_path,
                 split="train", test_size=800):
        self.image_root = image_root
        self.labels_dict = _load_labels_with_ct1(labels_path)

        organ_label_ids = set()
        with open(organ_labels_path, encoding="utf-8") as f:
            for line in f:
                organ_label_ids.add(json.loads(line)["report_id"])

        all_folders = sorted(os.listdir(image_root))
        valid_keys = [f for f in all_folders
                      if f in self.labels_dict and f in organ_label_ids]

        rng = random.Random(42)
        rng.shuffle(valid_keys)
        if split == "train":
            self.keys = valid_keys[:-test_size]
        else:
            self.keys = valid_keys[-test_size:]
        print(f"ClassificationDataset split={split}: {len(self.keys)} samples")

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        exam_id = self.keys[idx]
        folder = os.path.join(self.image_root, exam_id)
        pet = np.load(os.path.join(folder, "PET.npy")).astype(np.float32)
        ct = np.load(os.path.join(folder, "CT.npy")).astype(np.float32)
        pet_labels, ct_labels = self.labels_dict[exam_id]
        return {
            "pet": torch.from_numpy(pet[np.newaxis]),
            "ct": torch.from_numpy(ct[np.newaxis]),
            "pet_labels": torch.from_numpy(pet_labels),
            "ct_labels": torch.from_numpy(ct_labels),
            "index": idx,
        }


def collate_fn(batch):
    return {
        "pet": torch.stack([b["pet"] for b in batch]),
        "ct": torch.stack([b["ct"] for b in batch]),
        "pet_labels": torch.stack([b["pet_labels"] for b in batch]),
        "ct_labels": torch.stack([b["ct_labels"] for b in batch]),
        "index": torch.tensor([b["index"] for b in batch], dtype=torch.long),
    }


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class PETCTClassifier(nn.Module):
    def __init__(self, pretrained_path, num_regions=28,
                 num_pet_classes=5):
        super().__init__()

        print(f"Loading pretrained PETCTCLIP from {pretrained_path} ...")
        petctclip = PETCTCLIP.from_pretrained(pretrained_path)
        dim = petctclip.config.dim

        self.pet_encoder = petctclip.pet_encoder
        self.ct_encoder = petctclip.ct_encoder
        self.pet_vision_proj = petctclip.pet_vision_proj
        self.ct_vision_proj = petctclip.ct_vision_proj
        self.fusion_proj = petctclip.fusion_proj

        del petctclip
        gc.collect()

        for module in (self.pet_encoder, self.ct_encoder,
                       self.pet_vision_proj, self.ct_vision_proj,
                       self.fusion_proj):
            for p in module.parameters():
                p.requires_grad = False

        self.num_regions = num_regions
        self.num_pet_classes = num_pet_classes

        self.pet_head = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim, num_regions * num_pet_classes),
        )
        self.ct_head = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim, num_regions),
        )

    def _get_fusion_features(self, pet, ct):
        with torch.no_grad():
            F_pet = self.pet_encoder(pet)
            F_ct = self.ct_encoder(ct)
            v_pet = F.normalize(self.pet_vision_proj(F_pet.mean(dim=1)), dim=-1)
            v_ct = F.normalize(self.ct_vision_proj(F_ct.mean(dim=1)), dim=-1)
            v_fus = self.fusion_proj(torch.cat([v_pet, v_ct], dim=-1))
        return v_fus

    def forward(self, pet, ct, pet_labels=None, ct_labels=None, **kwargs):
        v_fus = self._get_fusion_features(pet, ct)
        B = v_fus.shape[0]

        pet_logits = self.pet_head(v_fus).view(B, self.num_regions, self.num_pet_classes)
        ct_logits = self.ct_head(v_fus).view(B, self.num_regions)

        loss = None
        if pet_labels is not None and ct_labels is not None:
            pet_loss = F.cross_entropy(
                pet_logits.reshape(-1, self.num_pet_classes),
                pet_labels.reshape(-1).long(),
                ignore_index=-1,
            )
            ct_loss = F.binary_cross_entropy_with_logits(
                ct_logits, ct_labels.float(),
            )
            loss = pet_loss + ct_loss

        return {"loss": loss, "pet_logits": pet_logits, "ct_logits": ct_logits}


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def evaluate_pet_grading(all_preds, all_labels, all_probs=None):
    num_classes = all_probs.shape[-1] if all_probs is not None else 0
    results = {}
    for ridx, region in enumerate(PETCT_REGIONS):
        mask = all_labels[:, ridx] >= 0
        if mask.sum() == 0:
            continue
        y_true = all_labels[mask, ridx].astype(int)
        y_pred = all_preds[mask, ridx]
        region_result = {
            "accuracy": accuracy_score(y_true, y_pred),
            "macro_prec": precision_score(y_true, y_pred, average="macro", zero_division=0),
            "macro_rec": recall_score(y_true, y_pred, average="macro", zero_division=0),
            "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
            "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
            "total_samples": int(mask.sum()),
        }
        if all_probs is not None:
            y_prob = all_probs[mask.nonzero()[0], ridx, :]
            present_classes = np.unique(y_true)
            if len(present_classes) >= 2:
                try:
                    region_result["auc"] = roc_auc_score(
                        y_true, y_prob, multi_class="ovr", average="macro",
                    )
                except ValueError:
                    region_result["auc"] = float("nan")
            else:
                region_result["auc"] = float("nan")
        else:
            region_result["auc"] = float("nan")
        results[region] = region_result

    macro_acc = np.mean([r["accuracy"] for r in results.values()])
    macro_prec = np.mean([r["macro_prec"] for r in results.values()])
    macro_rec = np.mean([r["macro_rec"] for r in results.values()])
    macro_f1 = np.mean([r["macro_f1"] for r in results.values()])
    macro_wf1 = np.mean([r["weighted_f1"] for r in results.values()])
    aucs = np.array([r["auc"] for r in results.values()])
    auc_mask = ~np.isnan(aucs)
    macro_auc = float(aucs[auc_mask].mean()) if auc_mask.any() else float("nan")

    print("\n" + "=" * 110)
    print("PET摄取分级 Results (5-class):")
    print("=" * 110)
    header = (f"{'Region':<25} {'Acc':>8} {'MacPrec':>8} {'MacRec':>8} "
              f"{'MacF1':>8} {'W-F1':>8} {'AUC':>8} {'Samples':>8}")
    print(header)
    print("-" * 110)
    for region, m in results.items():
        auc_str = f"{m['auc']:>8.4f}" if not np.isnan(m["auc"]) else f"{'N/A':>8}"
        print(f"{region:<25} {m['accuracy']:>8.4f} {m['macro_prec']:>8.4f} "
              f"{m['macro_rec']:>8.4f} {m['macro_f1']:>8.4f} "
              f"{m['weighted_f1']:>8.4f} {auc_str} {m['total_samples']:>8}")
    print("=" * 110)
    macro_auc_str = f"{macro_auc:>8.4f}" if not np.isnan(macro_auc) else f"{'N/A':>8}"
    print(f"{'Macro Average':<25} {macro_acc:>8.4f} {macro_prec:>8.4f} "
          f"{macro_rec:>8.4f} {macro_f1:>8.4f} {macro_wf1:>8.4f} {macro_auc_str}")
    print("=" * 110)

    results["__macro__"] = {
        "macro_acc": macro_acc, "macro_prec": macro_prec,
        "macro_rec": macro_rec, "macro_f1": macro_f1,
        "macro_wf1": macro_wf1, "macro_auc": macro_auc,
    }
    return results


def evaluate_ct_classification(all_probs, all_labels):
    """CT abnormal binary classification per region (1=abnormal, 0=normal)."""
    all_preds = np.zeros_like(all_probs)
    results = {}

    for ridx, region in enumerate(PETCT_REGIONS):
        y_true = all_labels[:, ridx]
        y_prob = all_probs[:, ridx]

        if y_true.sum() > 0 and y_true.sum() < len(y_true):
            fpr, tpr, thresholds = roc_curve(y_true, y_prob)
            best_idx = (fpr ** 2 + (1 - tpr) ** 2).argmin()
            best_threshold = float(thresholds[best_idx])
            auc = roc_auc_score(y_true, y_prob)
        else:
            best_threshold = 0.5
            auc = float("nan")

        y_pred = (y_prob >= best_threshold).astype(float)
        all_preds[:, ridx] = y_pred
        results[region] = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "auc": auc,
            "threshold": best_threshold,
            "abnormal_samples": int(y_true.sum()),
            "total_samples": len(y_true),
        }

    valid_results = {k: v for k, v in results.items() if not k.startswith("__")}
    accs = np.array([r["accuracy"] for r in valid_results.values()])
    precs = np.array([r["precision"] for r in valid_results.values()])
    recs = np.array([r["recall"] for r in valid_results.values()])
    f1s = np.array([r["f1"] for r in valid_results.values()])
    aucs = np.array([r["auc"] for r in valid_results.values()])

    macro_acc = accs.mean()
    macro_prec = precs.mean()
    macro_rec = recs.mean()
    macro_f1 = f1s.mean()
    auc_mask = ~np.isnan(aucs)
    macro_auc = float(aucs[auc_mask].mean()) if auc_mask.any() else float("nan")

    flat_labels = all_labels.reshape(-1)
    flat_preds = all_preds.reshape(-1)
    micro_prec = precision_score(flat_labels, flat_preds, zero_division=0)
    micro_rec = recall_score(flat_labels, flat_preds, zero_division=0)
    micro_f1 = f1_score(flat_labels, flat_preds, zero_division=0)

    print("\n" + "=" * 100)
    print("CT异常二分类 Results (1=异常, 0=正常):")
    print("=" * 100)
    print(f"{'Region':<30} {'Acc':>7} {'Prec':>7} {'Recall':>7} "
          f"{'F1':>7} {'AUC':>7} {'Abnorm/Tot':>12}")
    print("-" * 100)
    for region, m in valid_results.items():
        auc_str = f"{m['auc']:>7.4f}" if not np.isnan(m["auc"]) else f"{'N/A':>7}"
        print(f"{region:<30} {m['accuracy']:>7.4f} {m['precision']:>7.4f} "
              f"{m['recall']:>7.4f} {m['f1']:>7.4f} {auc_str} "
              f"{m['abnormal_samples']:>5}/{m['total_samples']:<5}")
    print("=" * 100)
    macro_auc_str = f"{macro_auc:>7.4f}" if not np.isnan(macro_auc) else f"{'N/A':>7}"
    print(f"{'Macro Average':<30} {macro_acc:>7.4f} "
          f"{macro_prec:>7.4f} {macro_rec:>7.4f} "
          f"{macro_f1:>7.4f} {macro_auc_str}")
    print(f"{'Micro Average':<30} {'-':>7} {micro_prec:>7.4f} {micro_rec:>7.4f} {micro_f1:>7.4f}")
    print("=" * 100)

    results["__macro__"] = {
        "macro_acc": macro_acc, "macro_prec": macro_prec,
        "macro_rec": macro_rec, "macro_f1": macro_f1,
        "macro_auc": macro_auc,
        "micro_prec": micro_prec, "micro_rec": micro_rec, "micro_f1": micro_f1,
    }
    return results, all_preds


# ---------------------------------------------------------------------------
# Distributed eval: gather predictions across ranks, deduplicate
# ---------------------------------------------------------------------------

def _gather_cat(tensor, world_size):
    gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered, tensor.contiguous())
    return torch.cat(gathered, dim=0)


@torch.no_grad()
def evaluate(model, dataset, batch_size, device, distributed, world_size, rank):
    model.eval()

    sampler = DistributedSampler(dataset, shuffle=False) if distributed else None
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        sampler=sampler, num_workers=6, pin_memory=True, collate_fn=collate_fn,
    )

    all_pet_probs, all_pet_labels = [], []
    all_ct_probs, all_ct_labels = [], []
    all_indices = []

    for batch in tqdm(loader, desc="Evaluating", disable=(rank != 0)):
        pet = batch["pet"].to(device)
        ct = batch["ct"].to(device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            outputs = model(pet, ct)
        all_pet_probs.append(
            torch.softmax(outputs["pet_logits"].float(), dim=-1).cpu())
        all_pet_labels.append(batch["pet_labels"])
        all_ct_probs.append(torch.sigmoid(outputs["ct_logits"]).float().cpu())
        all_ct_labels.append(batch["ct_labels"])
        all_indices.append(batch["index"])

    all_pet_probs = torch.cat(all_pet_probs)
    all_pet_labels = torch.cat(all_pet_labels)
    all_ct_probs = torch.cat(all_ct_probs)
    all_ct_labels = torch.cat(all_ct_labels)
    all_indices = torch.cat(all_indices)

    if distributed:
        all_pet_probs = _gather_cat(all_pet_probs.to(device), world_size).cpu()
        all_pet_labels = _gather_cat(all_pet_labels.to(device), world_size).cpu()
        all_ct_probs = _gather_cat(all_ct_probs.to(device), world_size).cpu()
        all_ct_labels = _gather_cat(all_ct_labels.to(device), world_size).cpu()
        all_indices = _gather_cat(all_indices.to(device), world_size).cpu()

        order = all_indices.argsort()
        all_pet_probs = all_pet_probs[order][:len(dataset)]
        all_pet_labels = all_pet_labels[order][:len(dataset)]
        all_ct_probs = all_ct_probs[order][:len(dataset)]
        all_ct_labels = all_ct_labels[order][:len(dataset)]

    all_pet_probs = all_pet_probs.numpy()
    all_pet_labels = all_pet_labels.numpy().astype(float)
    all_ct_probs = all_ct_probs.numpy()
    all_ct_labels = all_ct_labels.numpy()

    pet_results, ct_results = None, None
    if rank == 0:
        all_pet_preds = all_pet_probs.argmax(axis=-1).astype(float)
        pet_results = evaluate_pet_grading(
            all_pet_preds, all_pet_labels, all_pet_probs)
        ct_results, _ = evaluate_ct_classification(all_ct_probs, all_ct_labels)
    return pet_results, ct_results


def save_pet_results_csv(pet_results, output_path):
    with open(output_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Region", "Accuracy", "Macro_Precision", "Macro_Recall",
                     "Macro_F1", "Weighted_F1", "AUC", "Samples"])
        for region, m in pet_results.items():
            if region == "__macro__":
                continue
            auc_str = f"{m['auc']:.4f}" if not np.isnan(m["auc"]) else "N/A"
            w.writerow([region, f"{m['accuracy']:.4f}", f"{m['macro_prec']:.4f}",
                        f"{m['macro_rec']:.4f}", f"{m['macro_f1']:.4f}",
                        f"{m['weighted_f1']:.4f}", auc_str, m["total_samples"]])
        macro = pet_results["__macro__"]
        macro_auc_str = (f"{macro['macro_auc']:.4f}"
                         if not np.isnan(macro["macro_auc"]) else "N/A")
        w.writerow([])
        w.writerow(["Macro_Accuracy", f"{macro['macro_acc']:.4f}"])
        w.writerow(["Macro_Precision", f"{macro['macro_prec']:.4f}"])
        w.writerow(["Macro_Recall", f"{macro['macro_rec']:.4f}"])
        w.writerow(["Macro_F1", f"{macro['macro_f1']:.4f}"])
        w.writerow(["Macro_Weighted_F1", f"{macro['macro_wf1']:.4f}"])
        w.writerow(["Macro_AUC", macro_auc_str])
    print(f"PET results saved to {output_path}")


def save_ct_results_csv(ct_results, output_path):
    with open(output_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Region", "Accuracy", "Precision", "Recall",
                     "F1", "AUC", "Threshold", "Abnormal_Samples", "Total_Samples"])
        for key, m in ct_results.items():
            if key == "__macro__":
                continue
            auc_str = f"{m['auc']:.4f}" if not np.isnan(m["auc"]) else "N/A"
            w.writerow([key, f"{m['accuracy']:.4f}", f"{m['precision']:.4f}",
                        f"{m['recall']:.4f}", f"{m['f1']:.4f}", auc_str,
                        f"{m['threshold']:.4f}", m["abnormal_samples"],
                        m["total_samples"]])
        macro = ct_results["__macro__"]
        w.writerow([])
        w.writerow(["Macro_Accuracy", f"{macro['macro_acc']:.4f}"])
        w.writerow(["Macro_Precision", f"{macro['macro_prec']:.4f}"])
        w.writerow(["Macro_Recall", f"{macro['macro_rec']:.4f}"])
        w.writerow(["Macro_F1", f"{macro['macro_f1']:.4f}"])
        w.writerow(["Macro_AUC", f"{macro['macro_auc']:.4f}"])
        w.writerow(["Micro_Precision", f"{macro['micro_prec']:.4f}"])
        w.writerow(["Micro_Recall", f"{macro['micro_rec']:.4f}"])
        w.writerow(["Micro_F1", f"{macro['micro_f1']:.4f}"])
    print(f"CT results saved to {output_path}")


def _append_to_summary_csv(csv_path: str, row: dict) -> None:
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    if os.path.exists(csv_path):
        existing = pd.read_csv(csv_path)
        for col in row:
            if col not in existing.columns:
                existing[col] = ""
        new_row = pd.DataFrame([{col: row.get(col, "") for col in existing.columns}])
        updated = pd.concat([existing, new_row], ignore_index=True)
    else:
        updated = pd.DataFrame([row])
    updated.to_csv(csv_path, index=False)
    print(f"Summary row appended to {csv_path}")


def append_ct_summary(ct_results: dict, args, summary_dir: str) -> None:
    macro = ct_results["__macro__"]
    row = {
        "timestamp":       datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "pretrained_path": args.pretrained_path,
        "output_dir":      args.output_dir,
        "max_steps":       args.max_steps,
        "lr":              args.lr,
        "batch_size":      args.batch_size,
        "weight_decay":    args.weight_decay,
        "warmup_steps":    args.warmup_steps,
        "Macro_Accuracy":  round(macro["macro_acc"],  4),
        "Macro_Precision": round(macro["macro_prec"], 4),
        "Macro_Recall":    round(macro["macro_rec"],  4),
        "Macro_F1":        round(macro["macro_f1"],   4),
        "Macro_AUC":       round(macro["macro_auc"],  4) if not np.isnan(macro["macro_auc"]) else "N/A",
        "Micro_Precision": round(macro["micro_prec"], 4),
        "Micro_Recall":    round(macro["micro_rec"],  4),
        "Micro_F1":        round(macro["micro_f1"],   4),
    }
    _append_to_summary_csv(os.path.join(summary_dir, "ct.csv"), row)


def append_pet_summary(pet_results: dict, args, summary_dir: str) -> None:
    macro = pet_results["__macro__"]
    row = {
        "timestamp":        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "pretrained_path":  args.pretrained_path,
        "output_dir":       args.output_dir,
        "max_steps":        args.max_steps,
        "lr":               args.lr,
        "batch_size":       args.batch_size,
        "weight_decay":     args.weight_decay,
        "warmup_steps":     args.warmup_steps,
        "Macro_Accuracy":   round(macro["macro_acc"],  4),
        "Macro_Precision":  round(macro["macro_prec"], 4),
        "Macro_Recall":     round(macro["macro_rec"],  4),
        "Macro_F1":         round(macro["macro_f1"],   4),
        "Macro_Weighted_F1":round(macro["macro_wf1"],  4),
        "Macro_AUC":        round(macro["macro_auc"],  4) if not np.isnan(macro["macro_auc"]) else "N/A",
    }
    _append_to_summary_csv(os.path.join(summary_dir, "pet.csv"), row)


# ---------------------------------------------------------------------------
# LR scheduler
# ---------------------------------------------------------------------------

def get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args):
    distributed = int(os.environ.get("WORLD_SIZE", 1)) > 1
    if distributed:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        rank = 0
        world_size = 1

    is_main = rank == 0

    train_dataset = ClassificationDataset(
        image_root=args.image_root,
        labels_path=args.labels_path,
        organ_labels_path=args.organ_labels_path,
        split="train", test_size=args.test_size,
    )
    test_dataset = ClassificationDataset(
        image_root=args.image_root,
        labels_path=args.labels_path,
        organ_labels_path=args.organ_labels_path,
        split="test", test_size=args.test_size,
    )

    train_sampler = DistributedSampler(train_dataset, shuffle=True) if distributed else None
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=6, pin_memory=True, collate_fn=collate_fn, drop_last=True,
    )

    model = PETCTClassifier(pretrained_path=args.pretrained_path).to(device)
    if distributed:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)
    raw_model = model.module if distributed else model

    if is_main:
        trainable = sum(p.numel() for p in raw_model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in raw_model.parameters())
        print(f"Trainable parameters: {trainable:,} / {total:,}")

    optimizer = torch.optim.AdamW(
        [p for p in raw_model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay,
    )
    warmup_steps = (int(args.warmup_steps * args.max_steps)
                    if args.warmup_steps < 1 else int(args.warmup_steps))
    if is_main:
        print(f"Warmup steps: {warmup_steps} / {args.max_steps}")
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, warmup_steps, args.max_steps,
    )

    model.train()
    step = 0
    epoch = 0
    running_loss = 0.0
    pbar = tqdm(total=args.max_steps, desc="Training", disable=not is_main)

    while step < args.max_steps:
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        epoch += 1
        for batch in train_loader:
            if step >= args.max_steps:
                break

            pet = batch["pet"].to(device)
            ct = batch["ct"].to(device)
            pet_labels = batch["pet_labels"].to(device)
            ct_labels = batch["ct_labels"].to(device)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                outputs = model(pet, ct, pet_labels, ct_labels)
                loss = outputs["loss"]

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in raw_model.parameters() if p.requires_grad], 1.0,
            )
            optimizer.step()
            scheduler.step()

            step += 1
            running_loss += loss.item()
            pbar.update(1)

            if step % args.logging_steps == 0:
                avg = running_loss / args.logging_steps
                lr = scheduler.get_last_lr()[0]
                pbar.set_postfix(loss=f"{avg:.4f}", lr=f"{lr:.2e}")
                running_loss = 0.0

    pbar.close()

    if is_main:
        os.makedirs(args.output_dir, exist_ok=True)
        save_path = os.path.join(args.output_dir, "classifier_head.pt")
        torch.save({
            "pet_head": raw_model.pet_head.state_dict(),
            "ct_head": raw_model.ct_head.state_dict(),
            "pretrained_path": args.pretrained_path,
        }, save_path)
        print(f"\nClassification heads saved to {save_path}")

    if distributed:
        dist.barrier()

    print("\n" + "=" * 60) if is_main else None
    print("  Evaluation on test set") if is_main else None
    print("=" * 60) if is_main else None

    pet_results, ct_results = evaluate(
        raw_model, test_dataset, args.batch_size,
        device, distributed, world_size, rank,
    )

    if is_main:
        os.makedirs(args.output_dir, exist_ok=True)
        save_pet_results_csv(pet_results,
                             os.path.join(args.output_dir, "pet_grading_results.csv"))
        save_ct_results_csv(ct_results,
                            os.path.join(args.output_dir, "ct_classification_results.csv"))

        summary_dir = str(Path(__file__).parent.parent.parent / "output")
        args.warmup_steps = warmup_steps  # 统一记录实际步数
        append_ct_summary(ct_results, args, summary_dir)
        append_pet_summary(pet_results, args, summary_dir)

    if distributed:
        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Train classification heads on frozen PETCTCLIP")
    p.add_argument("--pretrained_path", type=str, required=True,
                   help="Path to pretrained PETCTCLIP checkpoint directory")
    p.add_argument("--image_root", type=str,
                   default="/data/home/run/data/PETCT/images")
    p.add_argument("--labels_path", type=str,
                   default="/data/home/run/data/PETCT/aggregated_labels.xlsx")
    p.add_argument("--organ_labels_path", type=str,
                   default="/data/home/run/data/PETCT/organ_labels.jsonl")
    p.add_argument("--output_dir", type=str, default=None,
                   help="Directory to save classifier head. Defaults to "
                        "classify-head/<run_name>/<checkpoint_name>/ "
                        "mirroring the last two components of --pretrained_path.")
    p.add_argument("--max_steps", type=int, default=10000)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_steps", type=float, default=500,
                   help="Warmup steps: integer (e.g. 500) or ratio of max_steps (e.g. 0.05)")
    p.add_argument("--test_size", type=int, default=800)
    p.add_argument("--logging_steps", type=int, default=10)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.output_dir is None:
        pretrained = os.path.normpath(args.pretrained_path)
        # 取预训练路径的最后两级，例如 .../7-xxx/checkpoint-9500 -> 7-xxx/checkpoint-9500
        parts = pretrained.split(os.sep)
        suffix = os.path.join(*parts[-2:]) if len(parts) >= 2 else parts[-1]
        args.output_dir = os.path.join("classify-head", suffix)
    train(args)
