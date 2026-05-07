#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageOps
from scipy.linalg import sqrtm
from scipy.ndimage import distance_transform_edt

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from transformers import AutoImageProcessor, AutoModel, CLIPImageProcessor, CLIPVisionModelWithProjection

try:
    import lpips  # type: ignore
except Exception:
    lpips = None


@dataclass
class EvalItem:
    index: int
    sample_id: str
    pred_panel_path: Path
    gt_panel_path: Path
    sketch_panel_path: Optional[Path]
    ref_selected: List[Dict[str, Any]]


class SquarePadImageDataset(Dataset):
    def __init__(self, paths: Sequence[Path], image_size: int, pad_value: int = 255):
        self.paths = list(paths)
        self.image_size = image_size
        self.pad_value = pad_value

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        path = self.paths[idx]
        img = Image.open(path)
        img = ImageOps.exif_transpose(img).convert("RGB")
        w, h = img.size
        scale = self.image_size / max(w, h)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        resized = img.resize((new_w, new_h), Image.Resampling.BICUBIC)
        canvas = Image.new("RGB", (self.image_size, self.image_size), (self.pad_value, self.pad_value, self.pad_value))
        off_x = (self.image_size - new_w) // 2
        off_y = (self.image_size - new_h) // 2
        canvas.paste(resized, (off_x, off_y))
        arr = np.asarray(canvas).astype(np.float32) / 255.0
        x = torch.from_numpy(arr).permute(2, 0, 1)
        return x, str(path)


def safe_image_open(path: Path, mode: str = "RGB") -> Image.Image:
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)
    return img.convert(mode)


def image_to_tensor_01(path: Path) -> torch.Tensor:
    img = safe_image_open(path, mode="RGB")
    arr = np.asarray(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)


def line_binary_map(path: Path, threshold: int = 245) -> np.ndarray:
    img = safe_image_open(path, mode="L")
    arr = np.asarray(img)
    return arr < threshold


def chamfer_distance_sym(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(bool)
    b = b.astype(bool)
    if a.sum() == 0 and b.sum() == 0:
        return 0.0
    if a.sum() == 0 or b.sum() == 0:
        return float("inf")
    dist_to_b = distance_transform_edt(~b)
    dist_to_a = distance_transform_edt(~a)
    d_ab = float(dist_to_b[a].mean()) if a.any() else 0.0
    d_ba = float(dist_to_a[b].mean()) if b.any() else 0.0
    return 0.5 * (d_ab + d_ba)


def bf_score_and_slr_from_binary(pred_bin: np.ndarray, gt_bin: np.ndarray, tau: float) -> Tuple[float, float]:
    pred_bin = pred_bin.astype(bool)
    gt_bin = gt_bin.astype(bool)
    pred_count = int(pred_bin.sum())
    gt_count = int(gt_bin.sum())
    if pred_count == 0 and gt_count == 0:
        return 1.0, 0.0
    if pred_count == 0:
        return 0.0, 0.0
    if gt_count == 0:
        return 0.0, float(pred_count)

    dist_to_gt = distance_transform_edt(~gt_bin)
    dist_to_pred = distance_transform_edt(~pred_bin)

    pred_match = pred_bin & (dist_to_gt <= float(tau))
    gt_match = gt_bin & (dist_to_pred <= float(tau))

    precision = float(pred_match.sum()) / max(pred_count, 1)
    recall = float(gt_match.sum()) / max(gt_count, 1)
    bf = 2.0 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0

    spurious = pred_bin & (dist_to_gt > float(tau))
    slr = float(spurious.sum()) / max(gt_count, 1)
    return float(bf), float(slr)


def crop_with_bbox_norm(img: Image.Image, bbox_norm: Sequence[float], min_side: int = 4) -> Image.Image:
    img = ImageOps.exif_transpose(img).convert("RGB")
    w, h = img.size
    if len(bbox_norm) != 4:
        return img
    x1, y1, x2, y2 = [float(v) for v in bbox_norm]
    x1 = max(0.0, min(1.0, x1))
    y1 = max(0.0, min(1.0, y1))
    x2 = max(x1 + 1e-6, min(1.0, x2))
    y2 = max(y1 + 1e-6, min(1.0, y2))
    l = int(np.floor(x1 * w))
    t = int(np.floor(y1 * h))
    r = int(np.ceil(x2 * w))
    b = int(np.ceil(y2 * h))
    r = max(l + min_side, r)
    b = max(t + min_side, b)
    l = max(0, min(w - 1, l))
    t = max(0, min(h - 1, t))
    r = max(l + 1, min(w, r))
    b = max(t + 1, min(h, b))
    return img.crop((l, t, r, b))


def resolve_path_maybe_relative(path_value: str, root: Path) -> Path:
    p = Path(path_value)
    if p.is_absolute():
        return p
    return (root / p).resolve()


class InceptionFeatureExtractor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        weights = models.Inception_V3_Weights.IMAGENET1K_V1
        model = models.inception_v3(weights=weights, transform_input=False)
        model.fc = torch.nn.Identity()
        self.model = model.eval()
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)
        x = (x - self.mean) / self.std
        x = self.model(x)
        if x.ndim > 2:
            x = torch.flatten(x, 1)
        return x


class CLIPFeatureExtractor(torch.nn.Module):
    def __init__(self, model_name: str, normalize_embeds: bool = False):
        super().__init__()
        self.processor = CLIPImageProcessor.from_pretrained(model_name)
        self.model = CLIPVisionModelWithProjection.from_pretrained(model_name).eval()
        self.normalize_embeds = normalize_embeds

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        imgs = [img.detach().cpu() for img in x]
        proc = self.processor(images=imgs, return_tensors="pt")
        proc = {k: v.to(x.device) for k, v in proc.items()}
        feat = self.model(**proc).image_embeds
        if self.normalize_embeds:
            feat = F.normalize(feat, dim=-1)
        return feat


class DINOFeatureExtractor(torch.nn.Module):
    def __init__(self, model_name: str):
        super().__init__()
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).eval()

    def forward(self, x) -> torch.Tensor:
        if isinstance(x, (list, tuple)):
            imgs = []
            for img in x:
                if isinstance(img, Image.Image):
                    imgs.append(ImageOps.exif_transpose(img).convert("RGB"))
                elif isinstance(img, torch.Tensor):
                    imgs.append(img.detach().cpu())
                else:
                    imgs.append(img)
            device = next(self.model.parameters()).device
        else:
            imgs = [img.detach().cpu() for img in x]
            device = x.device
        proc = self.processor(images=imgs, return_tensors="pt")
        proc = {k: v.to(device) for k, v in proc.items()}
        out = self.model(**proc)
        feat = getattr(out, "pooler_output", None)
        if feat is None:
            feat = out.last_hidden_state[:, 0]
        feat = F.normalize(feat, dim=-1)
        return feat


def compute_features(paths: Sequence[Path], model: torch.nn.Module, device: str, batch_size: int, num_workers: int, image_size: int) -> np.ndarray:
    ds = SquarePadImageDataset(paths, image_size=image_size)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    model = model.to(device).eval()
    feats: List[np.ndarray] = []
    with torch.inference_mode():
        for imgs, _ in dl:
            imgs = imgs.to(device, non_blocking=True)
            cur = model(imgs)
            feats.append(cur.detach().cpu().numpy())
    return np.concatenate(feats, axis=0)


def compute_fid_from_features(real_feats: np.ndarray, fake_feats: np.ndarray, eps: float = 1e-6) -> float:
    real_feats = np.asarray(real_feats, dtype=np.float64)
    fake_feats = np.asarray(fake_feats, dtype=np.float64)

    mu1 = np.mean(real_feats, axis=0)
    mu2 = np.mean(fake_feats, axis=0)
    sigma1 = np.cov(real_feats, rowvar=False)
    sigma2 = np.cov(fake_feats, rowvar=False)

    if sigma1.ndim == 0:
        sigma1 = np.array([[float(sigma1)]], dtype=np.float64)
    if sigma2.ndim == 0:
        sigma2 = np.array([[float(sigma2)]], dtype=np.float64)

    sigma1 = sigma1 + np.eye(sigma1.shape[0], dtype=np.float64) * eps
    sigma2 = sigma2 + np.eye(sigma2.shape[0], dtype=np.float64) * eps

    covmean = sqrtm(sigma1.dot(sigma2))
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0], dtype=np.float64) * (eps * 10.0)
        covmean = sqrtm((sigma1 + offset).dot(sigma2 + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    diff = mu1 - mu2
    fid = float(diff.dot(diff) + np.trace(sigma1 + sigma2 - 2.0 * covmean))
    if not np.isfinite(fid):
        raise ValueError(f"FID became non-finite: {fid}")
    return max(fid, 0.0)


def load_manifest(path: Path) -> List[EvalItem]:
    items: List[EvalItem] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            items.append(EvalItem(
                index=int(raw.get("index", len(items))),
                sample_id=str(raw["sample_id"]),
                pred_panel_path=Path(raw["pred_panel_path"]),
                gt_panel_path=Path(raw["gt_panel_path"]),
                sketch_panel_path=Path(raw["sketch_panel_path"]) if raw.get("sketch_panel_path") else None,
                ref_selected=raw.get("ref_selected", []) or [],
            ))
    return items


def compute_char_dino(valid: List[EvalItem], crop_root: Path, model_name: str, device: str, batch_size: int) -> Tuple[Optional[float], Dict[str, float]]:
    pair_imgs_pred: List[Image.Image] = []
    pair_imgs_ref: List[Image.Image] = []
    pair_owner: List[int] = []

    for midx, m in enumerate(valid):
        pred_panel = safe_image_open(m.pred_panel_path, mode="RGB")
        valid_ref_idx = 0
        for ref in m.ref_selected:
            ref_path_value = ref.get("ref_crop_path") or ref.get("path") or ref.get("ref_path")
            bbox_norm = ref.get("char_bbox_norm") or ref.get("bbox_norm") or ref.get("bbox")
            ref_valid = ref.get("ref_valid", ref.get("valid", 1))
            if not ref_path_value or not bbox_norm or not bool(ref_valid):
                continue
            ref_path = resolve_path_maybe_relative(str(ref_path_value), crop_root)
            if not ref_path.is_file():
                continue
            try:
                ref_img = safe_image_open(ref_path, mode="RGB")
                pred_crop = crop_with_bbox_norm(pred_panel, bbox_norm)
            except Exception:
                continue
            pair_imgs_pred.append(pred_crop)
            pair_imgs_ref.append(ref_img)
            pair_owner.append(midx)
            valid_ref_idx += 1

    if len(pair_imgs_pred) == 0:
        return None, {"num_char_pairs": 0.0, "num_samples_with_char_refs": 0.0}

    extractor = DINOFeatureExtractor(model_name).to(device).eval()
    sims: List[float] = []
    per_sample: Dict[int, List[float]] = {}
    with torch.inference_mode():
        for start in range(0, len(pair_imgs_pred), batch_size):
            pred_batch = pair_imgs_pred[start:start+batch_size]
            ref_batch = pair_imgs_ref[start:start+batch_size]
            pred_feat = extractor(pred_batch)
            ref_feat = extractor(ref_batch)
            sim_batch = F.cosine_similarity(pred_feat, ref_feat, dim=-1).detach().cpu().numpy().tolist()
            for off, sim in enumerate(sim_batch):
                owner = pair_owner[start + off]
                sims.append(float(sim))
                per_sample.setdefault(owner, []).append(float(sim))

    sample_means = [float(np.mean(v)) for v in per_sample.values() if len(v) > 0]
    overall = float(np.mean(sample_means)) if sample_means else float(np.mean(sims))
    return overall, {
        "num_char_pairs": float(len(sims)),
        "num_samples_with_char_refs": float(len(sample_means)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate release-style generated panel outputs.")
    parser.add_argument("--manifest_jsonl", required=True)
    parser.add_argument("--crop_root", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resolution", type=int, default=384)
    parser.add_argument("--metric_batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--clip_model_name", default="openai/clip-vit-base-patch32")
    parser.add_argument("--clip_normalize_for_fidc", action="store_true")
    parser.add_argument("--line_threshold", type=int, default=245)
    parser.add_argument("--bf_tau", type=float, default=2.0)
    parser.add_argument("--char_dino_model_name_or_path", default="facebook/dinov2-base")
    parser.add_argument("--char_dino_batch_size", type=int, default=32)
    parser.add_argument("--disable_char_dino", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    valid = [m for m in load_manifest(Path(args.manifest_jsonl)) if m.pred_panel_path.exists() and m.gt_panel_path.exists()]
    if not valid:
        raise RuntimeError("No valid pred/gt panel pairs found in manifest.")

    device = args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu"
    gt_paths = [m.gt_panel_path for m in valid]
    pred_paths = [m.pred_panel_path for m in valid]

    inception = InceptionFeatureExtractor()
    real_inception = compute_features(gt_paths, inception, device, args.metric_batch_size, args.num_workers, args.resolution)
    fake_inception = compute_features(pred_paths, inception, device, args.metric_batch_size, args.num_workers, args.resolution)
    fid_i = compute_fid_from_features(real_inception, fake_inception)

    clip_ext = CLIPFeatureExtractor(args.clip_model_name, normalize_embeds=args.clip_normalize_for_fidc)
    real_clip = compute_features(gt_paths, clip_ext, device, args.metric_batch_size, args.num_workers, args.resolution)
    fake_clip = compute_features(pred_paths, clip_ext, device, args.metric_batch_size, args.num_workers, args.resolution)
    fid_c = compute_fid_from_features(real_clip, fake_clip)

    if lpips is None:
        raise ImportError("lpips is not installed. Please install it first: pip install lpips")
    lpips_fn = lpips.LPIPS(net="alex").to(device).eval()

    rows: List[Dict[str, Any]] = []
    lpips_vals: List[float] = []
    chamfer_vals: List[float] = []
    bf_vals: List[float] = []
    slr_vals: List[float] = []

    with torch.inference_mode():
        for m in valid:
            pred_t = image_to_tensor_01(m.pred_panel_path).unsqueeze(0)
            gt_t = image_to_tensor_01(m.gt_panel_path).unsqueeze(0)
            if pred_t.shape[-2:] != gt_t.shape[-2:]:
                raise AssertionError(
                    f"panel pair size mismatch for {m.sample_id}: pred={tuple(pred_t.shape[-2:])} gt={tuple(gt_t.shape[-2:])}"
                )
            pred_lp = (pred_t * 2.0 - 1.0).to(device)
            gt_lp = (gt_t * 2.0 - 1.0).to(device)
            lpv = float(lpips_fn(pred_lp, gt_lp).item())
            lpips_vals.append(lpv)

            pred_bin = line_binary_map(m.pred_panel_path, threshold=args.line_threshold)
            gt_bin = line_binary_map(m.gt_panel_path, threshold=args.line_threshold)
            cd = chamfer_distance_sym(pred_bin, gt_bin)
            bf, slr = bf_score_and_slr_from_binary(pred_bin, gt_bin, tau=args.bf_tau)
            chamfer_vals.append(cd)
            bf_vals.append(bf)
            slr_vals.append(slr)

            rows.append({
                "index": m.index,
                "sample_id": m.sample_id,
                "pred_panel_path": str(m.pred_panel_path),
                "gt_panel_path": str(m.gt_panel_path),
                "bf_score_tau": bf,
                "slr": slr,
                "lpips": lpv,
                "fid_i_scope": "dataset",
                "fid_c_scope": "dataset",
                "chamfer_distance": cd,
            })

    char_dino = None
    char_dino_meta: Dict[str, float] = {"num_char_pairs": 0.0, "num_samples_with_char_refs": 0.0}
    if not args.disable_char_dino:
        char_dino, char_dino_meta = compute_char_dino(valid, Path(args.crop_root), args.char_dino_model_name_or_path, device, args.char_dino_batch_size)

    for row in rows:
        row["char_dino"] = char_dino if char_dino is not None else ""

    if rows:
        with (output_root / "per_image_metrics_release.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    metrics = {
        "BF-score@τ": float(np.mean(bf_vals)),
        "SLR": float(np.mean(slr_vals)),
        "LPIPS": float(np.mean(lpips_vals)),
        "FID-I": float(fid_i),
        "FID-C": float(fid_c),
        "Chamfer Distance": float(np.mean(chamfer_vals)),
        "Char-DINO": None if char_dino is None else float(char_dino),
    }

    summary = {
        "manifest_jsonl": str(args.manifest_jsonl),
        "crop_root": str(args.crop_root),
        "output_root": str(args.output_root),
        "num_evaluated": len(valid),
        "evaluation_space": "release_panel_space",
        "metric_config": {
            "line_threshold": args.line_threshold,
            "bf_tau": args.bf_tau,
            "char_dino_model_name_or_path": None if args.disable_char_dino else args.char_dino_model_name_or_path,
            "char_dino_batch_size": args.char_dino_batch_size,
        },
        "metrics": metrics,
        "char_dino_meta": char_dino_meta,
    }
    with (output_root / "metrics_summary_release.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[OK] summary -> {output_root / 'metrics_summary_release.json'}")
    print(f"[OK] per-image metrics -> {output_root / 'per_image_metrics_release.csv'}")


if __name__ == "__main__":
    main()
