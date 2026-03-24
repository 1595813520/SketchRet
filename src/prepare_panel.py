import os
import json
import argparse
from collections import defaultdict
from typing import Any, Dict, List, Tuple, Optional
from functools import partial
from concurrent.futures import ThreadPoolExecutor

from PIL import Image

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
from torchvision.io import write_png

# =========================================================
# tqdm 进度条
# =========================================================
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable=None, *args, **kwargs):
        return iterable


# =========================================================
# 基础工具
# =========================================================

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def get_book_id(image_path: str) -> str:
    parts = image_path.replace("\\", "/").split("/")
    return parts[0] if len(parts) > 1 else "default_book"


def get_stem_no_ext(image_path: str) -> str:
    base = os.path.basename(image_path)
    stem, _ = os.path.splitext(base)
    return stem


def xyxy_area(bbox: List[float]) -> float:
    x0, y0, x1, y1 = bbox
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def bbox_center_x(bbox: List[float]) -> float:
    x0, _, x1, _ = bbox
    return 0.5 * (x0 + x1)


def clamp_bbox_xyxy(bbox: List[float], w: int, h: int) -> List[int]:
    x0, y0, x1, y1 = bbox
    x0 = max(0, min(int(round(x0)), w - 1))
    y0 = max(0, min(int(round(y0)), h - 1))
    x1 = max(x0 + 1, min(int(round(x1)), w))
    y1 = max(y0 + 1, min(int(round(y1)), h))
    return [x0, y0, x1, y1]


def xyxy_global_to_panel_local(
    bbox_xyxy: List[float],
    panel_xyxy: List[float],
) -> List[float]:
    bx0, by0, bx1, by1 = bbox_xyxy
    px0, py0, px1, py1 = panel_xyxy
    return [bx0 - px0, by0 - py0, bx1 - px0, by1 - py0]


def xyxy_local_to_norm(
    bbox_xyxy_local: List[float],
    panel_w: float,
    panel_h: float,
) -> List[float]:
    x0, y0, x1, y1 = bbox_xyxy_local
    pw = max(panel_w, 1e-6)
    ph = max(panel_h, 1e-6)
    return [x0 / pw, y0 / ph, x1 / pw, y1 / ph]


def transform_bbox_xyxy(
    bbox_xyxy_local: List[float],
    scale: float,
    pad_x: int,
    pad_y: int,
) -> List[float]:
    x0, y0, x1, y1 = bbox_xyxy_local
    return [
        x0 * scale + pad_x,
        y0 * scale + pad_y,
        x1 * scale + pad_x,
        y1 * scale + pad_y,
    ]


def save_tensor_png(img_chw_uint8: torch.Tensor, abs_path: str) -> None:
    """
    保存 CHW uint8 tensor 为 png
    """
    ensure_dir(os.path.dirname(abs_path))
    if img_chw_uint8.device.type != "cpu":
        img_chw_uint8 = img_chw_uint8.cpu()
    img_chw_uint8 = img_chw_uint8.contiguous()
    write_png(img_chw_uint8, abs_path)


def resolve_device(device_str: str) -> str:
    if device_str == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"

    if device_str.startswith("cuda") and not torch.cuda.is_available():
        print(f"[WARN] Requested device={device_str}, but CUDA is not available. Fallback to cpu.")
        return "cpu"

    return device_str


def resolve_image_path(root: str, image_path: str) -> Optional[str]:
    """
    修复标注扩展名和实际文件扩展名不一致的问题。
    例如：
      标注里：xxx/yyy.jpg 或 xxx/yyy.jpeg
      实际上：xxx/yyy.png
    """
    image_path = image_path.replace("\\", "/")
    rel_no_ext, original_ext = os.path.splitext(image_path)

    candidate_exts = [
        original_ext,
        ".png", ".PNG",
        ".jpg", ".JPG",
        ".jpeg", ".JPEG",
        ".webp", ".WEBP",
        ".bmp", ".BMP",
    ]

    seen = set()
    candidates = []

    raw_candidate = os.path.join(root, image_path)
    if raw_candidate not in seen:
        candidates.append(raw_candidate)
        seen.add(raw_candidate)

    for ext in candidate_exts:
        if not ext:
            continue
        p = os.path.join(root, rel_no_ext + ext)
        if p not in seen:
            candidates.append(p)
            seen.add(p)

    for p in candidates:
        if os.path.exists(p):
            return p

    return None


# =========================================================
# Torch 图像几何变换：长边缩放 + pad 到方图
# =========================================================

@torch.inference_mode()
def resize_long_edge_and_pad_square_tensor(
    img_chw_uint8: torch.Tensor,
    target_size: int,
    fill: Tuple[int, int, int] = (255, 255, 255),
    interpolation: InterpolationMode = InterpolationMode.BILINEAR,
    device: str = "cpu",
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    输入:
      img_chw_uint8: CHW, uint8
    输出:
      out_chw_uint8: CHW, uint8, CPU tensor
      meta: 几何元信息
    """
    if img_chw_uint8.ndim != 3:
        raise ValueError(f"Expected CHW tensor, got shape={tuple(img_chw_uint8.shape)}")

    c, orig_h, orig_w = img_chw_uint8.shape
    if c == 1:
        img_chw_uint8 = img_chw_uint8.repeat(3, 1, 1)
        c = 3
    elif c != 3:
        raise ValueError(f"Only support 1/3 channel image, got C={c}")

    long_edge = max(orig_w, orig_h)
    scale = float(target_size) / float(long_edge)

    scaled_w = max(1, int(round(orig_w * scale)))
    scaled_h = max(1, int(round(orig_h * scale)))

    x = img_chw_uint8.to(device=device, dtype=torch.float32).unsqueeze(0) / 255.0

    mode = "bicubic" if interpolation == InterpolationMode.BICUBIC else "bilinear"
    x = F.interpolate(
        x,
        size=(scaled_h, scaled_w),
        mode=mode,
        align_corners=False,
    )

    canvas = torch.empty((1, 3, target_size, target_size), device=device, dtype=torch.float32)
    fill_tensor = torch.tensor(fill, device=device, dtype=torch.float32).view(1, 3, 1, 1) / 255.0
    canvas[:] = fill_tensor

    pad_x = (target_size - scaled_w) // 2
    pad_y = (target_size - scaled_h) // 2
    canvas[:, :, pad_y:pad_y + scaled_h, pad_x:pad_x + scaled_w] = x

    out = (canvas.squeeze(0).clamp(0, 1) * 255.0).round().to(torch.uint8).cpu()

    meta = {
        "orig_size": [orig_w, orig_h],
        "scaled_size": [scaled_w, scaled_h],
        "scale_ratio": scale,
        "pad_offset": [pad_x, pad_y],
        "target_size": target_size,
    }
    return out, meta


# =========================================================
# canonical ref bank
# =========================================================

def build_canonical_ref_bank(records_per_spread: Dict[str, List[Dict[str, Any]]]) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """
    最终规则：
    - 同 spread、同 id
    - count == 1: 不建 ref
    - count >= 2: canonical ref = 最大 bbox
    """
    ref_bank: Dict[Tuple[str, str], Dict[str, Any]] = {}

    for spread_key, panel_records in records_per_spread.items():
        char_groups = defaultdict(list)

        for rec in panel_records:
            for ch in rec["characters_all"]:
                char_groups[ch["id"]].append(
                    {
                        "sample_id": rec["sample_id"],
                        "frame_index": rec["frame_index"],
                        "image_path": rec["image_path"],
                        "bbox_global": ch["bbox_global"],
                        "area_global": ch["area_global"],
                        "char_id": ch["id"],
                    }
                )

        for char_id, items in char_groups.items():
            if len(items) < 2:
                continue
            items = sorted(items, key=lambda x: x["area_global"], reverse=True)
            ref_bank[(spread_key, str(char_id))] = items[0]

    return ref_bank


def select_topk_panel_characters(chars: List[Dict[str, Any]], k: int = 3) -> List[Dict[str, Any]]:
    """
    panel 内用于 ref 的角色：
    1) 先按面积选 top-k
    2) 再按 x_center 从左到右排序
    """
    if len(chars) == 0:
        return []
    topk = sorted(chars, key=lambda c: c["area_global"], reverse=True)[:k]
    topk = sorted(topk, key=lambda c: c["x_center_norm"])
    return topk


# =========================================================
# 并行 worker：处理单个 spread
# =========================================================

@torch.inference_mode()
def process_single_spread(
    entry: Dict[str, Any],
    line_root: str,
    sketch_root: str,
    crop_root: str,
    panel_target_size: int,
    device: str,
) -> Dict[str, Any]:
    image_path = entry["image_path"]
    spread_key = image_path
    book_id = get_book_id(image_path)
    spread_stem = get_stem_no_ext(image_path)

    line_abs = resolve_image_path(line_root, image_path)
    sketch_abs = resolve_image_path(sketch_root, image_path)

    if not (line_abs and sketch_abs):
        return {
            "status": "skip",
            "reason": "missing_pair",
            "image_path": image_path,
            "panel_records": [],
            "line_abs": None,
        }

    try:
        line_img = Image.open(line_abs).convert("RGB")
        sketch_img = Image.open(sketch_abs).convert("RGB")
    except Exception as e:
        return {
            "status": "skip",
            "reason": f"open_failed: {e}",
            "image_path": image_path,
            "panel_records": [],
            "line_abs": None,
        }

    if line_img.size != sketch_img.size:
        return {
            "status": "skip",
            "reason": "size_mismatch",
            "image_path": image_path,
            "panel_records": [],
            "line_abs": None,
        }

    W, H = line_img.size

    line_tensor = TF.pil_to_tensor(line_img).contiguous()
    sketch_tensor = TF.pil_to_tensor(sketch_img).contiguous()

    if device != "cpu":
        line_tensor = line_tensor.to(device, non_blocking=True)
        sketch_tensor = sketch_tensor.to(device, non_blocking=True)

    frames = entry.get("frames", [])
    indexed_frames = sorted(
        list(enumerate(frames)),
        key=lambda x: (x[1]["bbox"][1], x[1]["bbox"][0]),
    )

    panel_records = []

    for panel_order_in_spread, (frame_idx, frame) in enumerate(indexed_frames):
        panel_global_bbox = clamp_bbox_xyxy(frame["bbox"], W, H)
        px0, py0, px1, py1 = panel_global_bbox

        # GPU / CPU tensor 裁剪
        line_panel_raw = line_tensor[:, py0:py1, px0:px1]
        sketch_panel_raw = sketch_tensor[:, py0:py1, px0:px1]

        # resize + pad
        line_panel_384, panel_geom = resize_long_edge_and_pad_square_tensor(
            line_panel_raw,
            target_size=panel_target_size,
            fill=(255, 255, 255),
            interpolation=InterpolationMode.BILINEAR,
            device=device,
        )
        sketch_panel_384, _ = resize_long_edge_and_pad_square_tensor(
            sketch_panel_raw,
            target_size=panel_target_size,
            fill=(255, 255, 255),
            interpolation=InterpolationMode.BILINEAR,
            device=device,
        )

        panel_line_rel = os.path.join(
            "panels", "line", book_id, f"{spread_stem}__f{frame_idx}.png"
        ).replace("\\", "/")
        panel_sketch_rel = os.path.join(
            "panels", "sketch", book_id, f"{spread_stem}__f{frame_idx}.png"
        ).replace("\\", "/")

        save_tensor_png(line_panel_384, os.path.join(crop_root, panel_line_rel))
        save_tensor_png(sketch_panel_384, os.path.join(crop_root, panel_sketch_rel))

        scale = panel_geom["scale_ratio"]
        pad_x, pad_y = panel_geom["pad_offset"]

        chars_out = []
        for ch in frame.get("characters", []):
            ch_bbox_global = clamp_bbox_xyxy(ch["bbox"], W, H)
            ch_bbox_local_orig = xyxy_global_to_panel_local(ch_bbox_global, panel_global_bbox)
            ch_bbox_384 = transform_bbox_xyxy(ch_bbox_local_orig, scale, pad_x, pad_y)
            ch_bbox_norm = xyxy_local_to_norm(ch_bbox_384, panel_target_size, panel_target_size)

            chars_out.append(
                {
                    "id": str(ch["id"]),
                    "type": ch.get("type", 0),
                    "bbox_global": ch_bbox_global,
                    "bbox_local_orig": [float(v) for v in ch_bbox_local_orig],
                    "bbox_384": [float(v) for v in ch_bbox_384],
                    "bbox_norm": [float(v) for v in ch_bbox_norm],
                    "area_global": float(xyxy_area(ch_bbox_global)),
                    "x_center_norm": float(0.5 * (ch_bbox_norm[0] + ch_bbox_norm[2])),
                }
            )

        record = {
            "sample_id": f"{spread_key}::{frame_idx}",
            "book_id": book_id,
            "spread_key": spread_key,
            "image_path": image_path,
            "frame_index": frame_idx,
            "panel_order_in_spread": panel_order_in_spread,
            "panel_global_bbox": panel_global_bbox,
            "panel_geom": panel_geom,
            "panel_line_path": panel_line_rel,
            "panel_sketch_path": panel_sketch_rel,
            "caption": frame.get("caption", ""),
            "characters_all": chars_out,
            "ref_selected": [],
        }

        panel_records.append(record)

    # 尽量释放显存
    del line_tensor, sketch_tensor
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    return {
        "status": "ok",
        "reason": "",
        "image_path": image_path,
        "panel_records": panel_records,
        "line_abs": line_abs,
    }


# =========================================================
# 并行 worker：保存单个 ref crop
# =========================================================

@torch.inference_mode()
def save_single_ref_crop(
    task: Dict[str, Any],
    crop_root: str,
    ref_target_size: int,
    device: str,
) -> bool:
    line_abs = task["line_abs"]
    ref_bbox_global = task["ref_bbox_global"]
    ref_crop_rel = task["ref_crop_rel"]

    try:
        line_img = Image.open(line_abs).convert("RGB")
        line_tensor = TF.pil_to_tensor(line_img).contiguous()
        if device != "cpu":
            line_tensor = line_tensor.to(device, non_blocking=True)

        h, w = line_img.size[1], line_img.size[0]
        ref_bbox = clamp_bbox_xyxy(ref_bbox_global, w, h)

        x0, y0, x1, y1 = ref_bbox
        ref_crop_raw = line_tensor[:, y0:y1, x0:x1]

        ref_crop_224, _ = resize_long_edge_and_pad_square_tensor(
            ref_crop_raw,
            target_size=ref_target_size,
            fill=(255, 255, 255),
            interpolation=InterpolationMode.BICUBIC,
            device=device,
        )

        save_tensor_png(ref_crop_224, os.path.join(crop_root, ref_crop_rel))

        del line_tensor
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

        return True
    except Exception as e:
        print(f"[WARN] Failed ref crop: {ref_crop_rel}, error={e}")
        return False


# =========================================================
# 主流程
# =========================================================

def preprocess_dataset(
    ann_json: str,
    line_root: str,
    sketch_root: str,
    crop_root: str,
    panel_target_size: int = 384,
    ref_target_size: int = 224,
    max_refs_per_panel: int = 3,
    num_workers: int = 4,
    device: str = "cpu",
) -> None:
    device = resolve_device(device)

    with open(ann_json, "r", encoding="utf-8") as f:
        anns = json.load(f)

    ensure_dir(crop_root)

    panel_records: List[Dict[str, Any]] = []
    records_per_spread: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    resolved_line_path_map: Dict[str, str] = {}

    total_spreads = 0
    skipped_missing_pair = 0
    total_panels = 0

    # -----------------------------------------------------
    # 第一步：并行展开 panel，裁 panel，保存 panel crop
    # -----------------------------------------------------
    worker_fn = partial(
        process_single_spread,
        line_root=line_root,
        sketch_root=sketch_root,
        crop_root=crop_root,
        panel_target_size=panel_target_size,
        device=device,
    )

    total_spreads = len(anns)

    if num_workers <= 1:
        results_iter = map(worker_fn, anns)
        results_iter = tqdm(results_iter, total=len(anns), desc="Step 1/3: crop panels", unit="spread")
        results = list(results_iter)
    else:
        with ThreadPoolExecutor(max_workers=num_workers) as ex:
            results_iter = ex.map(worker_fn, anns)
            results_iter = tqdm(results_iter, total=len(anns), desc="Step 1/3: crop panels", unit="spread")
            results = list(results_iter)

    for result in results:
        status = result["status"]
        image_path = result["image_path"]

        if status != "ok":
            skipped_missing_pair += 1
            print(f"[WARN] Skip spread: {image_path}, reason={result['reason']}")
            continue

        line_abs = result["line_abs"]
        if line_abs is not None:
            resolved_line_path_map[image_path] = line_abs

        cur_records = result["panel_records"]
        panel_records.extend(cur_records)
        total_panels += len(cur_records)

        for rec in cur_records:
            records_per_spread[rec["spread_key"]].append(rec)

    # -----------------------------------------------------
    # 第二步：构建 canonical ref bank
    # -----------------------------------------------------
    canonical_ref_bank = build_canonical_ref_bank(records_per_spread)

    # -----------------------------------------------------
    # 第三步：先写 ref_selected，再并行保存唯一 ref crop
    # -----------------------------------------------------
    ref_tasks: Dict[Tuple[str, str], Dict[str, Any]] = {}

    for rec in tqdm(panel_records, desc="Step 2/3: assign refs", unit="panel"):
        spread_key = rec["spread_key"]
        sample_id = rec["sample_id"]
        image_path = rec["image_path"]
        book_id = rec["book_id"]
        spread_stem = get_stem_no_ext(image_path)

        selected_chars = select_topk_panel_characters(rec["characters_all"], k=max_refs_per_panel)
        ref_selected = []

        for ch in selected_chars:
            char_id = ch["id"]
            key = (spread_key, char_id)

            if key not in canonical_ref_bank:
                ref_selected.append(
                    {
                        "char_id": char_id,
                        "char_bbox_norm": ch["bbox_norm"],
                        "ref_valid": 0,
                        "ref_panel_id": "",
                        "ref_crop_path": "",
                    }
                )
                continue

            ref_info = canonical_ref_bank[key]

            if ref_info["sample_id"] == sample_id:
                ref_selected.append(
                    {
                        "char_id": char_id,
                        "char_bbox_norm": ch["bbox_norm"],
                        "ref_valid": 0,
                        "ref_panel_id": "",
                        "ref_crop_path": "",
                    }
                )
                continue

            ref_crop_rel = os.path.join(
                "refs", book_id, f"{spread_stem}__char_{char_id}_{ref_target_size}.png"
            ).replace("\\", "/")

            ref_selected.append(
                {
                    "char_id": char_id,
                    "char_bbox_norm": ch["bbox_norm"],
                    "ref_valid": 1,
                    "ref_panel_id": ref_info["sample_id"],
                    "ref_crop_path": ref_crop_rel,
                }
            )

            if key not in ref_tasks:
                line_abs = resolved_line_path_map.get(spread_key, None)
                if line_abs is not None:
                    ref_tasks[key] = {
                        "spread_key": spread_key,
                        "char_id": char_id,
                        "line_abs": line_abs,
                        "ref_bbox_global": ref_info["bbox_global"],
                        "ref_crop_rel": ref_crop_rel,
                    }

        rec["ref_selected"] = ref_selected

    # 并行保存唯一 ref crop
    ref_task_list = list(ref_tasks.values())

    if num_workers <= 1:
        ref_results_iter = map(
            partial(save_single_ref_crop, crop_root=crop_root, ref_target_size=ref_target_size, device=device),
            ref_task_list,
        )
        ref_results_iter = tqdm(ref_results_iter, total=len(ref_task_list), desc="Step 3/3: save refs", unit="ref")
        ref_results = list(ref_results_iter)
    else:
        with ThreadPoolExecutor(max_workers=num_workers) as ex:
            ref_results_iter = ex.map(
                partial(save_single_ref_crop, crop_root=crop_root, ref_target_size=ref_target_size, device=device),
                ref_task_list,
            )
            ref_results_iter = tqdm(ref_results_iter, total=len(ref_task_list), desc="Step 3/3: save refs", unit="ref")
            ref_results = list(ref_results_iter)

    total_ref_crops = sum(1 for x in ref_results if x)

    # -----------------------------------------------------
    # 第四步：保存 panel_index.jsonl
    # -----------------------------------------------------
    panel_index_path = os.path.join(crop_root, "panel_index.jsonl")
    with open(panel_index_path, "w", encoding="utf-8") as f:
        for rec in panel_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # -----------------------------------------------------
    # 第五步：保存 manifest
    # -----------------------------------------------------
    manifest = {
        "ann_json": ann_json,
        "line_root": line_root,
        "sketch_root": sketch_root,
        "crop_root": crop_root,
        "panel_target_size": panel_target_size,
        "ref_target_size": ref_target_size,
        "max_refs_per_panel": max_refs_per_panel,
        "num_workers": num_workers,
        "device": device,
        "total_spreads": total_spreads,
        "skipped_missing_pair": skipped_missing_pair,
        "total_panels": total_panels,
        "total_ref_crops": total_ref_crops,
        "panel_index_jsonl": "panel_index.jsonl",
    }

    with open(os.path.join(crop_root, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print("========== Done ==========")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ann_json",
        type=str,
        default="/data/DiffSensei-main/checkpoints/mangazero/annotations.json",
    )
    parser.add_argument(
        "--line_root",
        type=str,
        default="/data/Sketch/manga_line/Anime2Sketch/anime_style",
    )
    parser.add_argument(
        "--sketch_root",
        type=str,
        default="/data/Sketch/manga_line/Anime2Sketch/opensketch_style",
    )
    parser.add_argument(
        "--crop_root",
        type=str,
        default="/data/Sketch/manga_line/crop",
    )
    parser.add_argument(
        "--panel_target_size",
        type=int,
        default=384,
    )
    parser.add_argument(
        "--ref_target_size",
        type=int,
        default=224,
    )
    parser.add_argument(
        "--max_refs_per_panel",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="并行 worker 数，CPU/IO 密集场景建议 4~8，CUDA 模式建议先试 2~4",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="cpu / cuda / cuda:0 / auto",
    )

    args = parser.parse_args()

    preprocess_dataset(
        ann_json=args.ann_json,
        line_root=args.line_root,
        sketch_root=args.sketch_root,
        crop_root=args.crop_root,
        panel_target_size=args.panel_target_size,
        ref_target_size=args.ref_target_size,
        max_refs_per_panel=args.max_refs_per_panel,
        num_workers=args.num_workers,
        device=args.device,
    )


if __name__ == "__main__":
    main()
    
    
'''
nohup python /data/Sketch/src/prepare_panel.py --device cuda:0 --num_workers 8 > /data/Sketch/nohup.log 2>&1 &
'''
