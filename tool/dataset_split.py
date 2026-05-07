#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import math
import os
import random
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

BUCKET_ORDER = ["no_char", "single_char", "multi_char"]


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def safe_filename_token(value: Any) -> str:
    s = str(value)
    return s.replace("/", "__").replace("\\", "__").replace("::", "__").replace(":", "_")


def xyxy_area(bbox: List[float]) -> float:
    x0, y0, x1, y1 = bbox
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def clamp_bbox_xyxy(bbox: List[float], w: int, h: int) -> Optional[List[int]]:
    if not bbox or len(bbox) != 4:
        return None
    x0, y0, x1, y1 = bbox
    x0 = max(0, min(int(round(float(x0))), w - 1))
    y0 = max(0, min(int(round(float(y0))), h - 1))
    x1 = max(x0 + 1, min(int(round(float(x1))), w))
    y1 = max(y0 + 1, min(int(round(float(y1))), h))
    if x1 <= x0 or y1 <= y0:
        return None
    return [x0, y0, x1, y1]


def resize_long_edge_and_pad_square_pil(
    img: Image.Image,
    target_size: int,
    fill=(255, 255, 255),
    resample=Image.BICUBIC,
):
    orig_w, orig_h = img.size
    long_edge = max(orig_w, orig_h)
    scale = float(target_size) / float(max(long_edge, 1))
    scaled_w = max(1, int(round(orig_w * scale)))
    scaled_h = max(1, int(round(orig_h * scale)))
    resized = img.convert("RGB").resize((scaled_w, scaled_h), resample=resample)
    canvas = Image.new("RGB", (target_size, target_size), fill)
    pad_x = (target_size - scaled_w) // 2
    pad_y = (target_size - scaled_h) // 2
    canvas.paste(resized, (pad_x, pad_y))
    return canvas


def has_valid_ref(rec: Dict[str, Any]) -> bool:
    return any(x.get("ref_valid", 0) == 1 for x in rec.get("ref_selected", []))


def panel_char_bucket(rec: Dict[str, Any]) -> str:
    n = len(rec.get("characters", []) or [])
    if n == 0:
        return "no_char"
    if n == 1:
        return "single_char"
    return "multi_char"


def count_bucket_dict(records: List[Dict[str, Any]]) -> Dict[str, int]:
    cnt = Counter(panel_char_bucket(r) for r in records)
    return {k: int(cnt.get(k, 0)) for k in BUCKET_ORDER}


def parse_ratio_string(ratio: str) -> Dict[str, int]:
    if not ratio:
        raise ValueError("test bucket ratio must be non-empty, e.g. '1:5:4'")
    parts = [p.strip() for p in ratio.replace(",", ":").replace("/", ":").split(":") if p.strip()]
    if len(parts) != 3:
        raise ValueError(f"Invalid ratio: {ratio}. Expected format like '1:5:4'.")
    weights = [int(p) for p in parts]
    if any(w < 0 for w in weights) or sum(weights) <= 0:
        raise ValueError(f"Invalid ratio weights: {ratio}")
    return {
        "no_char": weights[0],
        "single_char": weights[1],
        "multi_char": weights[2],
    }


def allocate_target_counts(total: int, ratio_map: Dict[str, int]) -> Dict[str, int]:
    if total <= 0:
        return {k: 0 for k in BUCKET_ORDER}
    weight_sum = sum(ratio_map[k] for k in BUCKET_ORDER)
    raw = {k: total * ratio_map[k] / float(weight_sum) for k in BUCKET_ORDER}
    out = {k: int(math.floor(raw[k])) for k in BUCKET_ORDER}
    remain = total - sum(out.values())
    if remain > 0:
        frac_order = sorted(BUCKET_ORDER, key=lambda k: (raw[k] - out[k]), reverse=True)
        for k in frac_order[:remain]:
            out[k] += 1
    return out


def load_release_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception as e:
                raise ValueError(f"Bad JSON at line {line_idx} of {path}: {e}") from e
    return rows


def resolve_rel_or_abs(crop_root: str, p: str) -> str:
    pp = Path(str(p))
    if pp.is_absolute():
        return str(pp)
    return str((Path(crop_root) / pp).resolve())


def select_topk_panel_characters(chars: List[Dict[str, Any]], k: int = 3) -> List[Dict[str, Any]]:
    if len(chars) == 0:
        return []
    topk = sorted(chars, key=lambda c: c.get("area", xyxy_area(c.get("bbox", [0, 0, 0, 0]))), reverse=True)[:k]
    topk = sorted(
        topk,
        key=lambda c: c.get(
            "x_center_norm",
            0.5 * (c.get("bbox_norm", [0, 0, 1, 1])[0] + c.get("bbox_norm", [0, 0, 1, 1])[2]),
        ),
    )
    return topk


def build_ref_candidate_bank(records_per_ref_unit: Dict[str, List[Dict[str, Any]]]) -> Dict[Tuple[str, str], List[Dict[str, Any]]]:
    ref_bank: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for ref_unit_key, panel_records in records_per_ref_unit.items():
        char_groups = defaultdict(list)
        for rec in panel_records:
            for ch in rec["characters"]:
                char_groups[ch["id"]].append(
                    {
                        "sample_id": rec["sample_id"],
                        "dataset_id": rec["dataset_id"],
                        "page_id": rec["page_id"],
                        "panel_side": rec["panel_side"],
                        "panel_id": rec["panel_id"],
                        "panel_line_path": rec["panel_line_path"],
                        "bbox": ch["bbox"],
                        "bbox_norm": ch["bbox_norm"],
                        "area": ch["area"],
                    }
                )
        for char_id, items in char_groups.items():
            if len(items) < 2:
                continue
            items = sorted(items, key=lambda x: x["area"], reverse=True)
            ref_bank[(ref_unit_key, str(char_id))] = items
    return ref_bank


def select_ref_candidate(candidates: List[Dict[str, Any]], current_sample_id: str) -> Optional[Dict[str, Any]]:
    for cand in candidates:
        if cand["sample_id"] != current_sample_id:
            return cand
    return None


def save_single_ref_crop(task: Dict[str, Any], crop_root: str, ref_target_size: int) -> bool:
    try:
        panel_line_abs = resolve_rel_or_abs(crop_root, task["panel_line_path"])
        if not os.path.exists(panel_line_abs):
            return False
        img = Image.open(panel_line_abs).convert("RGB")
        w, h = img.size
        ref_bbox = clamp_bbox_xyxy(task["ref_bbox"], w, h)
        if ref_bbox is None:
            return False
        x0, y0, x1, y1 = ref_bbox
        crop = img.crop((x0, y0, x1, y1))
        crop = resize_long_edge_and_pad_square_pil(crop, target_size=ref_target_size, fill=(255, 255, 255), resample=Image.BICUBIC)
        ref_crop_abs = os.path.join(crop_root, task["ref_crop_rel"])
        ensure_dir(os.path.dirname(ref_crop_abs))
        crop.save(ref_crop_abs)
        return True
    except Exception:
        return False


def stratified_panel_split(
    panel_records: List[Dict[str, Any]],
    test_limit: int,
    ratio_map: Dict[str, int],
    rng: random.Random,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    target_counts = allocate_target_counts(test_limit, ratio_map)
    by_bucket: Dict[str, List[Dict[str, Any]]] = {k: [] for k in BUCKET_ORDER}
    for rec in panel_records:
        by_bucket[panel_char_bucket(rec)].append(rec)

    for k in BUCKET_ORDER:
        rng.shuffle(by_bucket[k])

    test_records: List[Dict[str, Any]] = []
    selected_ids = set()
    actual_target_pick = {}
    for k in BUCKET_ORDER:
        take_n = min(target_counts[k], len(by_bucket[k]))
        actual_target_pick[k] = take_n
        for rec in by_bucket[k][:take_n]:
            test_records.append(rec)
            selected_ids.add(rec["sample_id"])

    shortage = max(0, test_limit - len(test_records))
    if shortage > 0:
        remaining_pool = [r for r in panel_records if r["sample_id"] not in selected_ids]
        rng.shuffle(remaining_pool)
        for rec in remaining_pool[:shortage]:
            test_records.append(rec)
            selected_ids.add(rec["sample_id"])

    train_records = [r for r in panel_records if r["sample_id"] not in selected_ids]
    info = {
        "split_unit": "panel",
        "test_target_panels": test_limit,
        "requested_test_bucket_ratio": ratio_map,
        "target_test_bucket_counts": target_counts,
        "initial_bucket_picks": actual_target_pick,
        "selected_test_panels": len(test_records),
        "selected_test_units": len(test_records),
        "selected_test_bucket_counts": count_bucket_dict(test_records),
        "selected_train_bucket_counts": count_bucket_dict(train_records),
        "enough_for_target": len(panel_records) >= test_limit,
    }
    return train_records, test_records, info


def _unit_selection_priority(
    unit_counts: Dict[str, int],
    unit_total: int,
    need: Dict[str, int],
    current_total: int,
    target_total: int,
) -> Tuple[int, int, int, int]:
    contribution = sum(min(max(need[k], 0), unit_counts.get(k, 0)) for k in BUCKET_ORDER)
    overshoot = max(0, current_total + unit_total - target_total)
    remaining_total = max(0, target_total - current_total)
    fit_gap = abs(unit_total - remaining_total)
    return (contribution, -overshoot, -fit_gap, -unit_total)



def stratified_unit_split(
    panel_records: List[Dict[str, Any]],
    test_limit: int,
    ratio_map: Dict[str, int],
    rng: random.Random,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in panel_records:
        groups[r["ref_unit_key"]].append(r)

    candidate_units = []
    for unit_key, recs in groups.items():
        unit_counts = count_bucket_dict(recs)
        candidate_units.append(
            {
                "unit_key": unit_key,
                "records": recs,
                "counts": unit_counts,
                "total": len(recs),
            }
        )

    rng.shuffle(candidate_units)
    target_counts = allocate_target_counts(test_limit, ratio_map)
    need = dict(target_counts)

    selected_units = []
    selected_unit_keys = set()
    current_total = 0
    current_counts = {k: 0 for k in BUCKET_ORDER}
    remaining_units = list(candidate_units)

    while remaining_units and current_total < test_limit:
        best_idx = None
        best_priority = None
        for idx, unit in enumerate(remaining_units):
            prio = _unit_selection_priority(
                unit_counts=unit["counts"],
                unit_total=unit["total"],
                need=need,
                current_total=current_total,
                target_total=test_limit,
            )
            if best_priority is None or prio > best_priority:
                best_priority = prio
                best_idx = idx

        if best_idx is None:
            break

        unit = remaining_units.pop(best_idx)
        selected_units.append(unit)
        selected_unit_keys.add(unit["unit_key"])
        current_total += unit["total"]
        for k in BUCKET_ORDER:
            current_counts[k] += unit["counts"].get(k, 0)
            need[k] -= unit["counts"].get(k, 0)

    test_records = []
    for unit in selected_units:
        test_records.extend(unit["records"])

    train_records = [r for r in panel_records if r["ref_unit_key"] not in selected_unit_keys]

    info = {
        "split_unit": "side_page",
        "test_target_panels": test_limit,
        "requested_test_bucket_ratio": ratio_map,
        "target_test_bucket_counts": target_counts,
        "selected_test_panels": len(test_records),
        "selected_test_units": len(selected_units),
        "selected_test_bucket_counts": count_bucket_dict(test_records),
        "selected_train_bucket_counts": count_bucket_dict(train_records),
        "selected_unit_size_stats": {
            "min": min((u["total"] for u in selected_units), default=0),
            "max": max((u["total"] for u in selected_units), default=0),
            "avg": round(sum(u["total"] for u in selected_units) / max(len(selected_units), 1), 4),
        },
        "enough_for_target": len(panel_records) >= test_limit,
    }
    return train_records, test_records, info


def split_dataset(
    panel_records: List[Dict[str, Any]],
    test_limit: int,
    seed: int,
    split_unit: str = "side_page",
    test_bucket_ratio: str = "1:5:4",
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    rng = random.Random(seed)
    ratio_map = parse_ratio_string(test_bucket_ratio)

    if split_unit == "panel":
        train_records, test_records, info = stratified_panel_split(
            panel_records=panel_records,
            test_limit=test_limit,
            ratio_map=ratio_map,
            rng=rng,
        )
    else:
        train_records, test_records, info = stratified_unit_split(
            panel_records=panel_records,
            test_limit=test_limit,
            ratio_map=ratio_map,
            rng=rng,
        )

    info["selected_test_valid_ref_panels"] = sum(1 for r in test_records if r.get("has_valid_ref", False))
    info["selected_test_valid_ref_bucket_counts"] = count_bucket_dict([r for r in test_records if r.get("has_valid_ref", False)])
    return train_records, test_records, info


def preprocess_release_for_training(
    release_jsonl: str,
    crop_root: str,
    output_root: str,
    output_name: str = "panel_index_train_ref.jsonl",
    ref_target_size: int = 224,
    max_refs_per_panel: int = 5,
    test_ref_limit: int = 1000,
    split_unit: str = "side_page",
    test_bucket_ratio: str = "1:5:4",
    seed: int = 42,
    num_workers: int = 8,
) -> None:
    ensure_dir(output_root)
    ensure_dir(os.path.join(output_root, "splits"))

    rows = load_release_jsonl(release_jsonl)
    panel_records: List[Dict[str, Any]] = []
    records_per_ref_unit: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for row in rows:
        dataset_id = str(row.get("dataset_id", ""))
        page_id = str(row.get("page_id", ""))
        panel_side = str(row.get("panel_side", ""))
        panel_id = row.get("panel_id", 0)
        sample_id = str(row.get("sample_id", f"{dataset_id}/{page_id}::{panel_id}"))
        panel_line_path = row.get("panel_line_path", "")
        panel_sketch_path = row.get("panel_sketch_path", "")
        if not panel_line_path or not panel_sketch_path or not dataset_id or not page_id or not panel_side:
            continue

        characters_out = []
        for ch in row.get("characters", []) or []:
            bbox = ch.get("bbox")
            bbox_norm = ch.get("bbox_norm")
            if not bbox or len(bbox) != 4 or not bbox_norm or len(bbox_norm) != 4:
                continue
            area = float(xyxy_area(bbox))
            x_center_norm = float(0.5 * (bbox_norm[0] + bbox_norm[2]))
            characters_out.append(
                {
                    "id": str(ch.get("id", "")),
                    "name": ch.get("name", ""),
                    "state_cues": ch.get("state_cues", []) or [],
                    "bbox": [float(v) for v in bbox],
                    "bbox_norm": [float(v) for v in bbox_norm],
                    "area": area,
                    "x_center_norm": x_center_norm,
                }
            )

        ref_unit_key = f"{dataset_id}/{page_id}/{panel_side}"
        rec = {
            "sample_id": sample_id,
            "dataset_id": dataset_id,
            "page_id": page_id,
            "page_key": f"{dataset_id}/{page_id}",
            "panel_side": panel_side,
            "ref_unit_key": ref_unit_key,
            "panel_id": panel_id,
            "panel_line_path": panel_line_path,
            "panel_sketch_path": panel_sketch_path,
            "width": row.get("width"),
            "height": row.get("height"),
            "caption": row.get("caption", ""),
            "story": row.get("story", ""),
            "characters": characters_out,
            "ref_selected": [],
            "has_valid_ref": False,
        }
        panel_records.append(rec)
        records_per_ref_unit[ref_unit_key].append(rec)

    ref_candidate_bank = build_ref_candidate_bank(records_per_ref_unit)
    ref_tasks: Dict[Tuple[str, str, str], Dict[str, Any]] = {}

    for rec in panel_records:
        sample_id = rec["sample_id"]
        dataset_id = rec["dataset_id"]
        page_id = rec["page_id"]
        side = rec["panel_side"]
        ref_unit_key = rec["ref_unit_key"]

        selected_chars = select_topk_panel_characters(rec["characters"], k=max_refs_per_panel)
        ref_selected = []

        for ch in selected_chars:
            char_id = ch["id"]
            candidates = ref_candidate_bank.get((ref_unit_key, char_id), [])
            ref_info = select_ref_candidate(candidates, current_sample_id=sample_id)

            if ref_info is None:
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

            ref_panel_token = safe_filename_token(ref_info["sample_id"])
            ref_crop_rel = os.path.join(
                "refs",
                dataset_id,
                f"page_{page_id}__side_{side}__char_{char_id}__from_{ref_panel_token}_{ref_target_size}.png",
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

            task_key = (ref_unit_key, char_id, ref_info["sample_id"])
            if task_key not in ref_tasks:
                ref_tasks[task_key] = {
                    "panel_line_path": ref_info["panel_line_path"],
                    "ref_bbox": ref_info["bbox"],
                    "ref_crop_rel": ref_crop_rel,
                }

        rec["ref_selected"] = ref_selected
        rec["has_valid_ref"] = has_valid_ref(rec)

    ref_task_list = list(ref_tasks.values())
    if num_workers <= 1:
        ref_results = list(map(partial(save_single_ref_crop, crop_root=crop_root, ref_target_size=ref_target_size), ref_task_list))
    else:
        with ThreadPoolExecutor(max_workers=num_workers) as ex:
            ref_results = list(ex.map(partial(save_single_ref_crop, crop_root=crop_root, ref_target_size=ref_target_size), ref_task_list))

    total_ref_crops = sum(1 for x in ref_results if x)

    train_records, test_records, split_info = split_dataset(
        panel_records=panel_records,
        test_limit=test_ref_limit,
        seed=seed,
        split_unit=split_unit,
        test_bucket_ratio=test_bucket_ratio,
    )

    out_jsonl = os.path.join(output_root, output_name)
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for rec in panel_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    with open(os.path.join(output_root, "splits", "train.jsonl"), "w", encoding="utf-8") as f:
        for rec in train_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    with open(os.path.join(output_root, "splits", "test.jsonl"), "w", encoding="utf-8") as f:
        for rec in test_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    split_summary = {
        "total_panels": len(panel_records),
        "total_ref_crops": total_ref_crops,
        "total_ref_units": len(records_per_ref_unit),
        "ref_units_with_bank": len({k[0] for k in ref_candidate_bank.keys()}),
        "panels_with_any_character": sum(1 for r in panel_records if len(r.get("characters", [])) > 0),
        "panels_with_valid_ref": sum(1 for r in panel_records if r.get("has_valid_ref", False)),
        "overall_bucket_counts": count_bucket_dict(panel_records),
        "train_bucket_counts": count_bucket_dict(train_records),
        "test_bucket_counts": count_bucket_dict(test_records),
        "enough_ref_panels_for_target": sum(1 for r in panel_records if r.get("has_valid_ref", False)) >= test_ref_limit,
        "split_info": split_info,
    }

    manifest = {
        "release_jsonl": release_jsonl,
        "crop_root": crop_root,
        "output_root": output_root,
        "output_jsonl": output_name,
        "train_jsonl": "splits/train.jsonl",
        "test_jsonl": "splits/test.jsonl",
        "ref_target_size": ref_target_size,
        "max_refs_per_panel": max_refs_per_panel,
        "test_ref_limit": test_ref_limit,
        "split_unit": split_unit,
        "test_bucket_ratio": test_bucket_ratio,
        "seed": seed,
        "num_workers": num_workers,
        "ref_identity_scope": "side_page",
        "ref_selection_strategy": "best_non_self_candidate_with_fallback",
        "notes": [
            "Built only from panel_index_release_en.jsonl.",
            "Uses side-level reference pools: dataset_id/page_id/panel_side.",
            "Pre-crops ref images from panel_line_path using character panel-local bbox.",
            "test.jsonl keeps all selected test panels, including those without valid refs.",
            "For side_page split, test bucket ratio is matched approximately at panel level while preserving unit integrity.",
            "Output rows are compatible with current panel_dataset_tr.py / train.py.",
        ],
        "split_summary": split_summary,
    }
    with open(os.path.join(output_root, "manifest_train_ref.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"[DONE] wrote {len(panel_records)} records to: {out_jsonl}")
    print(f"[DONE] wrote train split to: {os.path.join(output_root, 'splits', 'train.jsonl')}")
    print(f"[DONE] wrote test split to: {os.path.join(output_root, 'splits', 'test.jsonl')}")
    print(f"[DONE] wrote manifest to: {os.path.join(output_root, 'manifest_train_ref.json')}")



def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release_jsonl", type=str, default="/data4/Sketch/SketchRet/TuringSketchLine/panel_index_en.jsonl")
    parser.add_argument("--crop_root", type=str, default="/data4/Sketch/SketchRet/TuringSketchLine")
    parser.add_argument("--output_root", type=str, default="/data4/Sketch/SketchRet/TuringSketchLine")
    parser.add_argument("--output_name", type=str, default="panel_index_train_ref.jsonl")
    parser.add_argument("--ref_target_size", type=int, default=224)
    parser.add_argument("--max_refs_per_panel", type=int, default=3)
    parser.add_argument("--test_ref_limit", type=int, default=1000, help="Target number of panels in test.jsonl.")
    parser.add_argument("--split_unit", type=str, choices=["side_page", "panel"], default="side_page")
    parser.add_argument("--test_bucket_ratio", type=str, default="1:5:4", help="Bucket ratio for no_char:single_char:multi_char in test split.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=8)
    args = parser.parse_args()

    preprocess_release_for_training(
        release_jsonl=args.release_jsonl,
        crop_root=args.crop_root,
        output_root=args.output_root,
        output_name=args.output_name,
        ref_target_size=args.ref_target_size,
        max_refs_per_panel=args.max_refs_per_panel,
        test_ref_limit=args.test_ref_limit,
        split_unit=args.split_unit,
        test_bucket_ratio=args.test_bucket_ratio,
        seed=args.seed,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()

