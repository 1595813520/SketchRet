import os
import json
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Iterable, DefaultDict
from collections import defaultdict

import torch
from torch.utils.data import Dataset, Sampler
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
from PIL import Image


# =========================================================
# 基础工具
# =========================================================

def get_book_id(image_path: str) -> str:
    parts = image_path.replace("\\", "/").split("/")
    return parts[0] if len(parts) > 1 else "default_book"


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


def crop_pil(img: Image.Image, bbox_xyxy: List[int]) -> Image.Image:
    return img.crop(tuple(bbox_xyxy))


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


def norm_bbox_hflip(bbox_norm: List[float]) -> List[float]:
    """
    水平翻转后，bbox 归一化坐标更新
    [x0,y0,x1,y1] -> [1-x1,y0,1-x0,y1]
    """
    x0, y0, x1, y1 = bbox_norm
    return [1.0 - x1, y0, 1.0 - x0, y1]


def pil_to_line_tensor(img: Image.Image, resolution: int) -> torch.Tensor:
    """
    line_gt -> 给 VAE / SD 用，归一化到 [-1,1]
    """
    img = img.convert("RGB")
    img = TF.resize(img, [resolution, resolution], interpolation=InterpolationMode.BILINEAR)
    x = TF.to_tensor(img)
    x = (x - 0.5) / 0.5
    return x


def pil_to_sketch_tensor(img: Image.Image, resolution: int) -> torch.Tensor:
    """
    sketch -> 给 SketchEncoder，保持 [0,1] 单通道
    """
    img = img.convert("L")
    img = TF.resize(img, [resolution, resolution], interpolation=InterpolationMode.BILINEAR)
    x = TF.to_tensor(img)
    return x


def pil_to_ref_tensor_clip(img: Image.Image, resolution: int) -> torch.Tensor:
    """
    ref crop -> 给冻结的 CLIP image encoder
    """
    img = img.convert("RGB")
    img = TF.resize(img, [resolution, resolution], interpolation=InterpolationMode.BICUBIC)
    x = TF.to_tensor(img)
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)
    x = (x - mean) / std
    return x


# =========================================================
# 元信息
# =========================================================

@dataclass
class PanelMeta:
    sample_id: str
    book_id: str
    spread_key: str       # image_path（整张双页拼接图）
    image_path: str
    frame_index: int
    panel_global_bbox: List[int]
    panel_order_in_spread: int



# =========================================================
# 图像 -> Tensor
# =========================================================

def pil_line_to_tensor_384(img: Image.Image) -> torch.Tensor:
    """
    line_gt for SD / VAE
    预处理阶段已经是 384x384，这里不再 resize
    输出范围 [-1, 1]
    """
    img = img.convert("RGB")
    if img.size != (384, 384):
        raise ValueError(f"line image must be 384x384, got {img.size}")
    x = TF.to_tensor(img)
    x = (x - 0.5) / 0.5
    return x


def pil_sketch_to_tensor_384(img: Image.Image) -> torch.Tensor:
    """
    sketch for SketchEncoder
    预处理阶段已经是 384x384，这里不再 resize
    输出 [0, 1] 单通道
    """
    img = img.convert("L")
    if img.size != (384, 384):
        raise ValueError(f"sketch image must be 384x384, got {img.size}")
    x = TF.to_tensor(img)  # [1,384,384], [0,1]
    return x


def pil_ref_to_clip_tensor_224(img: Image.Image) -> torch.Tensor:
    """
    ref crop for frozen CLIP image encoder
    预处理阶段已经是 224x224，这里不再 resize
    只做 CLIP normalization
    """
    img = img.convert("RGB")
    if img.size != (224, 224):
        raise ValueError(f"ref image must be 224x224, got {img.size}")
    x = TF.to_tensor(img)
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)
    x = (x - mean) / std
    return x


# =========================================================
# Dataset
# =========================================================

class MangaPanelIndexDataset(Dataset):
    """
    对接离线预处理后的 panel_index.jsonl

    每条记录至少包含：
      - sample_id
      - book_id
      - spread_key
      - image_path
      - frame_index
      - panel_order_in_spread
      - panel_global_bbox
      - panel_line_path
      - panel_sketch_path
      - caption
      - characters_all
      - ref_selected
    """

    def __init__(
        self,
        crop_root: str,
        index_file: str = "panel_index.jsonl",
        train_stage: str = "main",   # "main" / "ref"
        load_ref_images: bool = True,
        strict_exist_check: bool = True,
    ):
        super().__init__()
        assert train_stage in ["main", "ref"]
        self.crop_root = crop_root
        self.index_path = os.path.join(crop_root, index_file)
        self.train_stage = train_stage
        self.load_ref_images = load_ref_images
        self.strict_exist_check = strict_exist_check

        self.records: List[Dict[str, Any]] = []
        self._load_index()

    # -------------------------
    # 内部函数
    # -------------------------
    def _abs_path(self, rel_path: str) -> str:
        return os.path.join(self.crop_root, rel_path)

    def _load_jsonl(self, path: str) -> List[Dict[str, Any]]:
        items = []
        with open(path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    items.append(json.loads(line))
                except Exception as e:
                    raise ValueError(f"Invalid json at line {line_idx + 1} in {path}: {e}")
        return items

    def _check_record_paths(self, rec: Dict[str, Any]) -> bool:
        line_ok = os.path.exists(self._abs_path(rec["panel_line_path"]))
        sketch_ok = os.path.exists(self._abs_path(rec["panel_sketch_path"]))
        if not (line_ok and sketch_ok):
            return False

        if self.train_stage == "ref":
            for item in rec.get("ref_selected", []):
                if int(item.get("ref_valid", 0)) == 1:
                    ref_rel = item.get("ref_crop_path", "")
                    if ref_rel == "" or not os.path.exists(self._abs_path(ref_rel)):
                        return False
        return True

    def _load_index(self):
        if not os.path.exists(self.index_path):
            raise FileNotFoundError(f"panel index not found: {self.index_path}")

        raw_records = self._load_jsonl(self.index_path)

        kept = []
        skipped = 0
        for rec in raw_records:
            if self.strict_exist_check:
                ok = self._check_record_paths(rec)
                if not ok:
                    skipped += 1
                    continue
            kept.append(rec)

        self.records = kept
        print(
            f"[MangaPanelIndexDataset] loaded {len(self.records)} records "
            f"from {self.index_path}, skipped={skipped}"
        )

    # -------------------------
    # dataset api
    # -------------------------
    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec = self.records[idx]

        line_img = Image.open(self._abs_path(rec["panel_line_path"])).convert("RGB")
        sketch_img = Image.open(self._abs_path(rec["panel_sketch_path"])).convert("RGB")

        out: Dict[str, Any] = {
            "sample_id": rec["sample_id"],
            "book_id": rec["book_id"],
            "spread_key": rec["spread_key"],
            "image_path": rec["image_path"],
            "frame_index": rec["frame_index"],
            "panel_order_in_spread": rec["panel_order_in_spread"],
            "panel_global_bbox": torch.tensor(rec["panel_global_bbox"], dtype=torch.long),

            "caption": rec["caption"],
            "pixel_values": pil_line_to_tensor_384(line_img),         # [3,384,384], [-1,1]
            "sketch_values": pil_sketch_to_tensor_384(sketch_img),   # [1,384,384], [0,1]

            # 保留 metadata 方便调试或后续扩展
            "characters_all": rec.get("characters_all", []),
            "ref_selected_meta": rec.get("ref_selected", []),

            # 一个很轻的 panel metadata
            "panel_meta": torch.tensor(
                [float(rec["panel_order_in_spread"])],
                dtype=torch.float,
            ),
        }

        if self.train_stage == "ref":
            ref_pixel_values = []
            ref_bboxes = []
            ref_valid_mask = []
            ref_char_ids = []
            ref_panel_ids = []

            for item in rec.get("ref_selected", []):
                ref_valid = int(item.get("ref_valid", 0))
                ref_valid_mask.append(float(ref_valid))
                ref_bboxes.append(torch.tensor(item["char_bbox_norm"], dtype=torch.float))
                ref_char_ids.append(item.get("char_id", ""))
                ref_panel_ids.append(item.get("ref_panel_id", ""))

                if ref_valid == 1 and self.load_ref_images:
                    ref_img = Image.open(self._abs_path(item["ref_crop_path"])).convert("RGB")
                    ref_pixel_values.append(pil_ref_to_clip_tensor_224(ref_img))
                else:
                    ref_pixel_values.append(torch.zeros(3, 224, 224, dtype=torch.float))

            out["ref_pixel_values"] = ref_pixel_values     # list[Tensor(3,224,224)]
            out["ref_bboxes"] = ref_bboxes                 # list[Tensor(4)]
            out["ref_valid_mask"] = ref_valid_mask         # list[float]
            out["ref_char_ids"] = ref_char_ids             # list[str]
            out["ref_panel_ids"] = ref_panel_ids           # list[str]

        return out


class SameSpreadBatchSampler(Sampler[List[int]]):
    """
    尽量让同一个 spread 的 panel 进入同一个 batch
    训练单元仍然是 panel
    """

    def __init__(
        self,
        dataset: MangaPanelIndexDataset,
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = False,
        seed: int = 42,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.rng = random.Random(seed)

        self.spread_to_indices: DefaultDict[str, List[int]] = defaultdict(list)
        for idx, rec in enumerate(dataset.records):
            self.spread_to_indices[rec["spread_key"]].append(idx)

        self.spread_keys = list(self.spread_to_indices.keys())

    def __iter__(self) -> Iterable[List[int]]:
        spread_keys = list(self.spread_keys)
        if self.shuffle:
            self.rng.shuffle(spread_keys)

        all_batches = []
        for spread_key in spread_keys:
            indices = list(self.spread_to_indices[spread_key])

            if self.shuffle:
                self.rng.shuffle(indices)

            for i in range(0, len(indices), self.batch_size):
                batch = indices[i:i + self.batch_size]
                if len(batch) < self.batch_size and self.drop_last:
                    continue
                all_batches.append(batch)

        if self.shuffle:
            self.rng.shuffle(all_batches)

        for batch in all_batches:
            yield batch

    def __len__(self) -> int:
        total = 0
        for _, indices in self.spread_to_indices.items():
            if self.drop_last:
                total += len(indices) // self.batch_size
            else:
                total += (len(indices) + self.batch_size - 1) // self.batch_size
        return total

# =========================================================
# 同 spread batch sampler
# =========================================================
class MangaPanelIndexCollator:
    """
    对接离线预处理版 panel_index.jsonl

    负责：
    - caption dropout
    - sketch dropout mask
    - ref padding（只在 ref stage）
    """

    def __init__(
        self,
        tokenizer,
        train_stage: str = "main",      # "main" / "ref"
        caption_dropout_prob: float = 0.25,
        sketch_dropout_prob: float = 0.05,
        max_refs_per_panel: int = 3,
    ):
        assert train_stage in ["main", "ref"]
        self.tokenizer = tokenizer
        self.train_stage = train_stage
        self.caption_dropout_prob = caption_dropout_prob
        self.sketch_dropout_prob = sketch_dropout_prob
        self.max_refs_per_panel = max_refs_per_panel

    def _tokenize_with_dropout(self, captions: List[str]) -> torch.Tensor:
        dropped = []
        for cap in captions:
            if random.random() < self.caption_dropout_prob:
                cap = ""
            dropped.append(cap)

        tokens = self.tokenizer(
            dropped,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )
        return tokens.input_ids

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        B = len(examples)

        pixel_values = torch.stack([ex["pixel_values"] for ex in examples], dim=0)   # [B,3,384,384]
        sketch_values = torch.stack([ex["sketch_values"] for ex in examples], dim=0) # [B,1,384,384]

        captions = [ex["caption"] for ex in examples]
        input_ids = self._tokenize_with_dropout(captions)

        batch: Dict[str, Any] = {
            "sample_ids": [ex["sample_id"] for ex in examples],
            "book_ids": [ex["book_id"] for ex in examples],
            "spread_keys": [ex["spread_key"] for ex in examples],
            "image_paths": [ex["image_path"] for ex in examples],

            "frame_indices": torch.tensor([ex["frame_index"] for ex in examples], dtype=torch.long),
            "panel_order_in_spread": torch.tensor([ex["panel_order_in_spread"] for ex in examples], dtype=torch.long),
            "panel_global_bboxes": torch.stack([ex["panel_global_bbox"] for ex in examples], dim=0),
            "panel_meta": torch.stack([ex["panel_meta"] for ex in examples], dim=0),

            "pixel_values": pixel_values,
            "sketch_values": sketch_values,
            "input_ids": input_ids,

            # metadata
            "characters_all": [ex["characters_all"] for ex in examples],
        }

        # sketch 低概率 dropout
        batch["sketch_keep_mask"] = (torch.rand(B) > self.sketch_dropout_prob).float()

        if self.train_stage == "ref":
            max_n_ref = self.max_refs_per_panel
            ref_shape = (3, 224, 224)

            ref_pixel_values = []
            ref_bboxes = []
            ref_valid_mask = []
            ref_char_ids = []
            ref_panel_ids = []

            for ex in examples:
                cur_imgs = ex["ref_pixel_values"]
                cur_boxes = ex["ref_bboxes"]
                cur_valid = ex["ref_valid_mask"]
                cur_ids = ex["ref_char_ids"]
                cur_ref_panel_ids = ex["ref_panel_ids"]

                n = min(len(cur_imgs), max_n_ref)

                pad_imgs = []
                pad_boxes = []
                pad_valid = []
                pad_ids = []
                pad_ref_panel_ids = []

                for i in range(max_n_ref):
                    if i < n:
                        pad_imgs.append(cur_imgs[i])
                        pad_boxes.append(cur_boxes[i])
                        pad_valid.append(cur_valid[i])
                        pad_ids.append(cur_ids[i])
                        pad_ref_panel_ids.append(cur_ref_panel_ids[i])
                    else:
                        pad_imgs.append(torch.zeros(ref_shape, dtype=torch.float))
                        pad_boxes.append(torch.zeros(4, dtype=torch.float))
                        pad_valid.append(0.0)
                        pad_ids.append("")
                        pad_ref_panel_ids.append("")

                ref_pixel_values.append(torch.stack(pad_imgs, dim=0))         # [N_ref,3,224,224]
                ref_bboxes.append(torch.stack(pad_boxes, dim=0))              # [N_ref,4]
                ref_valid_mask.append(torch.tensor(pad_valid, dtype=torch.float))
                ref_char_ids.append(pad_ids)
                ref_panel_ids.append(pad_ref_panel_ids)

            batch["ref_pixel_values"] = torch.stack(ref_pixel_values, dim=0)   # [B,N_ref,3,224,224]
            batch["ref_bboxes"] = torch.stack(ref_bboxes, dim=0)               # [B,N_ref,4]
            batch["ref_valid_mask"] = torch.stack(ref_valid_mask, dim=0)       # [B,N_ref]
            batch["ref_char_ids"] = ref_char_ids
            batch["ref_panel_ids"] = ref_panel_ids

        return batch