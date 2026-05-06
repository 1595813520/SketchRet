
from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch
from torch.utils.data import Dataset, Sampler

from utils.image_ops import (
    preprocess_line_target,
    preprocess_ref_image,
    preprocess_sketch_image,
)


# Prefer panel-level paths over page/global image paths.
TARGET_ALIASES = [
    'panel_line_path',
    'line_path',
    'target_path',
    'lineart_path',
    'final_path',
    'gt_path',
    'pixel_path',
    'image_path',
]
SKETCH_ALIASES = [
    'panel_sketch_path',
    'sketch_path',
    'draft_path',
    'rough_path',
    'condition_path',
    'scribble_path',
]
# Prefer story-aware text on the new dataset; fall back to short captions/prompts.
CAPTION_ALIASES = ['story', 'caption', 'text', 'prompt', 'description']
REF_LIST_ALIASES = [
    'ref_selected',
    'refs',
    'references',
    'reference_images',
    'ref_images',
]
REF_PATH_ALIASES = [
    'ref_crop_path',
    'path',
    'image_path',
    'ref_path',
    'reference_path',
    'line_path',
]
REF_BBOX_ALIASES = [
    'char_bbox_norm',
    'bbox_norm',
    'bbox',
    'box',
    'ref_bbox',
    'role_bbox',
]
REF_VALID_ALIASES = ['ref_valid', 'valid', 'is_valid']
ID_ALIASES = ['sample_id', 'id', 'name', 'uid']
# For the new Turing dataset, prefer grouping by side-page / page key.
GROUP_ALIASES = [
    'ref_unit_key',
    'side_page_key',
    'page_key',
    'spread_key',
    'spread_id',
    'group_id',
    'episode_id',
]
PANEL_GEOM_ALIASES = ['panel_geom', 'panel_geoms', 'geom']


def _first_existing(data: Dict[str, Any], aliases: Sequence[str], default=None):
    for key in aliases:
        if key in data and data[key] is not None:
            return data[key]
    return default


@dataclass
class PanelExample:
    sample_id: str
    target_path: str
    sketch_path: str
    caption: str
    refs: List[Dict[str, Any]]
    group_id: Optional[str] = None
    panel_geom: Optional[Dict[str, Any]] = None


class MangaPanelIndexDataset(Dataset):
    def __init__(
        self,
        crop_root: str,
        index_file: str,
        strict_exist_check: bool = False,
    ):
        super().__init__()
        self.crop_root = Path(crop_root)
        self.index_file = Path(index_file)
        self.strict_exist_check = strict_exist_check
        self.examples: List[PanelExample] = []
        self._num_examples_with_ref = 0
        self._group_sizes: Dict[str, int] = {}
        self._load_index()

    def _resolve_path(self, path_value: str) -> str:
        path = Path(path_value)
        if path.is_absolute():
            return str(path)
        return str((self.crop_root / path).resolve())

    def _extract_caption(self, row: Dict[str, Any]) -> str:
        story = row.get('story', '')
        caption = _first_existing(row, CAPTION_ALIASES, default='')
        story = '' if story is None else str(story).strip()
        caption = '' if caption is None else str(caption).strip()

        if story and caption and story != caption:
            return f'{story} {caption}'
        return story or caption or ''

    def _extract_refs(self, row: Dict[str, Any]) -> List[Dict[str, Any]]:
        refs = _first_existing(row, REF_LIST_ALIASES, default=[])
        if isinstance(refs, dict):
            refs = refs.get('items', [])
        if not refs:
            maybe_single = _first_existing(row, ['ref_crop_path', 'ref_path', 'reference_path', 'reference_image_path'])
            if maybe_single:
                refs = [{
                    'ref_crop_path': maybe_single,
                    'char_bbox_norm': _first_existing(row, REF_BBOX_ALIASES, [0.0, 0.0, 1.0, 1.0]),
                    'ref_valid': 1,
                }]

        out: List[Dict[str, Any]] = []
        for item in refs:
            if isinstance(item, str):
                path = item
                bbox = [0.0, 0.0, 1.0, 1.0]
                valid = True
            elif isinstance(item, dict):
                path = _first_existing(item, REF_PATH_ALIASES)
                bbox = _first_existing(item, REF_BBOX_ALIASES, default=[0.0, 0.0, 1.0, 1.0])
                valid_value = _first_existing(item, REF_VALID_ALIASES, default=1)
                valid = bool(valid_value)
            else:
                continue

            if not path:
                continue

            resolved_path = self._resolve_path(str(path))
            out.append({'path': resolved_path, 'bbox': bbox, 'valid': valid})
        return out

    def _load_index(self) -> None:
        with self.index_file.open('r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)

                target = _first_existing(row, TARGET_ALIASES)
                sketch = _first_existing(row, SKETCH_ALIASES)
                if target is None or sketch is None:
                    if self.strict_exist_check:
                        raise KeyError(
                            'Each row must contain one panel-level target image path and one panel-level sketch path.'
                        )
                    continue

                sample_id = str(_first_existing(row, ID_ALIASES, default=Path(str(target)).stem))
                caption = self._extract_caption(row)
                group_id = _first_existing(row, GROUP_ALIASES, default=None)
                panel_geom = _first_existing(row, PANEL_GEOM_ALIASES, default=None)
                refs = self._extract_refs(row)

                example = PanelExample(
                    sample_id=sample_id,
                    target_path=self._resolve_path(str(target)),
                    sketch_path=self._resolve_path(str(sketch)),
                    caption=caption,
                    refs=refs,
                    group_id=None if group_id is None else str(group_id),
                    panel_geom=panel_geom,
                )

                if self.strict_exist_check:
                    if not os.path.exists(example.target_path):
                        raise FileNotFoundError(example.target_path)
                    if not os.path.exists(example.sketch_path):
                        raise FileNotFoundError(example.sketch_path)
                    for ref in example.refs:
                        if ref['valid'] and not os.path.exists(ref['path']):
                            raise FileNotFoundError(ref['path'])

                if any(ref.get('valid', False) for ref in example.refs):
                    self._num_examples_with_ref += 1
                if example.group_id is not None:
                    self._group_sizes[example.group_id] = self._group_sizes.get(example.group_id, 0) + 1

                self.examples.append(example)

    @property
    def num_examples_with_ref(self) -> int:
        return self._num_examples_with_ref

    @property
    def num_groups(self) -> int:
        return len(self._group_sizes)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        ex = self.examples[index]
        return {
            'sample_id': ex.sample_id,
            'target_path': ex.target_path,
            'sketch_path': ex.sketch_path,
            'caption': ex.caption,
            'refs': ex.refs,
            # Keep both keys for backward compatibility with existing training code.
            'group_id': ex.group_id,
            'spread_id': ex.group_id,
            'panel_geom': ex.panel_geom,
        }


class SpreadGroupedSampler(Sampler[int]):
    """Group samples by side-page/page/spread id when available; otherwise random order."""

    def __init__(self, dataset: MangaPanelIndexDataset, shuffle_spreads: bool = True, shuffle_within_spread: bool = False, seed: int = 42):
        self.dataset = dataset
        self.shuffle_spreads = shuffle_spreads
        self.shuffle_within_spread = shuffle_within_spread
        self.seed = seed
        self.groups = self._build_groups()

    def _build_groups(self) -> List[List[int]]:
        groups: Dict[str, List[int]] = {}
        ungrouped: List[List[int]] = []
        for idx, ex in enumerate(self.dataset.examples):
            if ex.group_id is None:
                ungrouped.append([idx])
            else:
                groups.setdefault(ex.group_id, []).append(idx)
        return list(groups.values()) + ungrouped

    def __iter__(self):
        rng = random.Random(self.seed)
        groups = [g[:] for g in self.groups]
        if self.shuffle_spreads:
            rng.shuffle(groups)
        for g in groups:
            if self.shuffle_within_spread:
                rng.shuffle(g)
            for idx in g:
                yield idx

    def __len__(self) -> int:
        return len(self.dataset)


class MangaPanelIndexCollator:
    def __init__(
        self,
        tokenizer,
        resolution: int = 384,
        caption_dropout_prob: float = 0.0,
        sketch_dropout_prob: float = 0.0,
        max_refs_per_panel: int = 3,
        ref_resolution: int = 224,
        max_length: int = 77,
        fixed_prompt: Optional[str] = None,
    ):
        self.tokenizer = tokenizer
        self.resolution = resolution
        self.caption_dropout_prob = caption_dropout_prob
        self.sketch_dropout_prob = sketch_dropout_prob
        self.max_refs_per_panel = max_refs_per_panel
        self.ref_resolution = ref_resolution
        self.max_length = max_length
        self.fixed_prompt = fixed_prompt

    @staticmethod
    def _maybe_drop_caption(caption: str, p: float) -> str:
        if p <= 0.0:
            return caption
        return '' if random.random() < p else caption

    @staticmethod
    def _make_bbox_tensor(box: Any) -> torch.Tensor:
        if box is None:
            return torch.tensor([0.0, 0.0, 1.0, 1.0], dtype=torch.float32)
        x = torch.tensor(box, dtype=torch.float32).flatten()
        if x.numel() != 4:
            return torch.tensor([0.0, 0.0, 1.0, 1.0], dtype=torch.float32)
        x[0::2] = x[0::2].clamp(0.0, 1.0)
        x[1::2] = x[1::2].clamp(0.0, 1.0)
        x[2] = max(float(x[2]), float(x[0]) + 1e-4)
        x[3] = max(float(x[3]), float(x[1]) + 1e-4)
        return x

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        sample_ids = [item['sample_id'] for item in batch]
        if self.fixed_prompt is not None:
            captions = [self.fixed_prompt for _ in batch]
        else:
            captions = [self._maybe_drop_caption(item.get('caption', ''), self.caption_dropout_prob) for item in batch]
        tokenized = self.tokenizer(
            captions,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt',
        )

        pixel_values = []
        sketch_values = []
        valid_masks = []
        panel_geoms = []
        ref_pixel_values = []
        ref_bboxes = []
        ref_valid_mask = []

        for item in batch:
            target_tensor, target_meta = preprocess_line_target(item['target_path'], self.resolution)
            sketch_tensor, sketch_meta, valid_mask = preprocess_sketch_image(item['sketch_path'], self.resolution)
            if self.sketch_dropout_prob > 0.0 and random.random() < self.sketch_dropout_prob:
                sketch_tensor = torch.ones_like(sketch_tensor)

            refs = item.get('refs', [])[: self.max_refs_per_panel]
            ref_imgs = []
            ref_boxes = []
            ref_valid = []
            for ref in refs:
                ref_imgs.append(preprocess_ref_image(ref['path'], self.ref_resolution))
                ref_boxes.append(self._make_bbox_tensor(ref.get('bbox')))
                ref_valid.append(1.0 if ref.get('valid', True) else 0.0)
            while len(ref_imgs) < self.max_refs_per_panel:
                ref_imgs.append(torch.zeros(3, self.ref_resolution, self.ref_resolution, dtype=torch.float32))
                ref_boxes.append(torch.tensor([0.0, 0.0, 1.0, 1.0], dtype=torch.float32))
                ref_valid.append(0.0)

            pixel_values.append(target_tensor)
            sketch_values.append(sketch_tensor)
            valid_masks.append(valid_mask)
            panel_geoms.append(item.get('panel_geom', target_meta))
            ref_pixel_values.append(torch.stack(ref_imgs, dim=0))
            ref_bboxes.append(torch.stack(ref_boxes, dim=0))
            ref_valid_mask.append(torch.tensor(ref_valid, dtype=torch.float32))

        return {
            'sample_ids': sample_ids,
            'input_ids': tokenized.input_ids,
            'attention_mask': tokenized.attention_mask,
            'pixel_values': torch.stack(pixel_values, dim=0),
            'sketch_values': torch.stack(sketch_values, dim=0),
            'valid_mask': torch.stack(valid_masks, dim=0),
            'panel_geoms': panel_geoms,
            'ref_pixel_values': torch.stack(ref_pixel_values, dim=0),
            'ref_bboxes': torch.stack(ref_bboxes, dim=0),
            'ref_valid_mask': torch.stack(ref_valid_mask, dim=0),
        }
