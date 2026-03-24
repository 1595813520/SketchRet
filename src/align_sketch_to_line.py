import os
import json
import argparse
from typing import Optional, List, Tuple

from PIL import Image

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable=None, *args, **kwargs):
        return iterable


SUPPORTED_EXTS = [
    ".png", ".PNG",
    ".jpg", ".JPG",
    ".jpeg", ".JPEG",
    ".webp", ".WEBP",
    ".bmp", ".BMP",
]


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def resolve_image_path(root: str, image_path: str, preferred_exts: Optional[List[str]] = None) -> Optional[str]:
    """
    根据 anno 里的 image_path（可能是 jpg）去 root 下找真实文件（通常是 png）。
    """
    image_path = image_path.replace("\\", "/")
    rel_no_ext, original_ext = os.path.splitext(image_path)

    candidate_exts = []
    seen = set()

    if preferred_exts is not None:
        for ext in preferred_exts:
            if ext not in seen:
                candidate_exts.append(ext)
                seen.add(ext)

    if original_ext and original_ext not in seen:
        candidate_exts.append(original_ext)
        seen.add(original_ext)

    for ext in SUPPORTED_EXTS:
        if ext not in seen:
            candidate_exts.append(ext)
            seen.add(ext)

    for ext in candidate_exts:
        p = os.path.join(root, rel_no_ext + ext)
        if os.path.exists(p):
            return p
    return None


def get_rel_path_with_actual_ext(root: str, abs_path: str) -> str:
    return os.path.relpath(abs_path, root).replace("\\", "/")


def align_sketch_to_line_size(
    sketch_img: Image.Image,
    target_w: int,
    target_h: int,
    fill_value: int = 255,
) -> Tuple[Image.Image, str]:
    """
    以左上角对齐，把 sketch 调整到和 line 完全同尺寸：
    - sketch 大了：裁掉右边/下边
    - sketch 小了：在右边/下边补白
    返回:
      aligned_img, mode_string
    """
    sw, sh = sketch_img.size

    # 先裁到不超过目标尺寸
    crop_w = min(sw, target_w)
    crop_h = min(sh, target_h)
    sketch_cropped = sketch_img.crop((0, 0, crop_w, crop_h))

    # 如果已经正好一样，直接返回
    if crop_w == target_w and crop_h == target_h:
        if sw == target_w and sh == target_h:
            return sketch_cropped, "same"
        return sketch_cropped, "crop_only"

    # 否则补到目标尺寸（右边/下边补）
    if sketch_cropped.mode == "L":
        canvas = Image.new("L", (target_w, target_h), color=fill_value)
    else:
        canvas = Image.new(sketch_cropped.mode, (target_w, target_h), color=(fill_value, fill_value, fill_value))

    canvas.paste(sketch_cropped, (0, 0))

    if sw >= target_w and sh >= target_h:
        mode = "crop_only"
    elif sw <= target_w and sh <= target_h:
        mode = "pad_only"
    else:
        mode = "crop_and_pad"

    return canvas, mode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ann_json",
        type=str,
        required=True,
        help="annotations.json 路径",
    )
    parser.add_argument(
        "--line_root",
        type=str,
        required=True,
        help="线稿根目录",
    )
    parser.add_argument(
        "--sketch_root",
        type=str,
        required=True,
        help="原草稿根目录",
    )
    parser.add_argument(
        "--aligned_sketch_root",
        type=str,
        required=True,
        help="对齐后的草稿输出根目录",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="是否覆盖已存在文件",
    )
    args = parser.parse_args()

    with open(args.ann_json, "r", encoding="utf-8") as f:
        anns = json.load(f)

    ensure_dir(args.aligned_sketch_root)

    total = 0
    missing_line = 0
    missing_sketch = 0
    open_failed = 0
    same_count = 0
    crop_only_count = 0
    pad_only_count = 0
    crop_and_pad_count = 0

    for entry in tqdm(anns, desc="Align sketch to line", unit="img"):
        total += 1
        image_path = entry["image_path"]

        line_abs = resolve_image_path(args.line_root, image_path, preferred_exts=[".png", ".PNG"])
        sketch_abs = resolve_image_path(args.sketch_root, image_path, preferred_exts=[".png", ".PNG"])

        if line_abs is None:
            missing_line += 1
            print(f"[WARN] missing line: {image_path}")
            continue
        if sketch_abs is None:
            missing_sketch += 1
            print(f"[WARN] missing sketch: {image_path}")
            continue

        rel_out = get_rel_path_with_actual_ext(args.sketch_root, sketch_abs)
        rel_out_no_ext, _ = os.path.splitext(rel_out)
        out_abs = os.path.join(args.aligned_sketch_root, rel_out_no_ext + ".png")

        if (not args.overwrite) and os.path.exists(out_abs):
            continue

        try:
            with Image.open(line_abs) as line_img:
                line_img = line_img.convert("L")
                target_w, target_h = line_img.size

            with Image.open(sketch_abs) as sketch_img:
                sketch_img = sketch_img.convert("L")
                aligned_img, mode = align_sketch_to_line_size(
                    sketch_img=sketch_img,
                    target_w=target_w,
                    target_h=target_h,
                    fill_value=255,
                )

            ensure_dir(os.path.dirname(out_abs))
            aligned_img.save(out_abs, format="PNG")

            if mode == "same":
                same_count += 1
            elif mode == "crop_only":
                crop_only_count += 1
            elif mode == "pad_only":
                pad_only_count += 1
            else:
                crop_and_pad_count += 1

        except Exception as e:
            open_failed += 1
            print(f"[WARN] failed: {image_path} | error={e}")

    summary = {
        "total": total,
        "missing_line": missing_line,
        "missing_sketch": missing_sketch,
        "open_failed": open_failed,
        "same_count": same_count,
        "crop_only_count": crop_only_count,
        "pad_only_count": pad_only_count,
        "crop_and_pad_count": crop_and_pad_count,
        "aligned_sketch_root": args.aligned_sketch_root,
    }

    print("========== Done ==========")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
    
'''
python /data/Sketch/src/align_sketch_to_line.py \
  --ann_json /data/DiffSensei-main/checkpoints/mangazero/annotations.json \
  --line_root /data/Sketch/manga_line/Anime2Sketch/anime_style \
  --sketch_root /data/Sketch/manga_line/Anime2Sketch/opensketch_style \
  --aligned_sketch_root /data/Sketch/manga_line/Anime2Sketch/opensketch_style_aligned

'''