import os
from typing import Optional, List, Tuple
from PIL import Image

ANN_IMAGE_PATH = "naruto/764.jpg"
LINE_ROOT = "/data/Sketch/manga_line/Anime2Sketch/anime_style"
SKETCH_ROOT = "/data/Sketch/manga_line/Anime2Sketch/opensketch_style"


def resolve_image_path(root: str, image_path: str, preferred_exts: Optional[List[str]] = None) -> Optional[str]:
    """
    按同名 stem 搜索文件，优先使用 preferred_exts 指定的后缀。
    """
    image_path = image_path.replace("\\", "/")
    rel_no_ext, original_ext = os.path.splitext(image_path)

    all_exts = [
        ".png", ".PNG",
        ".jpg", ".JPG",
        ".jpeg", ".JPEG",
        ".webp", ".WEBP",
        ".bmp", ".BMP",
    ]

    candidate_exts = []
    seen_exts = set()

    if preferred_exts is not None:
        for ext in preferred_exts:
            if ext not in seen_exts:
                candidate_exts.append(ext)
                seen_exts.add(ext)

    if original_ext and original_ext not in seen_exts:
        candidate_exts.append(original_ext)
        seen_exts.add(original_ext)

    for ext in all_exts:
        if ext not in seen_exts:
            candidate_exts.append(ext)
            seen_exts.add(ext)

    print("=" * 80)
    print(f"[Resolver] root = {root}")
    print(f"[Resolver] image_path = {image_path}")
    print(f"[Resolver] rel_no_ext = {rel_no_ext}")
    print(f"[Resolver] candidate_exts = {candidate_exts}")

    for ext in candidate_exts:
        p = os.path.join(root, rel_no_ext + ext)
        exists = os.path.exists(p)
        print(f"  try: {p} | exists={exists}")
        if exists:
            print(f"  => HIT: {p}")
            return p

    print("  => HIT: None")
    return None


def list_all_candidates(root: str, image_path: str):
    image_path = image_path.replace("\\", "/")
    rel_no_ext, _ = os.path.splitext(image_path)

    all_exts = [
        ".png", ".PNG",
        ".jpg", ".JPG",
        ".jpeg", ".JPEG",
        ".webp", ".WEBP",
        ".bmp", ".BMP",
    ]

    print("=" * 80)
    print(f"[All Candidates] root = {root}")
    found_any = False
    for ext in all_exts:
        p = os.path.join(root, rel_no_ext + ext)
        if os.path.exists(p):
            found_any = True
            try:
                with Image.open(p) as im:
                    print(f"  {p} | size={im.size} | mode={im.mode} | format={im.format}")
            except Exception as e:
                print(f"  {p} | OPEN FAILED: {e}")
    if not found_any:
        print("  No candidates found.")


def debug_one_pair(image_path: str, line_root: str, sketch_root: str):
    print(f"\n[DEBUG] anno image_path = {image_path}")

    line_abs = resolve_image_path(line_root, image_path, preferred_exts=[".png", ".PNG"])
    sketch_abs = resolve_image_path(sketch_root, image_path, preferred_exts=[".png", ".PNG"])

    print("\n[Chosen Paths]")
    print("line_abs  =", line_abs)
    print("sketch_abs=", sketch_abs)

    print("\n[Candidate Scan]")
    list_all_candidates(line_root, image_path)
    list_all_candidates(sketch_root, image_path)

    if line_abs is None or sketch_abs is None:
        print("\n[RESULT] missing_pair")
        return

    print("\n[Open Chosen Files]")
    try:
        with Image.open(line_abs) as line_img:
            print(f"line  => path={line_abs}")
            print(f"         size={line_img.size}, mode={line_img.mode}, format={line_img.format}")
            line_size = line_img.size
    except Exception as e:
        print(f"line open failed: {e}")
        return

    try:
        with Image.open(sketch_abs) as sketch_img:
            print(f"sketch=> path={sketch_abs}")
            print(f"         size={sketch_img.size}, mode={sketch_img.mode}, format={sketch_img.format}")
            sketch_size = sketch_img.size
    except Exception as e:
        print(f"sketch open failed: {e}")
        return

    print("\n[Compare]")
    print("line_size  =", line_size)
    print("sketch_size=", sketch_size)
    print("same_size? =", line_size == sketch_size)

    if line_size != sketch_size:
        print("\n[RESULT] size_mismatch")
    else:
        print("\n[RESULT] sizes match, not a size_mismatch case")


if __name__ == "__main__":
    debug_one_pair(ANN_IMAGE_PATH, LINE_ROOT, SKETCH_ROOT)