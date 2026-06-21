from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw


def parse_box(value: str) -> tuple[int, int, int, int]:
    parts = [int(part.strip()) for part in value.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("boxes must use left,top,right,bottom")
    left, top, right, bottom = parts
    if right <= left or bottom <= top:
        raise argparse.ArgumentTypeError("box right/bottom must be greater than left/top")
    return left, top, right, bottom


def main() -> None:
    parser = argparse.ArgumentParser(description="Create and preview an inpainting mask for LAAS image edits.")
    parser.add_argument("--image", required=True, type=Path, help="Source image used for preview sizing.")
    parser.add_argument("--mask", required=True, type=Path, help="Output grayscale PNG mask path.")
    parser.add_argument("--preview", type=Path, help="Optional red overlay preview PNG path.")
    parser.add_argument("--rect", action="append", type=parse_box, default=[], help="White rectangle: left,top,right,bottom.")
    parser.add_argument("--ellipse", action="append", type=parse_box, default=[], help="White ellipse: left,top,right,bottom.")
    args = parser.parse_args()

    base = Image.open(args.image).convert("RGBA")
    mask = Image.new("L", base.size, 0)
    draw = ImageDraw.Draw(mask)
    for box in args.rect:
        draw.rectangle(box, fill=255)
    for box in args.ellipse:
        draw.ellipse(box, fill=255)
    args.mask.parent.mkdir(parents=True, exist_ok=True)
    mask.save(args.mask)

    if args.preview:
        overlay = Image.new("RGBA", base.size, (255, 0, 0, 0))
        overlay.putalpha(mask.point(lambda pixel: 100 if pixel else 0))
        args.preview.parent.mkdir(parents=True, exist_ok=True)
        Image.alpha_composite(base, overlay).save(args.preview)


if __name__ == "__main__":
    main()
