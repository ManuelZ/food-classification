"""
Pre-resize images to target size.
Usage:
  python resize_images.py --images_dir /workspace/images/images --output_dir /workspace/opencv-pytorch-classification-project-2-resized/images/images --size 256
"""

import argparse
import glob
from multiprocessing import Pool, cpu_count
from pathlib import Path

import cv2


def resize_image(args):
    src, dst, size, quality = args
    img = cv2.imread(src)
    if img is None:
        return src, False
    h, w = img.shape[:2]
    if dst == src and h == size and w == size:
        return src, True
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    cv2.imwrite(dst, img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return src, True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_dir", required=True)
    parser.add_argument(
        "--output_dir", default=None, help="If omitted, resizes in-place"
    )
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--quality", type=int, default=95)
    parser.add_argument("--workers", type=int, default=cpu_count())
    args = parser.parse_args()

    src_dir = Path(args.images_dir)
    dst_dir = Path(args.output_dir) if args.output_dir else src_dir
    dst_dir.mkdir(parents=True, exist_ok=True)

    paths = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        paths += glob.glob(str(src_dir / ext))

    if not paths:
        print(f"No images found in {args.images_dir}")
        return

    print(
        f"Resizing {len(paths)} images to {args.size}x{args.size} using {args.workers} workers..."
    )
    if dst_dir != src_dir:
        print(f"Output: {dst_dir}")

    tasks = [(p, str(dst_dir / Path(p).name), args.size, args.quality) for p in paths]
    failed = []

    with Pool(args.workers) as pool:
        for i, (path, ok) in enumerate(pool.imap_unordered(resize_image, tasks), 1):
            if not ok:
                failed.append(path)
            if i % 500 == 0 or i == len(paths):
                print(f"  {i}/{len(paths)}")

    print(f"Done. {len(paths) - len(failed)} resized, {len(failed)} failed.")
    if failed:
        for p in failed:
            print(f"  FAILED: {p}")


if __name__ == "__main__":
    main()
