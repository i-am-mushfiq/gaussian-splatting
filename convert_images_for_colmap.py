# convert_images_for_colmap.py
import os
import cv2
import numpy as np
from pathlib import Path

INPUT = Path(r"Data\Brain_new\synth_views")   # folder with your generated images
OUTPUT = Path(r"Data\Brain_new\for_colmap")   # folder to create for COLMAP consumption
BAD = OUTPUT / "bad_files"
OUTPUT.mkdir(parents=True, exist_ok=True)
BAD.mkdir(parents=True, exist_ok=True)

def convert_file(src_path, dst_path):
    # load with unchanged flag to preserve bit depth
    img = cv2.imread(str(src_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        return False, "read_failed"
    # If float image, scale to 0-255
    if img.dtype == np.float32 or img.dtype == np.float64:
        img = np.clip(img, 0.0, 1.0)
        img = (img * 255.0).round().astype(np.uint8)

    # If 16-bit, convert to 8-bit by right shift (fast) — preserves contrast
    if img.dtype == np.uint16:
        # If values are already in 0..65535, reduce to 0..255 by >> 8
        img8 = (img >> 8).astype(np.uint8)
        img = img8

    # If single channel (H, W), convert to 3-channel BGR
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.ndim == 3 and img.shape[2] == 4:
        # If RGBA, drop alpha
        img = img[:, :, :3]

    # write as 8-bit PNG
    ok = cv2.imwrite(str(dst_path), img)
    if not ok:
        return False, "write_failed"

    # verify read-back
    chk = cv2.imread(str(dst_path), cv2.IMREAD_UNCHANGED)
    if chk is None:
        return False, "verify_read_failed"
    return True, "ok"

cnt = 0
bad_list = []
for p in sorted(INPUT.iterdir()):
    if p.suffix.lower() not in (".png", ".jpg", ".jpeg", ".tif", ".tiff"):
        continue
    dst = OUTPUT / p.name
    success, reason = convert_file(p, dst)
    if not success:
        print(f"[BAD] {p.name} -> {reason}; moving to bad folder")
        p.rename(BAD / p.name)
        bad_list.append((p.name, reason))
    else:
        cnt += 1
        if cnt % 100 == 0:
            print(f"[INFO] converted {cnt} images...")

print(f"[DONE] Converted {cnt} images. {len(bad_list)} bad files (moved to {BAD}).")
if bad_list:
    print("Bad files sample:", bad_list[:10])
