#!/usr/bin/env python3
"""
GPU-accelerated orbital view generator (PyTorch grid_sample).
Generates MIP projections from rotated 3D volume and writes images to disk immediately.

Patch notes:
- Adds DEPTH_DOWNSAMPLE to reduce depth (D).
- Uses smaller BATCH_SIZE and DOWNSAMPLE_FACTOR defaults suitable for ~12GB GPUs.
- Adds try/except to fall back to per-sample processing when a batch causes OOM.
- Frees CUDA cache between batches.
- Writes images immediately (no large lists).
"""

import os
import math
import numpy as np
import SimpleITK as sitk
import cv2
from tqdm import tqdm

# Try to import torch; if not available, we'll fallback later
try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

# ---------------------------
# CONFIG (tweak these if needed)
# ---------------------------
INPUT_DIR = r"Data\Brain2_training\images"
OUTPUT_DIR = r"Data\Brain_new\synth_views"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Orbital grid parameters -> adjust to get desired count
num_azimuth = 40   # horizontal steps
num_elevation = 30 # vertical steps  -> 40 * 30 = 1200
azs = np.linspace(0.0, 360.0, num_azimuth, endpoint=False)
els = np.linspace(-60.0, 60.0, num_elevation)
ROTATION_LIST = [(float(el), float(az), 0.0) for el in els for az in azs]  # (rx, ry, rz) in degrees

# Memory / speed controls (safe defaults)
BATCH_SIZE = 1           # set 1 if you have limited GPU memory
DOWNSAMPLE_FACTOR = 3    # in-plane downsample factor (1=no downsample, 2=half, 3~=1/3)
DEPTH_DOWNSAMPLE = 2     # take every nth slice along depth (1=no subsample)

# Batch size for fallback single-sample processing (internal)
FALLBACK_BATCH = 1

# Save as 16-bit PNG
SAVE_16BIT = True

# ---------------------------
# HELPERS: IO & normalization
# ---------------------------
def load_volume_from_folder(input_dir):
    """Load DICOM series if possible, else load standard image files. Returns np.float32 volume (D,H,W)."""
    try:
        dicom_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(input_dir)
        if len(dicom_names) > 0:
            reader = sitk.ImageSeriesReader()
            reader.SetFileNames(dicom_names)
            sitk_vol = reader.Execute()
            print(f"[INFO] Loaded DICOM series ({len(dicom_names)} files)")
        else:
            raise RuntimeError("No DICOM series found")
    except Exception:
        # fallback to image files
        imgs = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(('.png','.tif','.tiff','.jpg','.jpeg'))])
        if len(imgs) == 0:
            raise FileNotFoundError("No images found in folder")
        slices = [sitk.ReadImage(os.path.join(input_dir, f), sitk.sitkFloat32) for f in imgs]
        sitk_vol = sitk.JoinSeries(slices)
        print(f"[INFO] Loaded {len(imgs)} image slices from folder")

    # Convert: SimpleITK -> numpy (z,y,x)
    vol = sitk.GetArrayFromImage(sitk_vol).astype(np.float32)
    spacing = sitk_vol.GetSpacing() if hasattr(sitk_vol, 'GetSpacing') else (1.0,1.0,1.0)
    print(f"[INFO] Volume shape (D,H,W): {vol.shape}, spacing: {spacing}")
    return vol, spacing

def normalize_and_clahe(volume_np, clip_percentiles=(1,99), clahe_clip=2.0, grid=(8,8)):
    """Clip percentiles, normalize to [0,1], apply CLAHE slice-wise. Returns float32 volume (D,H,W)."""
    v = volume_np.copy()
    p0, p100 = np.percentile(v, clip_percentiles)
    v = np.clip(v, p0, p100)
    vmin, vmax = float(v.min()), float(v.max())
    if vmax - vmin < 1e-6:
        v_norm = np.zeros_like(v, dtype=np.float32)
    else:
        v_norm = (v - vmin) / (vmax - vmin)
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=grid)
    D = v_norm.shape[0]
    out = np.zeros_like(v_norm, dtype=np.float32)
    for z in range(D):
        s8 = np.round(v_norm[z] * 255.0).astype(np.uint8)
        s_cl = clahe.apply(s8)
        out[z] = s_cl.astype(np.float32) / 255.0
    return out

# ---------------------------
# GPU RESAMPLING via PyTorch grid_sample
# ---------------------------
def euler_angles_to_rotation_matrix(rx_deg, ry_deg, rz_deg):
    """Return 3x3 rotation matrix from Euler angles in degrees.
       Angles: rx (pitch around X), ry (yaw around Y), rz (roll around Z).
       Rotation order: first Rx, then Ry, then Rz (so R = Rz @ Ry @ Rx).
    """
    rx = math.radians(rx_deg)
    ry = math.radians(ry_deg)
    rz = math.radians(rz_deg)
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)

    Rx = np.array([[1,0,0],[0,cx,-sx],[0,sx,cx]], dtype=np.float64)
    Ry = np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]], dtype=np.float64)
    Rz = np.array([[cz,-sz,0],[sz,cz,0],[0,0,1]], dtype=np.float64)

    R = Rz @ Ry @ Rx
    return R.astype(np.float32)

def make_affine_norm_from_R(R, D, H, W):
    """Build 3x4 affine in normalized coordinates for affine_grid."""
    cx = (W - 1) / 2.0
    cy = (H - 1) / 2.0
    cz = (D - 1) / 2.0
    c = np.array([cx, cy, cz], dtype=np.float64)

    t = c - R.astype(np.float64) @ c

    M = np.eye(4, dtype=np.float64)
    M[0:3, 0:3] = R
    M[0:3, 3] = t

    sx = (W - 1) / 2.0
    sy = (H - 1) / 2.0
    sz = (D - 1) / 2.0
    T = np.eye(4, dtype=np.float64)
    T[0,0] = sx; T[1,1] = sy; T[2,2] = sz
    T[0,3] = cx; T[1,3] = cy; T[2,3] = cz

    Tinv = np.linalg.inv(T)
    A = Tinv @ M @ T
    A = A.astype(np.float32)
    return A[0:3, 0:4]

def generate_views_torch(volume_np, rotations, out_dir, batch_size=1, downsample=3, device='cuda'):
    """
    volume_np: (D,H,W) float32 in [0,1]
    rotations: list of (rx, ry, rz) tuples in degrees
    Writes images to out_dir. Uses GPU device.
    """
    assert TORCH_AVAILABLE, "PyTorch not available."
    use_cuda = (device == 'cuda') and torch.cuda.is_available()
    dev = torch.device('cuda' if use_cuda else 'cpu')
    if not use_cuda:
        print("[WARN] CUDA not available, using CPU via PyTorch (slow).")

    # Optionally downsample to make resampling faster (depth + in-plane)
    vol = volume_np
    D0, H0, W0 = vol.shape

    # Depth subsample
    if DEPTH_DOWNSAMPLE > 1:
        vol = vol[::DEPTH_DOWNSAMPLE, :, :]

    # In-plane downsample if requested
    if downsample > 1:
        D, H, W = vol.shape
        H2 = max(8, H // downsample)
        W2 = max(8, W // downsample)
        vol_small = np.zeros((D, H2, W2), dtype=np.float32)
        for z in range(D):
            vol_small[z] = cv2.resize(vol[z], (W2, H2), interpolation=cv2.INTER_LINEAR)
        vol = vol_small

    D, H, W = vol.shape
    print(f"[INFO] Using volume shape for resampling: (D,H,W) = ({D},{H},{W}) (original was {D0},{H0},{W0})")
    tvol = torch.from_numpy(vol).unsqueeze(0).unsqueeze(0).to(dev)  # (1,1,D,H,W)

    total = len(rotations)

    for bstart in tqdm(range(0, total, batch_size), desc="Batches"):
        bend = min(total, bstart + batch_size)
        batch_rots = rotations[bstart:bend]
        batch_n = len(batch_rots)
        affines = np.zeros((batch_n, 3, 4), dtype=np.float32)
        for i, (rx, ry, rz) in enumerate(batch_rots):
            R = euler_angles_to_rotation_matrix(rx, ry, rz)
            A = make_affine_norm_from_R(R, D, H, W)
            affines[i] = A
        A_t = torch.from_numpy(affines).to(dev)  # (B,3,4)

        # Try batch processing, fallback to per-sample on OOM
        try:
            grid = F.affine_grid(A_t, size=(batch_n, 1, D, H, W), align_corners=True)
            vbat = tvol.repeat(batch_n, 1, 1, 1, 1)
            sampled = F.grid_sample(vbat, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
            mip = torch.amax(sampled, dim=2)
            mip = mip.squeeze(1).cpu().numpy()  # (B,H,W)
            # save batch
            for i in range(batch_n):
                out_idx = bstart + i
                rx, ry, rz = batch_rots[i]
                fname = f"rot_{out_idx:04d}_rx{rx:+.2f}_ry{ry:+.2f}_rz{rz:+.2f}.png"
                path = os.path.join(out_dir, fname)
                img = np.clip(mip[i], 0.0, 1.0)
                if SAVE_16BIT:
                    tosave = (img * 65535.0).round().astype(np.uint16)
                else:
                    tosave = (img * 255.0).round().astype(np.uint8)
                cv2.imwrite(path, tosave)
        except RuntimeError as err:
            # If CUDA OOM, fallback to per-sample processing to reduce peak usage
            msg = str(err).lower()
            if 'out of memory' in msg and use_cuda:
                print("[WARN] CUDA OOM in batch - falling back to per-sample processing for this batch.")
                torch.cuda.empty_cache()
                for i_single in range(batch_n):
                    try:
                        Ai = A_t[i_single:i_single+1]
                        grid_s = F.affine_grid(Ai, size=(1,1,D,H,W), align_corners=True)
                        # use the base volume (no repeat)
                        samp = F.grid_sample(tvol, grid_s, mode='bilinear', padding_mode='zeros', align_corners=True)
                        mip_s = torch.amax(samp, dim=2).squeeze(1).cpu().numpy()  # (1,H,W)
                        rx, ry, rz = batch_rots[i_single]
                        out_idx = bstart + i_single
                        fname = f"rot_{out_idx:04d}_rx{rx:+.2f}_ry{ry:+.2f}_rz{rz:+.2f}.png"
                        path = os.path.join(out_dir, fname)
                        img = np.clip(mip_s[0], 0.0, 1.0)
                        if SAVE_16BIT:
                            tosave = (img * 65535.0).round().astype(np.uint16)
                        else:
                            tosave = (img * 255.0).round().astype(np.uint8)
                        cv2.imwrite(path, tosave)
                    except RuntimeError as e2:
                        # if even a single sample OOMs, raise (should be rare after downsample)
                        raise e2
            else:
                # Not an OOM - re-raise
                raise

        # free GPU cache to reduce fragmentation
        if use_cuda:
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

    print(f"[INFO] Generated and saved {len(rotations)} images to {out_dir}")

# ---------------------------
# FALLBACK CPU path (SimpleITK), single-threaded, slower
# ---------------------------
def generate_views_sitk(volume_np, rotations, out_dir):
    """Simple fallback using SimpleITK resample + MIP (very slow for many views)."""
    sitk_vol = sitk.GetImageFromArray(volume_np)
    size = sitk_vol.GetSize()
    center_index = [s/2.0 for s in size]
    center_phys = sitk_vol.TransformContinuousIndexToPhysicalPoint(center_index)
    idx = 0
    for rx, ry, rz in tqdm(rotations, desc="rotations"):
        transform = sitk.Euler3DTransform()
        transform.SetRotation(math.radians(rx), math.radians(ry), math.radians(rz))
        transform.SetCenter(center_phys)
        rotated = sitk.Resample(sitk_vol, sitk_vol, transform, sitk.sitkLinear, 0.0)
        rotated_np = sitk.GetArrayFromImage(rotated)
        mip = np.max(rotated_np, axis=0)
        if mip.max() != mip.min():
            mip_norm = (mip - mip.min()) / (mip.max() - mip.min())
        else:
            mip_norm = np.zeros_like(mip)
        fname = f"rot_{idx:04d}_rx{rx:+.2f}_ry{ry:+.2f}_rz{rz:+.2f}.png"
        path = os.path.join(out_dir, fname)
        if SAVE_16BIT:
            cv2.imwrite(path, (mip_norm * 65535.0).round().astype(np.uint16))
        else:
            cv2.imwrite(path, (mip_norm * 255.0).round().astype(np.uint8))
        idx += 1
    print(f"[INFO] Fallback: saved {idx} images to {out_dir}")

# ---------------------------
# MAIN
# ---------------------------
def main():
    print("[INFO] Loading volume...")
    vol, spacing = load_volume_from_folder(INPUT_DIR)

    print("[INFO] Normalizing + CLAHE...")
    vol_enh = normalize_and_clahe(vol)

    rotations = ROTATION_LIST
    print(f"[INFO] Total rotations to generate: {len(rotations)}")

    # Choose GPU path if torch + cuda available
    if TORCH_AVAILABLE and torch.cuda.is_available():
        print("[INFO] Using PyTorch on CUDA for acceleration.")
        generate_views_torch(vol_enh, rotations, OUTPUT_DIR, batch_size=BATCH_SIZE, downsample=DOWNSAMPLE_FACTOR, device='cuda')
    elif TORCH_AVAILABLE:
        print("[INFO] PyTorch available but CUDA not found. Using CPU PyTorch (slow).")
        generate_views_torch(vol_enh, rotations, OUTPUT_DIR, batch_size=BATCH_SIZE, downsample=DOWNSAMPLE_FACTOR, device='cpu')
    else:
        print("[WARN] PyTorch not available. Falling back to SimpleITK CPU resampling (very slow for many views).")
        generate_views_sitk(vol_enh, rotations, OUTPUT_DIR)

if __name__ == "__main__":
    main()
