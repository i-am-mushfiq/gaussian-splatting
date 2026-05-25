#!/usr/bin/env python3
"""
Fixed end-to-end script:
- Loads a DICOM series (or a folder of image slices)
- Builds 3D volume and extracts correct metadata
- Normalizes intensities, applies CLAHE per slice
- Generates rotated/resliced views using SimpleITK (Euler3DTransform centered at volume center)
- Produces MIP projections and exports images
- (Optionally) initializes a COLMAP DB via pycolmap and runs COLMAP feature/match/mapper steps via subprocess
"""

import os
import sys
import numpy as np
import SimpleITK as sitk
import cv2
import imageio.v3 as iio
import subprocess
from scipy.ndimage import affine_transform  # may be unused but kept if you want scipy-based transforms

# --- Configuration Parameters ---
INPUT_DIR = r"Data\Brain2_training\images"
OUTPUT_DIR = r"Data\Brain_new\synth_views"
COLMAP_DB = 'colmap_database.db'

# Virtual camera settings (list of degrees to apply as pitch rotations -- script uses SimpleITK so these are safe)
ROTATION_ANGLES = [1.0, -1.0, 2.0, -2.0]  # small pitch rotations (degrees)

# If you want orbital generation (e.g., 600 views), generate a grid of (azimuth, elevation) externally and pass here:
# e.g., make rotation_angles a list of (rx_deg, ry_deg, rz_deg) tuples. This script supports single-angle floats (applied to X)
# and also supports passing tuples (rx, ry, rz).
# Example orbital generator (not run by default):
# azs = np.linspace(0, 360, 30, endpoint=False); els = np.linspace(-60, 60, 20)
# ROTATION_ANGLES = [(el, az, 0) for el in els for az in azs]   # 30 * 20 = 600

# ---------------------------
# 1. Data Loading, Metadata Extraction, and Registration
# ---------------------------
def load_and_register_volume(input_dir):
    """
    Loads 2D slices from a directory (attempts DICOM-series first, then falls back to common image types).
    Returns:
        - volume_np: numpy array with shape (Z, Y, X) of float32
        - metadata: dict with keys: W, H, D, spacing (tuple (sx, sy, sz)), origin, direction
    """
    # Try reading DICOM series
    try:
        dicom_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(input_dir)
        if len(dicom_names) == 0:
            raise RuntimeError("No DICOM series found.")
        reader = sitk.ImageSeriesReader()
        reader.SetFileNames(dicom_names)
        volume_sitk = reader.Execute()
        print(f"[INFO] Loaded DICOM series with {len(dicom_names)} files.")
    except Exception as e:
        # Fallback: load common image files sorted by name (PNG/TIF/JPG)
        print(f"[INFO] DICOM load failed or not found: {e}. Falling back to image files in folder.")
        img_files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.tif', '.tiff', '.jpg', '.jpeg'))])
        if not img_files:
            raise FileNotFoundError(f"No image files found in {input_dir}")
        slices = [sitk.ReadImage(os.path.join(input_dir, f), sitk.sitkFloat32) for f in img_files]
        volume_sitk = sitk.JoinSeries(slices)
        print(f"[INFO] Loaded {len(img_files)} image slices.")

    # Extract size and spacing
    size = volume_sitk.GetSize()      # (size_x, size_y, size_z)
    spacing = volume_sitk.GetSpacing()  # (sx, sy, sz)
    origin = volume_sitk.GetOrigin()
    direction = volume_sitk.GetDirection()
    W, H, D = int(size[0]), int(size[1]), int(size[2])

    metadata = {
        'W': W,
        'H': H,
        'D': D,
        'spacing': spacing,
        'origin': origin,
        'direction': direction
    }

    # Convert to numpy array (SimpleITK uses GetArrayFromImage -> (z,y,x))
    volume_np = sitk.GetArrayFromImage(volume_sitk).astype(np.float32)

    print(f"[INFO] Volume shape (Z, Y, X): {volume_np.shape}, spacing: {spacing}")
    return volume_np, metadata


# ---------------------------
# 2. Intensity Standardization and Feature Enhancement
# ---------------------------
def normalize_and_enhance(volume_np, clip_percentiles=(1, 99), clahe_clip=2.0, clahe_grid=(8, 8)):
    """
    Applies percentile-based clipping + linear scaling to [0,1] then CLAHE per slice (8-bit intermediate).
    Input: volume_np shape (Z, Y, X), dtype float32
    Returns: enhanced_volume same shape, dtype float32 in [0,1]
    """
    vol = volume_np.copy()
    # Robust clipping using percentiles
    p0, p100 = np.percentile(vol, clip_percentiles)
    vol = np.clip(vol, p0, p100)
    # Scale to 0-1
    vmin, vmax = vol.min(), vol.max()
    if vmax - vmin < 1e-6:
        vol_norm = np.zeros_like(vol, dtype=np.float32)
    else:
        vol_norm = (vol - vmin) / (vmax - vmin)
    # Apply CLAHE slice-by-slice (convert to 8-bit, apply CLAHE, convert back)
    z_dim = vol_norm.shape[0]
    enhanced = np.zeros_like(vol_norm, dtype=np.float32)
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=clahe_grid)
    for z in range(z_dim):
        slice_f = vol_norm[z, ...]
        slice_8 = np.round(slice_f * 255.0).astype(np.uint8)
        slice_clahe = clahe.apply(slice_8)
        enhanced[z, ...] = slice_clahe.astype(np.float32) / 255.0
    print(f"[INFO] Normalized and applied CLAHE to {z_dim} slices.")
    return enhanced


# ---------------------------
# 3. Intrinsic Parameter Calculation
# ---------------------------
def calculate_intrinsics_from_metadata(metadata, focal_factor=1.2):
    """
    Calculates a PINHOLE-like intrinsics vector [fx, fy, cx, cy] in pixel units,
    using the image dimensions from metadata.
    """
    W = metadata['W']
    H = metadata['H']
    # Principal point at image center
    cx = W / 2.0
    cy = H / 2.0
    # Approximate focal length in pixels (coarse heuristic)
    f_pixel = focal_factor * max(W, H)
    fx = f_pixel
    fy = f_pixel
    intrinsics = [float(fx), float(fy), float(cx), float(cy)]
    print(f"[INFO] Intrinsics estimated: fx={fx:.2f}, fy={fy:.2f}, cx={cx:.1f}, cy={cy:.1f}")
    return intrinsics, f_pixel


# ---------------------------
# 4. Geometric Parallax Generator (SimpleITK-based rotations/reslicing)
# ---------------------------
def generate_synthetic_views(volume_np, metadata, rotation_angles):
    """
    Generate synthetic views by applying small 3D rotations (centered at the volume center)
    and producing a 2D projection (MIP) for each rotated volume.

    rotation_angles: list of either:
        - floats (interpreted as rotation about X axis in degrees), or
        - tuples/lists (rx_deg, ry_deg, rz_deg) for explicit rotations in degrees.

    Returns:
        synthetic_images: list of dicts { 'image_data': np.array (Y,X) floats in [0,1], 'filename': str }
    """
    synthetic_images = []

    # Convert numpy volume back to SimpleITK image for correct spacing/origin
    sitk_vol = sitk.GetImageFromArray(volume_np)  # expects z,y,x ordering
    if 'spacing' in metadata:
        # metadata['spacing'] is (sx, sy, sz) where SimpleITK expects (sx, sy, sz)
        try:
            sitk_vol.SetSpacing(metadata['spacing'])
        except Exception:
            # If metadata spacing doesn't match in shape, ignore
            pass
    # Set center for transform in physical coordinates (continuous index -> physical point)
    size = sitk_vol.GetSize()  # (x,y,z)
    center_index = [s / 2.0 for s in size]
    center_phys = sitk_vol.TransformContinuousIndexToPhysicalPoint(center_index)

    def apply_rotation_and_mip(rx_deg, ry_deg, rz_deg, out_index):
        transform = sitk.Euler3DTransform()
        # Set rotation in radians. Euler3DTransform.SetRotation(rx, ry, rz) expects radians.
        transform.SetRotation(np.deg2rad(rx_deg), np.deg2rad(ry_deg), np.deg2rad(rz_deg))
        transform.SetCenter(center_phys)
        # Resample with same grid as original (preserves spacing and size)
        rotated = sitk.Resample(sitk_vol, sitk_vol, transform, sitk.sitkLinear, 0.0)
        rotated_np = sitk.GetArrayFromImage(rotated)  # (z,y,x)
        # Create MIP along Z axis (axis 0)
        mip = np.max(rotated_np, axis=0)
        # Normalize mip to [0,1]
        if np.allclose(mip.max(), mip.min()):
            mip_norm = np.zeros_like(mip, dtype=np.float32)
        else:
            mip_norm = (mip - float(mip.min())) / float(mip.max() - mip.min())
        fname = f"rot_{out_index:04d}_rx{rx_deg:+.2f}_ry{ry_deg:+.2f}_rz{rz_deg:+.2f}.png"
        return mip_norm.astype(np.float32), fname

    idx = 0
    for ang in rotation_angles:
        if isinstance(ang, (list, tuple)) and len(ang) == 3:
            rx, ry, rz = float(ang[0]), float(ang[1]), float(ang[2])
        else:
            # If a single float is provided, interpret as rotation around X (pitch)
            rx, ry, rz = float(ang), 0.0, 0.0
        mip_img, fname = apply_rotation_and_mip(rx, ry, rz, idx)
        synthetic_images.append({
            'image_data': mip_img,
            'filename': fname,
            'pose_id': f"rx{rx:+.2f}_ry{ry:+.2f}_rz{rz:+.2f}"
        })
        idx += 1

    # Optionally include original center (MIP of original) for baseline
    original_mip = np.max(volume_np, axis=0)
    if np.allclose(original_mip.max(), original_mip.min()):
        original_norm = np.zeros_like(original_mip, dtype=np.float32)
    else:
        original_norm = (original_mip - float(original_mip.min())) / float(original_mip.max() - original_mip.min())
    synthetic_images.append({
        'image_data': original_norm.astype(np.float32),
        'filename': "original_mip.png",
        'pose_id': 'original'
    })

    print(f"[INFO] Generated {len(synthetic_images)} synthetic views.")
    return synthetic_images


# ---------------------------
# 5. Image Export and File Management
# ---------------------------
def export_images(synthetic_images, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for img_dict in synthetic_images:
        path = os.path.join(output_dir, img_dict['filename'])
        # Clip and scale to 16-bit unsigned integers for good dynamic range
        img = img_dict['image_data']
        img_clipped = np.clip(img, 0.0, 1.0)
        data_to_save = (img_clipped * 65535.0).round().astype(np.uint16)
        iio.imwrite(path, data_to_save)
    print(f"[INFO] Exported {len(synthetic_images)} images to {output_dir}")


# ---------------------------
# 6. COLMAP Database Initialization (pycolmap optional)
# ---------------------------
def setup_colmap_database(output_dir, db_path, camera_params, W, H):
    """
    Try to use pycolmap to prepare a database with a single PINHOLE camera and add images.
    If pycolmap is not installed, print instructions.
    """
    try:
        import pycolmap
    except Exception:
        print("[WARN] pycolmap is not installed. Skipping automatic DB creation. You can create the COLMAP database manually or install pycolmap.")
        return

    # Remove old DB if exists
    if os.path.exists(db_path):
        os.remove(db_path)
    db = pycolmap.Database(db_path)
    # Add camera: params for PINHOLE are [fx, fy, cx, cy]
    cam_id = db.add_camera('PINHOLE', int(W), int(H), camera_params)
    print(f"[INFO] Registered camera ID {cam_id} with params {camera_params}")

    # Add images
    image_files = sorted([f for f in os.listdir(output_dir) if f.lower().endswith(('.png', '.tif', '.tiff', '.jpg', '.jpeg'))])
    for im in image_files:
        db.add_image(im, cam_id)
    db.close()
    print(f"[INFO] Added {len(image_files)} images to COLMAP DB: {db_path}")


# ---------------------------
# 7. Execute COLMAP Pipeline via subprocess (feature extraction, matching, mapper)
# ---------------------------
def run_colmap_pipeline(db_path, image_path, output_sparse_dir=None):
    """
    Runs COLMAP CLI steps using subprocess. Requires 'colmap' in PATH.
    This function performs:
      - feature_extractor
      - exhaustive_matcher
      - mapper (sparse reconstruction)
    """
    if shutil_which('colmap') is None:
        print("[WARN] 'colmap' binary not found in PATH. Skipping COLMAP pipeline execution. Install COLMAP and ensure it's in PATH.")
        return

    # Feature extraction
    print("[INFO] Running COLMAP feature_extractor...")
    fe_command = [
        'colmap', 'feature_extractor',
        '--database_path', db_path,
        '--image_path', image_path,
        '--ImageReader.single_camera', '1',        # we used single camera model
        '--SiftExtraction.estimate_affine_shape', '0',
        '--SiftExtraction.domain_size_pooling', '1'
    ]
    subprocess.run(fe_command, check=True)

    # Feature matching (exhaustive)
    print("[INFO] Running COLMAP exhaustive_matcher...")
    fm_command = [
        'colmap', 'exhaustive_matcher',
        '--database_path', db_path
    ]
    subprocess.run(fm_command, check=True)

    # Run mapper (sparse)
    # Create output directory for sparse results
    if output_sparse_dir is None:
        output_sparse_dir = os.path.join(image_path, 'sparse')
    os.makedirs(output_sparse_dir, exist_ok=True)

    print("[INFO] Running COLMAP mapper (sparse reconstruction)...")
    map_command = [
        'colmap', 'mapper',
        '--database_path', db_path,
        '--image_path', image_path,
        '--export_path', output_sparse_dir,
        # relax tri_min_angle to allow small baselines if necessary
        '--Mapper.tri_min_angle', '1'
    ]
    # Note: depending on COLMAP version, flags may differ (e.g., 'mapper' vs 'automatic_reconstructor' or 'bundle_adjuster').
    subprocess.run(map_command, check=True)

    print(f"[INFO] COLMAP pipeline completed. Sparse models at {output_sparse_dir}")


# helper to check binary existence
def shutil_which(cmd):
    import shutil
    return shutil.which(cmd)


# ---------------------------
# Main execution
# ---------------------------
if __name__ == '__main__':
    # 1. Load and prepare volume
    volume_3d, meta = load_and_register_volume(INPUT_DIR)

    # 2. Normalize and enhance
    enhanced_volume = normalize_and_enhance(volume_3d)

    # 3. Generate synthetic views (using rotation angles)
    synthetic_views_data = generate_synthetic_views(enhanced_volume, meta, ROTATION_ANGLES)

    # 4. Export synthetic images
    export_images(synthetic_views_data, OUTPUT_DIR)

    # 5. Setup COLMAP DB (optional, requires pycolmap)
    intrinsics, fpix = calculate_intrinsics_from_metadata(meta)
    setup_colmap_database(OUTPUT_DIR, COLMAP_DB, intrinsics, meta['W'], meta['H'])

    # 6. Run COLMAP pipeline (optional, requires colmap in PATH)
    try:
        run_colmap_pipeline(COLMAP_DB, OUTPUT_DIR)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] COLMAP subprocess failed: {e}. You can run the commands manually.")
    except Exception as e:
        print(f"[WARN] Skipping COLMAP pipeline: {e}")

    print("[INFO] Script finished.")
