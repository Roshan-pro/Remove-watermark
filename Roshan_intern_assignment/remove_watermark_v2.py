"""
Advanced Watermark Removal & Image Enhancement
===============================================
Handles:
  1. Blue/grey/red semi-transparent watermarks on diagrams
  2. Bluish color cast + low contrast (cool-tone scans)
  3. Auto-detection of clean images (skip unnecessary processing)

Output: Sharp, high-contrast grayscale PNG with clean white background.
Diagram content (lines, arrows, labels, hatching) is fully preserved.

Usage:
    python remove_watermark_v2.py <input_dir> [output_dir]
    python remove_watermark_v2.py samples/ output/

Requirements:
    pip install opencv-python-headless numpy pillow scipy
"""

import io
import os
import sys
import shutil
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    from PIL import Image, ImageFilter, ImageEnhance
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


# ─────────────────────────────────────────────────────────────────────────────
# STEP 0 — Clean image detection
# ─────────────────────────────────────────────────────────────────────────────

def _is_clean_image(img_bgr: np.ndarray) -> bool:
    """
    Returns True if the image appears watermark-free and high-contrast.
    Heuristics:
      - Very low blue-channel dominance over red/green
      - High contrast (std dev of grayscale > threshold)
      - No significant mid-tone "fog" (watermark tends to push pixels 100-200)
    """
    b, g, r = img_bgr[:, :, 0], img_bgr[:, :, 1], img_bgr[:, :, 2]

    # 1. Color cast: blue channel significantly brighter than red?
    blue_excess = float(np.mean(b.astype(int) - r.astype(int)))
    # > 8 means noticeable blue cast

    # 2. Midtone fog: fraction of pixels in the 100-220 gray range
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    midtone_fraction = float(np.mean((gray > 100) & (gray < 220)))
    # Watermarked images tend to have lots of mid-tone pixels

    # 3. Overall contrast
    dark_pixels = np.sum(gray < 80)
    bright_pixels = np.sum(gray > 230)
    total = gray.size
    contrast_ratio = (dark_pixels + bright_pixels) / total
    # High-contrast clean images: most pixels are either dark or bright

    is_clean = (
        blue_excess < 6 and       # minimal blue cast
        midtone_fraction < 0.15 and  # low midtone fog
        contrast_ratio > 0.75      # high black/white ratio
    )
    return is_clean


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Red/maroon watermark removal (HSV masking)
# ─────────────────────────────────────────────────────────────────────────────

def _remove_red_watermark(
    img_bgr: np.ndarray,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Detect and erase red/maroon watermark pixels using HSV color masking.
    Returns (modified_image, dilated_mask_or_None).
    The dilated mask is reused after grayscale processing to force those
    areas fully white (prevents sharpening from re-darkening edges).
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # Red wraps around 0/180 in HSV
    masks = [
        cv2.inRange(hsv, np.array([0,   50,  50]),  np.array([10,  255, 255])),
        cv2.inRange(hsv, np.array([170, 50,  50]),  np.array([180, 255, 255])),
        cv2.inRange(hsv, np.array([0,   40,  30]),  np.array([15,  255, 200])),  # maroon
    ]
    red_mask = masks[0] | masks[1] | masks[2]

    count = int(np.count_nonzero(red_mask))
    total = img_bgr.shape[0] * img_bgr.shape[1]

    if count < 10 or count > total * 0.20:
        return img_bgr, None

    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(red_mask, kernel, iterations=3)

    result = img_bgr.copy()
    result[dilated > 0] = [255, 255, 255]
    return result, dilated


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Blue watermark / color-cast removal via channel mixing
# ─────────────────────────────────────────────────────────────────────────────

def _neutralize_blue_cast(img_bgr: np.ndarray) -> np.ndarray:
    """
    Convert to grayscale using a channel mix that suppresses blue watermarks.

    Standard luminance (0.114B + 0.587G + 0.299R) keeps blue at ~11%.
    We suppress blue further and boost red to counteract blue-tinted watermarks:
      gray = 0.05*B + 0.50*G + 0.45*R

    This makes blue watermark pixels appear much lighter (closer to white)
    while black ink (equal RGB near 0) stays dark.
    """
    b = img_bgr[:, :, 0].astype(np.float32)
    g = img_bgr[:, :, 1].astype(np.float32)
    r = img_bgr[:, :, 2].astype(np.float32)

    # Watermark-suppressing mix (weights sum to 1.0)
    gray = 0.05 * b + 0.50 * g + 0.45 * r
    return np.clip(gray, 0, 255).astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Adaptive LUT: push light pixels to white, keep dark pixels
# ─────────────────────────────────────────────────────────────────────────────

def _build_lut(gray: np.ndarray) -> np.ndarray:
    """
    Build a 256-entry LUT based on image histogram.
    - Pixels above 'threshold' (watermark/bg) → 255 (white)
    - Pixels below 'threshold-ramp' (text/lines) → stretched darker
    - Soft transition in between (ramp zone)
    """
    # Find the dominant bright peak = background/watermark level
    p5  = float(np.percentile(gray, 5))
    p50 = float(np.percentile(gray, 50))
    p90 = float(np.percentile(gray, 90))

    # Threshold: midpoint between text cluster and background
    # Clamp between 150 and 215 to be conservative
    threshold = int(np.clip(p50 + 25, 150, 215))
    ramp_width = 45  # pixels: transition zone below threshold

    lut = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        if i >= threshold:
            lut[i] = 255
        elif i >= threshold - ramp_width:
            # Blend from dark→white in ramp zone
            t = (i - (threshold - ramp_width)) / float(ramp_width)
            lut[i] = int(i * (1 - t) + 255 * t)
        else:
            # Dark zone: keep but scale to preserve contrast
            lut[i] = i
    return lut


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Auto-contrast stretch
# ─────────────────────────────────────────────────────────────────────────────

def _auto_contrast(gray: np.ndarray, clip_low: float = 1.0) -> np.ndarray:
    """Stretch histogram so darkest non-white pixels → 0, white stays 255."""
    dark = gray[gray < 250]
    if dark.size == 0:
        return gray
    p_low = float(np.percentile(dark, clip_low))
    p_high = 255.0
    if p_low >= p_high:
        return gray
    stretched = (gray.astype(np.float32) - p_low) / (p_high - p_low) * 255
    return np.clip(stretched, 0, 255).astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — Sharpening (preserves diagram lines without haloing)
# ─────────────────────────────────────────────────────────────────────────────

def _sharpen(gray: np.ndarray, strength: float = 0.6) -> np.ndarray:
    """
    Unsharp masking: adds back high-frequency details.
    strength=0.6 is gentle — preserves hairlines without over-sharpening.
    """
    blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=1.5)
    sharpened = cv2.addWeighted(gray, 1 + strength, blurred, -strength, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — Noise / grid artifact removal via morphological closing
# ─────────────────────────────────────────────────────────────────────────────

def _remove_grid_artifacts(gray: np.ndarray) -> np.ndarray:
    """
    Some scanned images have a faint dot/grid watermark pattern.
    A small morphological closing followed by a median blur smooths these
    without blurring real diagram lines (which are thicker).
    """
    # Median blur radius 3 removes isolated single-pixel noise
    denoised = cv2.medianBlur(gray, 3)
    return denoised


# ─────────────────────────────────────────────────────────────────────────────
# MAIN pipeline
# ─────────────────────────────────────────────────────────────────────────────

def remove_watermark(img_path: str) -> Tuple[Optional[bytes], str]:
    """
    Full pipeline. Returns (png_bytes, status_string).
    status is one of: 'cleaned', 'skipped_clean', 'failed'
    """
    if not HAS_CV2:
        return _fallback_pil(img_path)

    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        return None, "failed"

    # ── Auto-detect: skip if already clean ──────────────────────────────────
    if _is_clean_image(img_bgr):
        # Still convert to grayscale for uniformity, but no watermark processing
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        ok, buf = cv2.imencode('.png', gray)
        return (buf.tobytes() if ok else None), "skipped_clean"

    # ── Step 0: Remove red watermark ────────────────────────────────────────
    img_bgr, red_mask = _remove_red_watermark(img_bgr)

    # ── Step 1: Channel-mix grayscale (suppress blue) ────────────────────────
    gray = _neutralize_blue_cast(img_bgr)

    # ── Step 2: Remove dot/grid noise ────────────────────────────────────────
    gray = _remove_grid_artifacts(gray)

    # ── Step 3: LUT — push watermark pixels to white ─────────────────────────
    lut = _build_lut(gray)
    gray = cv2.LUT(gray, lut)

    # ── Step 4: Auto-contrast stretch ────────────────────────────────────────
    gray = _auto_contrast(gray, clip_low=1.0)

    # ── Step 5: Gentle sharpening ────────────────────────────────────────────
    gray = _sharpen(gray, strength=0.5)

    # ── Step 6: Re-apply red mask (prevents edge darkening) ──────────────────
    if red_mask is not None:
        # Resize mask if needed (shouldn't differ, but be safe)
        if red_mask.shape != gray.shape:
            red_mask = cv2.resize(red_mask, (gray.shape[1], gray.shape[0]),
                                  interpolation=cv2.INTER_NEAREST)
        gray[red_mask > 0] = 255

    ok, buf = cv2.imencode('.png', gray)
    return (buf.tobytes() if ok else None), "cleaned"


# ─────────────────────────────────────────────────────────────────────────────
# PIL fallback (no OpenCV)
# ─────────────────────────────────────────────────────────────────────────────

def _fallback_pil(img_path: str) -> Tuple[Optional[bytes], str]:
    if not HAS_PIL:
        print("ERROR: Install opencv-python or pillow.")
        return None, "failed"

    try:
        img = Image.open(img_path).convert("RGB")
    except Exception as e:
        print(f"  PIL open error: {e}")
        return None, "failed"

    arr = np.array(img, dtype=np.float32)
    b, g, r = arr[:, :, 2], arr[:, :, 1], arr[:, :, 0]

    # Same blue-suppressing channel mix
    gray_f = 0.05 * b + 0.50 * g + 0.45 * r
    gray = np.clip(gray_f, 0, 255).astype(np.uint8)

    p50 = float(np.percentile(gray, 50))
    threshold = int(np.clip(p50 + 25, 150, 215))
    ramp = 45

    result = gray.astype(np.float32)
    hi_mask = result >= threshold
    ramp_mask = (result >= threshold - ramp) & ~hi_mask
    t = (result[ramp_mask] - (threshold - ramp)) / ramp
    result[ramp_mask] = result[ramp_mask] * (1 - t) + 255 * t
    result[hi_mask] = 255
    result = np.clip(result, 0, 255).astype(np.uint8)

    # Auto-contrast
    dark = result[result < 250]
    if dark.size > 0:
        lo = float(np.percentile(dark, 1))
        result = np.clip((result.astype(float) - lo) / (255 - lo) * 255, 0, 255).astype(np.uint8)

    out = Image.fromarray(result).filter(ImageFilter.SHARPEN)
    buf = io.BytesIO()
    out.save(buf, format="PNG")
    return buf.getvalue(), "cleaned"


# ─────────────────────────────────────────────────────────────────────────────
# CLI — batch directory processing
# ─────────────────────────────────────────────────────────────────────────────

def process_directory(input_dir: str, output_dir: str):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
    images = sorted([f for f in input_path.iterdir() if f.suffix.lower() in exts])

    if not images:
        print(f"No images found in {input_dir}")
        return

    print(f"Processing {len(images)} image(s)...\n")
    counts = {"cleaned": 0, "skipped_clean": 0, "failed": 0}

    for img_path in images:
        png_bytes, status = remove_watermark(str(img_path))
        out_file = output_path / img_path.with_suffix(".png").name

        icon = {"cleaned": "✓", "skipped_clean": "○", "failed": "✗"}[status]
        label = {"cleaned": "cleaned", "skipped_clean": "clean (passed through)", "failed": "FAILED"}[status]
        print(f"  {icon} {img_path.name} → {label}")

        if png_bytes:
            with open(out_file, "wb") as f:
                f.write(png_bytes)
        else:
            shutil.copy2(img_path, output_path / img_path.name)

        counts[status] += 1

    print(f"\n{'─'*50}")
    print(f"Results: {counts['cleaned']} cleaned | "
          f"{counts['skipped_clean']} already clean | "
          f"{counts['failed']} failed")
    print(f"Output → {output_path.resolve()}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    in_dir = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "Roshan_intern_assignment/output_cleaned"
    process_directory(in_dir, out_dir)