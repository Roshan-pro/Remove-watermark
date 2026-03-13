import cv2
import numpy as np
import os
import sys
from pathlib import Path

def enhance_and_clean(img_path, output_path):
    # 1. Load Image
    img = cv2.imread(str(img_path))
    if img is None:
        return False

    # 2. Channel Mixing (Suppress Blue/Red watermarks)
    # We give more weight to Green because it's usually the cleanest channel 
    # in diagrams with blue or red watermarks.
    b, g, r = cv2.split(img)
    # gray = 0.05*B + 0.50*G + 0.45*R
    gray = cv2.addWeighted(g, 0.45, r, 0.8, 0)
    gray = cv2.addWeighted(gray, 1.0, b, 0.05, 0)

    # 3. Denoise (Removes JPEG artifacts and watermark "dust")
    # fastNlMeansDenoising is great for cleaning backgrounds without losing line art
    denoised = cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)

    # 4. Adaptive Thresholding / Background Leveling
    # This pushes the greyish background (watermark remnants) to pure white
    # while keeping the diagram lines black.
    kernel_size = 31 # Adjust based on image resolution
    thresh = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY, kernel_size, 15)

    # 5. Sharpening (The "Blurry to Clear" step)
    # Using a sharpening kernel to define edges
    kernel = np.array([[-1,-1,-1], 
                       [-1, 9,-1],
                       [-1,-1,-1]])
    sharpened = cv2.filter2D(thresh, -1, kernel)

    # 6. Final Morphological Clean-up
    # Removes tiny specks that might remain
    kernel_clean = np.ones((2,2), np.uint8)
    final = cv2.morphologyEx(sharpened, cv2.MORPH_OPEN, kernel_clean)

    # Save Output
    cv2.imwrite(str(output_path), final)
    return True

def main():
    if len(sys.argv) < 2:
        print("Usage: python script.py <input_folder> [output_folder]")
        return

    input_dir = Path(sys.argv[1])
    output_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("cleaned_diagrams")
    output_dir.mkdir(parents=True, exist_ok=True)

    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    files = [f for f in input_dir.iterdir() if f.suffix.lower() in extensions]

    print(f"Processing {len(files)} images...")
    for f in files:
        out_file = output_dir / f"{f.stem}_cleaned.png"
        success = enhance_and_clean(f, out_file)
        if success:
            print(f"✓ Cleaned: {f.name}")
        else:
            print(f"✗ Failed: {f.name}")

if __name__ == "__main__":
    main()