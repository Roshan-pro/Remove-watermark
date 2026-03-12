## Watermark Removal for Scanned Question Paper Images

## 1. Problem Overview

Many scanned question papers contain semi-transparent watermarks such as:

- Institution branding
- Watermark text
- Colored overlays (blue, red, grey)
- Grid or dotted watermark patterns

These watermarks reduce the readability of diagrams and text.

The goal of this project is to remove these watermarks while preserving the original content, including:

- Black text
- Diagrams
- Arrows
- Labels
- Line art

The output must be a clean grayscale image with a white background suitable for reading or printing.

---

## 2. Challenges in Watermark Removal

Removing watermarks from scanned documents presents several difficulties.

### 1. Colored overlays

Watermarks may appear in colors such as:

- Blue
- Red
- Grey
- Semi-transparent tint

### 2. Transparency

Watermarks are often blended with the background, making them difficult to isolate.

### 3. Diagram preservation

Aggressive watermark removal may damage:

- Thin diagram lines
- Labels
- Arrows
- Graph edges

### 4. Already clean images

Some images contain no watermark, so unnecessary processing may reduce quality.

---

## 3. Exploration and Approaches

Several approaches were explored during development.

### Approach 1 — Simple Thresholding

Convert image to grayscale and apply thresholding.

**Problem:**

- Watermark pixels often overlap with diagram intensities.
- Thin diagram lines get removed.

Therefore this approach was not suitable.

---

### Approach 2 — Color Filtering

Watermarks frequently contain distinct color components, so filtering based on color space was tested.

Color spaces explored:

- RGB
- HSV
- LAB

HSV was particularly effective for detecting red watermarks.

---

### Approach 3 — Channel Mixing for Blue Watermarks

Blue watermarks are difficult to remove using standard grayscale conversion.

Standard grayscale formula:

```
Gray = 0.114B + 0.587G + 0.299R
```

This still keeps blue visible.

Instead, a modified grayscale conversion was used:

```
Gray = 0.05B + 0.50G + 0.45R
```

This suppresses blue watermark pixels while keeping black ink dark.

---

### Approach 4 — Histogram Based Separation

Watermarked images usually contain many mid-tone pixels.

By analyzing the grayscale histogram we can separate:

- Diagram pixels (dark)
- Watermark pixels (mid tones)
- Background pixels (bright)

This allows adaptive thresholding.

---

### Approach 5 — LUT-Based Pixel Transformation

A Look-Up Table (LUT) transforms pixel intensities.

**Behavior:**

| Pixel Type    | Transformation              |
|---------------|-----------------------------|
| Dark pixels   | Preserve                    |
| Mid pixels    | Gradually move toward white |
| Bright pixels | Convert to pure white       |

This removes watermark haze while preserving diagrams.

---

## 4. Final Processing Pipeline

The final solution uses a multi-stage image processing pipeline.

```
Input Image
      │
      ▼
Step 0: Clean Image Detection
      │
      ▼
Step 1: Red Watermark Removal (HSV masking)
      │
      ▼
Step 2: Blue Watermark Suppression
      │
      ▼
Step 3: Noise / Grid Artifact Removal
      │
      ▼
Step 4: Adaptive LUT Thresholding
      │
      ▼
Step 5: Auto Contrast Stretch
      │
      ▼
Step 6: Edge Preserving Sharpening
      │
      ▼
Output Clean Image
```

---

## 5. Detailed Explanation of Steps

### Step 0 — Clean Image Detection

Some images are already clean.

The algorithm checks:

- Blue channel dominance
- Mid-tone pixel ratio
- Overall contrast

If the image appears clean, watermark removal is skipped.

---

### Step 1 — Red Watermark Removal

Red or maroon watermark pixels are detected using HSV color masking.

Hue ranges used:

```
0–10
170–180
```

Detected regions are dilated slightly and replaced with white pixels.

---

### Step 2 — Blue Watermark Suppression

A custom grayscale channel mix suppresses blue watermark intensity:

```
Gray = 0.05B + 0.50G + 0.45R
```

This makes blue watermark pixels lighter while preserving black text.

---

### Step 3 — Grid Artifact Removal

Some scans contain faint grid or dotted watermark patterns.

A median blur filter removes small noise without damaging real diagram lines.

---

### Step 4 — Adaptive LUT Transformation

A histogram analysis determines a dynamic threshold.

Pixels are categorized as:

- Diagram pixels
- Watermark pixels
- Background pixels

The LUT pushes watermark pixels toward white.

---

### Step 5 — Auto Contrast Stretch

Contrast is enhanced by stretching the grayscale range.

Dark pixels move toward **0** while background remains **255**.

This improves readability.

---

### Step 6 — Edge Preserving Sharpening

Unsharp masking enhances diagram edges:

```
Sharpened = Original + Strength × (Original − Blur)
```

This preserves:

- Thin lines
- Diagram boundaries
- Labels

---

## 6. Libraries Used

| Library | Purpose |
|--------|--------|
| OpenCV | Image processing operations |
| NumPy | Numerical computations |
| Pillow | Image loading fallback |
| SciPy | Optional image operations |

Install dependencies:

```
pip install opencv-python-headless numpy pillow scipy
```

---

## 7. Usage

Run the script using:

```
python Roshan_intern_assignment/remove_watermark_v2.py <input_directory> <output_directory>
```

**Example:**

```
python Roshan_intern_assignment/remove_watermark_v2.py samples/watermarked output/
```

**The script will:**

1. Process all images in the input directory  
2. Save cleaned images in the output directory  
3. Print the processing result for each file  

**Example output:**

```
✓ image1.png → cleaned
○ image2.png → clean (passed through)
✗ image3.png → FAILED
```

---

## 8. Output Format

All processed images are saved as:

- PNG format
- Grayscale
- High contrast
- White background

This ensures clear readability of diagrams and text.