#!/usr/bin/env python3
"""
Mosaic green-tesserae fluorescence simulator (tile-wise), with debug outputs.

- Segments individual greenish tesserae by region-growing on HSV.
- Classifies per tile: bright-green (strong), pale-green (weak), bluish (none).
- Applies fluorescence only to whole tiles with soft-edged borders.
- Exports:
    01/02  candidate masks
    03     RED bounding-box overlay (segmentation)
    04     classification overlay
    10/11  fluorescence renders (black / on mosaic)
    plots_*.png  debug charts
    tiles_classification.csv  per-tile stats

Requires: pillow, numpy, matplotlib, pandas

Run:
    python mosaic_fluorescence.py --src path/to/image.jpg --out out_dir
"""

import os, argparse
import numpy as np
from collections import deque
from PIL import Image, ImageFilter, ImageDraw
import matplotlib.pyplot as plt
import pandas as pd


# ---------------------------- Parameters ----------------------------

# Working resolution (long side). Higher is slower but gives smoother borders.
LONG_SIDE = 2200

# Broad HSV gate for "greenish" candidates (Pillow HSV: H in [0..255]):
H_MIN, H_MAX = 35, 110
S_MIN, V_MIN = 40, 40

# Region-growing tolerances around each seed (narrow in Hue, looser in S/V):
TOL_H, TOL_S, TOL_V = 12, 90, 90

# Area filter for tile components (pixels at working resolution):
MIN_AREA, MAX_AREA = 25, 20000

# Fluorescence tint (~525 nm green) in RGB:
TINT = np.array([0.0, 1.0, 0.55], dtype=np.float32)

# Multi-scale bloom radii and weights (per channel):
BLOOM = [(0, 1.3), (2, 1.0), (5, 0.8), (12, 0.6), (24, 0.45)]  # radius, weight

# --------------------------------------------------------------------


def load_and_rescale(path, long_side=LONG_SIDE):
    im0 = Image.open(path).convert("RGB")
    w0, h0 = im0.size
    scale = long_side / max(w0, h0)
    im = im0.resize((int(w0 * scale), int(h0 * scale)), resample=Image.Resampling.LANCZOS)
    return im


def pil_to_hsv_arrays(pil_img):
    hsv = pil_img.convert("HSV")
    arr = np.array(hsv).astype(np.int16)
    H, S, V = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    return H, S, V


def mask_closing_bool(mask_bool, size=3):
    img = Image.fromarray((mask_bool.astype(np.uint8) * 255))
    img = img.filter(ImageFilter.MaxFilter(size=size))  # dilate
    img = img.filter(ImageFilter.MinFilter(size=size))  # erode
    return np.array(img) > 0


def grow_region(seed_y, seed_x, H, S, V, cand_mask):
    """4-connected flood-fill constrained by proximity to seed HSV."""
    Ht, Wt = cand_mask.shape
    Hs, Ss, Vs = int(H[seed_y, seed_x]), int(S[seed_y, seed_x]), int(V[seed_y, seed_x])
    q = deque([(seed_y, seed_x)])
    visited_local = set()
    coords = []
    hsum = ssum = vsum = 0

    while q:
        y, x = q.pop()
        if not (0 <= y < Ht and 0 <= x < Wt):
            continue
        if (y, x) in visited_local:
            continue
        visited_local.add((y, x))
        if not cand_mask[y, x]:
            continue
        if (
            abs(int(H[y, x]) - Hs) <= TOL_H
            and abs(int(S[y, x]) - Ss) <= TOL_S
            and abs(int(V[y, x]) - Vs) <= TOL_V
        ):
            coords.append((y, x))
            hsum += int(H[y, x]); ssum += int(S[y, x]); vsum += int(V[y, x])
            q.extend(((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)))

    if not coords:
        return None

    ys = [c[0] for c in coords]; xs = [c[1] for c in coords]
    y0, y1 = min(ys), max(ys); x0, x1 = min(xs), max(xs)
    mask_local = np.zeros((y1 - y0 + 1, x1 - x0 + 1), dtype=bool)
    for (yy, xx) in coords:
        mask_local[yy - y0, xx - x0] = True
    Hmean, Smean, Vmean = hsum / len(coords), ssum / len(coords), vsum / len(coords)
    return {"mask": mask_local, "bbox": (x0, y0, x1, y1),
            "Hmean": Hmean, "Smean": Smean, "Vmean": Vmean, "area": len(coords)}


def classify_tile(Hm, Sm, Vm):
    """Return (intensity in [0,1], label)."""
    # Hue logic (0..255 scale)
    if Hm >= 90:                # cyan/blueish -> no fluorescence
        return 0.0, "bluish"
    if Hm < 40:
        hue_w = 0.5             # yellow-green -> weaker
    elif Hm <= 85:
        hue_w = 1.0             # proper green
    else:                        # edge-cyan
        hue_w = 0.2

    # Pale vs bright modulation by S and V
    wS = max(0.0, min(1.0, (Sm - 70) / (180 - 70)))
    wV = max(0.0, min(1.0, (Vm - 80) / (220 - 80)))
    intensity = hue_w * (0.6 * wS + 0.4 * wV)
    intensity = float(max(0.0, min(1.0, intensity)))

    label = "bright-green" if intensity > 0.65 else ("pale-green" if intensity > 0.15 else "very-pale")
    return intensity, label


def gaussian_blur_gray(arr, radius):
    img = Image.fromarray(np.uint8(np.clip(arr * 255, 0, 255)))
    img = img.filter(ImageFilter.GaussianBlur(radius=radius))
    return np.array(img).astype(np.float32) / 255.0


def main(src, outdir):
    os.makedirs(outdir, exist_ok=True)
    im = load_and_rescale(src, LONG_SIDE)
    w, h = im.size
    H, S, V = pil_to_hsv_arrays(im)

    # Candidate greenish pixels
    cand = (H >= H_MIN) & (H <= H_MAX) & (S >= S_MIN) & (V >= V_MIN)
    Image.fromarray((cand.astype(np.uint8) * 255)).save(os.path.join(outdir, "01_candidate_mask.png"))

    cand_clean = mask_closing_bool(cand, size=3)
    Image.fromarray((cand_clean.astype(np.uint8) * 255)).save(os.path.join(outdir, "02_candidate_mask_closed.png"))

    # Segment tiles
    visited = np.zeros((h, w), dtype=bool)
    tiles = []
    for yy in range(h):
        for xx in range(w):
            if cand_clean[yy, xx] and not visited[yy, xx]:
                t = grow_region(yy, xx, H, S, V, cand_clean)
                if not t:
                    continue
                x0, y0, x1, y1 = t["bbox"]
                visited[y0:y1 + 1, x0:x1 + 1] |= t["mask"]  # mark covered region
                if MIN_AREA <= t["area"] <= MAX_AREA:
                    tiles.append(t)

    # RED bounding-box overlay
    overlay = im.copy().convert("RGBA")
    draw = ImageDraw.Draw(overlay)
    for t in tiles:
        x0, y0, x1, y1 = t["bbox"]
        draw.rectangle([x0, y0, x1, y1], outline=(255, 0, 0, 170), width=2)  # RED
    overlay.save(os.path.join(outdir, "03_tiles_bbox_overlay_RED.png"))

    # Classify tiles + table
    rows = []
    for i, t in enumerate(tiles):
        intensity, label = classify_tile(t["Hmean"], t["Smean"], t["Vmean"])
        t["intensity"] = intensity
        t["label"] = label
        x0, y0, x1, y1 = t["bbox"]
        rows.append({
            "tile_id": i, "x0": x0, "y0": y0, "x1": x1, "y1": y1,
            "area_px": t["area"], "Hmean": t["Hmean"], "Smean": t["Smean"], "Vmean": t["Vmean"],
            "label": label, "intensity": intensity
        })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(outdir, "tiles_classification.csv"), index=False)

    # Classification overlay (colored by class)
    class_overlay = im.copy().convert("RGBA")
    draw = ImageDraw.Draw(class_overlay)
    for t in tiles:
        x0, y0, x1, y1 = t["bbox"]
        if t["label"] == "bright-green":
            col = (0, 255, 0, 150)
        elif t["label"] in ("pale-green", "very-pale"):
            col = (180, 255, 180, 120)
        else:
            col = (80, 80, 255, 150)  # bluish/excluded
        draw.rectangle([x0, y0, x1, y1], outline=col, width=2)
    class_overlay.save(os.path.join(outdir, "04_tiles_class_overlay.png"))

    # Fluorescence emission buffer
    emission = np.zeros((h, w, 3), dtype=np.float32)
    for t in tiles:
        if t["intensity"] <= 0.0:
            continue
        x0, y0, x1, y1 = t["bbox"]
        tile_mask = t["mask"]
        # Soft-edged alpha for whole-tile emission
        alpha = Image.fromarray((tile_mask.astype(np.uint8) * 255))
        alpha = alpha.filter(ImageFilter.MaxFilter(size=3))
        alpha = alpha.filter(ImageFilter.GaussianBlur(radius=1.2))
        alpha = np.array(alpha).astype(np.float32) / 255.0

        core = TINT * (0.55 + 0.45 * t["intensity"])  # brighter color for stronger tiles
        sl_y, sl_x = slice(y0, y1 + 1), slice(x0, x1 + 1)
        emission[sl_y, sl_x, :] += (alpha[..., None] * core[None, None, :]) * t["intensity"]

    emission = np.clip(emission, 0, 1)

    # Multi-scale bloom
    glow = np.zeros_like(emission)
    for c in range(3):
        chan = emission[:, :, c]
        acc = np.zeros_like(chan, dtype=np.float32)
        for radius, weight in BLOOM:
            if radius == 0:
                acc += weight * chan
            else:
                acc += weight * gaussian_blur_gray(chan, radius)
        glow[:, :, c] = acc

    # Normalize glow to avoid hard clipping
    gmax = glow.max() if glow.max() > 1e-6 else 1.0
    glow = np.clip(glow / gmax, 0, 1)

    # Compose renders
    im_np = np.array(im).astype(np.float32) / 255.0
    base_dark = np.clip(im_np * 0.18, 0, 1)  # dim ambient scene

    fluoro_black = (glow * 255).astype(np.uint8)
    fluoro_on = np.clip(base_dark + glow * 1.2, 0, 1)
    fluoro_on = (fluoro_on * 255).astype(np.uint8)

    Image.fromarray(fluoro_black).save(os.path.join(outdir, "10_fluorescence_black.png"))
    Image.fromarray(fluoro_on).save(os.path.join(outdir, "11_fluorescence_on_mosaic.png"))

    # ------------------------- Debug plots -------------------------
    # Note: each chart has its own figure and uses default colors.

    plt.figure()
    plt.imshow(im)
    plt.title("Original (rescaled)")
    plt.axis("off")
    plt.savefig(os.path.join(outdir, "plot_01_original.png"), bbox_inches="tight")

    plt.figure()
    plt.imshow(cand.astype(np.uint8) * 255)
    plt.title("Candidate green pixels (raw mask)")
    plt.axis("off")
    plt.savefig(os.path.join(outdir, "plot_02_candidate_raw.png"), bbox_inches="tight")

    plt.figure()
    plt.imshow(cand_clean.astype(np.uint8) * 255)
    plt.title("Candidate green pixels (closed)")
    plt.axis("off")
    plt.savefig(os.path.join(outdir, "plot_03_candidate_closed.png"), bbox_inches="tight")

    plt.figure()
    plt.imshow(Image.open(os.path.join(outdir, "03_tiles_bbox_overlay_RED.png")))
    plt.title("Tile segmentation (RED bounding boxes)")
    plt.axis("off")
    plt.savefig(os.path.join(outdir, "plot_04_tiles_bbox.png"), bbox_inches="tight")

    plt.figure()
    plt.imshow(Image.open(os.path.join(outdir, "04_tiles_class_overlay.png")))
    plt.title("Tile classification overlay (bright / pale / bluish)")
    plt.axis("off")
    plt.savefig(os.path.join(outdir, "plot_05_tiles_class.png"), bbox_inches="tight")

    # Distributions
    if len(df) > 0:
        plt.figure()
        plt.hist(df["Hmean"].values, bins=40)
        plt.title("Histogram of tile mean hue (0..255)")
        plt.xlabel("Hmean"); plt.ylabel("Count")
        plt.savefig(os.path.join(outdir, "plot_06_hist_Hmean.png"), bbox_inches="tight")

        plt.figure()
        plt.scatter(df["Smean"].values, df["intensity"].values, s=8)
        plt.title("Fluorescence intensity vs Smean")
        plt.xlabel("Smean"); plt.ylabel("Intensity (0..1)")
        plt.savefig(os.path.join(outdir, "plot_07_scatter_S_vs_intensity.png"), bbox_inches="tight")

        plt.figure()
        plt.scatter(df["Vmean"].values, df["intensity"].values, s=8)
        plt.title("Fluorescence intensity vs Vmean")
        plt.xlabel("Vmean"); plt.ylabel("Intensity (0..1)")
        plt.savefig(os.path.join(outdir, "plot_08_scatter_V_vs_intensity.png"), bbox_inches="tight")

    print(f"Done. Outputs in: {outdir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tile-wise green tesserae fluorescence simulator with debug outputs.")
    parser.add_argument("--src", required=True, help="Path to input mosaic image")
    parser.add_argument("--out", default="uranium_pipeline_out", help="Output directory")
    args = parser.parse_args()
    main(args.src, args.out)