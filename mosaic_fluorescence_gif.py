#!/usr/bin/env python3
"""
End-to-end mosaic fluorescence + animated GIFs.

Outputs (in --out):
  01/02... candidate masks
  03_tiles_bbox_overlay_RED.png
  04_tiles_class_overlay.png
  10_fluorescence_black.png
  11_fluorescence_on_mosaic.png
  fluorescence_pulse_black.gif
  fluorescence_pulse_on_mosaic.gif
  tiles_classification.csv
"""

import os, argparse, math
import numpy as np
from PIL import Image, ImageFilter, ImageDraw
import matplotlib.pyplot as plt
import pandas as pd

# ----------------------- Tunable parameters -----------------------
LONG_SIDE = 2200            # working resolution (long side, px)
H_MIN, H_MAX = 35, 110      # greenish gate (Pillow HSV hue 0..255)
S_MIN, V_MIN = 40, 40
TOL_H, TOL_S, TOL_V = 12, 90, 90   # region-growing tolerances
MIN_AREA, MAX_AREA = 25, 20000     # tile component area limits (px)
TINT = np.array([0.0, 1.0, 0.55], dtype=np.float32)  # ~525 nm green
BLOOM = [(0,1.3),(2,1.0),(5,0.8),(12,0.6),(24,0.45)] # radius, weight

# GIF animation settings
NUM_FRAMES = 24
FRAME_MS   = 80
MAX_GIF_LONG = 1200         # downscale for lighter GIFs
FLICKER_SIGMA = 0.03        # per-frame spatial flicker

# ----------------------- Utility functions -----------------------
def load_and_rescale(path, long_side=LONG_SIDE):
    im0 = Image.open(path).convert("RGB")
    w0, h0 = im0.size
    s = long_side / max(w0, h0)
    return im0.resize((int(w0*s), int(h0*s)), Image.Resampling.LANCZOS)

def pil_to_hsv_arrays(img):
    hsv = img.convert("HSV")
    a = np.array(hsv).astype(np.int16)
    return a[:,:,0], a[:,:,1], a[:,:,2]

def closing(mask_bool, size=3):
    img = Image.fromarray((mask_bool.astype(np.uint8)*255))
    img = img.filter(ImageFilter.MaxFilter(size=size))  # dilate
    img = img.filter(ImageFilter.MinFilter(size=size))  # erode
    return np.array(img) > 0

def gaussian_blur_gray(arr, radius):
    img = Image.fromarray(np.uint8(np.clip(arr*255,0,255)))
    img = img.filter(ImageFilter.GaussianBlur(radius=radius))
    return np.array(img).astype(np.float32)/255.0

def grow_region(seed_y, seed_x, H, S, V, cand_mask):
    from collections import deque
    Ht, Wt = cand_mask.shape
    Hs, Ss, Vs = int(H[seed_y,seed_x]), int(S[seed_y,seed_x]), int(V[seed_y,seed_x])
    q = deque([(seed_y, seed_x)])
    seen = set()
    coords = []
    hsum=ssum=vsum=0
    while q:
        y,x = q.pop()
        if not (0<=y<Ht and 0<=x<Wt): continue
        if (y,x) in seen: continue
        seen.add((y,x))
        if not cand_mask[y,x]: continue
        if (abs(int(H[y,x])-Hs)<=TOL_H and
            abs(int(S[y,x])-Ss)<=TOL_S and
            abs(int(V[y,x])-Vs)<=TOL_V):
            coords.append((y,x))
            hsum += int(H[y,x]); ssum += int(S[y,x]); vsum += int(V[y,x])
            q.extend(((y-1,x),(y+1,x),(y,x-1),(y,x+1)))
    if not coords: return None
    ys=[c[0] for c in coords]; xs=[c[1] for c in coords]
    y0,y1=min(ys),max(ys); x0,x1=min(xs),max(xs)
    mask_local = np.zeros((y1-y0+1, x1-x0+1), dtype=bool)
    for yy,xx in coords: mask_local[yy-y0, xx-x0]=True
    return dict(mask=mask_local, bbox=(x0,y0,x1,y1),
                Hmean=hsum/len(coords), Smean=ssum/len(coords),
                Vmean=vsum/len(coords), area=len(coords))

def classify_tile(Hm, Sm, Vm):
    if Hm >= 90:  # cyan/blueish -> none
        return 0.0, "bluish"
    hue_w = 0.5 if Hm < 40 else (1.0 if Hm <= 85 else 0.2)
    wS = max(0.0, min(1.0, (Sm - 70) / (180 - 70)))
    wV = max(0.0, min(1.0, (Vm - 80) / (220 - 80)))
    intensity = float(max(0.0, min(1.0, hue_w*(0.6*wS+0.4*wV))))
    label = "bright-green" if intensity>0.65 else ("pale-green" if intensity>0.15 else "very-pale")
    return intensity, label

# ----------------------- Pipeline -----------------------
def run_pipeline(src_path, outdir):
    os.makedirs(outdir, exist_ok=True)
    im = load_and_rescale(src_path)
    w,h = im.size
    H,S,V = pil_to_hsv_arrays(im)

    cand = (H>=H_MIN)&(H<=H_MAX)&(S>=S_MIN)&(V>=V_MIN)
    Image.fromarray((cand.astype(np.uint8)*255)).save(os.path.join(outdir,"01_candidate_mask.png"))
    cand2 = closing(cand,3)
    Image.fromarray((cand2.astype(np.uint8)*255)).save(os.path.join(outdir,"02_candidate_mask_closed.png"))

    # segment
    visited = np.zeros((h,w), dtype=bool)
    tiles=[]
    for yy in range(h):
        for xx in range(w):
            if cand2[yy,xx] and not visited[yy,xx]:
                t = grow_region(yy,xx,H,S,V,cand2)
                if not t: continue
                x0,y0,x1,y1 = t["bbox"]
                visited[y0:y1+1, x0:x1+1] |= t["mask"]
                if MIN_AREA <= t["area"] <= MAX_AREA:
                    tiles.append(t)

    # red bbox overlay
    ov = im.copy().convert("RGBA")
    dr = ImageDraw.Draw(ov)
    for t in tiles:
        x0,y0,x1,y1 = t["bbox"]
        dr.rectangle([x0,y0,x1,y1], outline=(255,0,0,170), width=2)
    ov.save(os.path.join(outdir,"03_tiles_bbox_overlay_RED.png"))

    # classify and table
    rows=[]
    for i,t in enumerate(tiles):
        inten,label = classify_tile(t["Hmean"], t["Smean"], t["Vmean"])
        t["intensity"]=inten; t["label"]=label
        x0,y0,x1,y1 = t["bbox"]
        rows.append(dict(tile_id=i,x0=x0,y0=y0,x1=x1,y1=y1,area_px=t["area"],
                         Hmean=t["Hmean"],Smean=t["Smean"],Vmean=t["Vmean"],
                         label=label,intensity=inten))
    pd.DataFrame(rows).to_csv(os.path.join(outdir,"tiles_classification.csv"), index=False)

    # class overlay (for visual QA)
    cls = im.copy().convert("RGBA")
    dr = ImageDraw.Draw(cls)
    for t in tiles:
        x0,y0,x1,y1 = t["bbox"]
        if t["label"]=="bright-green": col=(0,255,0,150)
        elif t["label"] in ("pale-green","very-pale"): col=(180,255,180,120)
        else: col=(80,80,255,150)
        dr.rectangle([x0,y0,x1,y1], outline=col, width=2)
    cls.save(os.path.join(outdir,"04_tiles_class_overlay.png"))

    # emission buffer
    emission = np.zeros((h,w,3), dtype=np.float32)
    for t in tiles:
        if t["intensity"]<=0: continue
        x0,y0,x1,y1 = t["bbox"]
        a = Image.fromarray((t["mask"].astype(np.uint8)*255))
        a = a.filter(ImageFilter.MaxFilter(size=3))
        a = a.filter(ImageFilter.GaussianBlur(radius=1.2))
        a = np.array(a).astype(np.float32)/255.0
        color = TINT * (0.55 + 0.45*t["intensity"])
        sl_y = slice(y0,y1+1); sl_x = slice(x0,x1+1)
        emission[sl_y,sl_x,:] += (a[...,None]*color[None,None,:]) * t["intensity"]

    emission = np.clip(emission,0,1)
    # bloom
    glow = np.zeros_like(emission)
    for c in range(3):
        ch = emission[:,:,c]
        acc = np.zeros_like(ch)
        for r,wgt in BLOOM:
            acc += (gaussian_blur_gray(ch,r) if r>0 else ch)*wgt
        glow[:,:,c]=acc
    gmax = glow.max() if glow.max()>1e-6 else 1.0
    glow = np.clip(glow/gmax, 0,1)

    # base scene (dark)
    im_np = np.array(im).astype(np.float32)/255.0
    base_dark = np.clip(im_np*0.18, 0,1)

    # save static renders
    Image.fromarray(np.uint8(glow*255)).save(os.path.join(outdir,"10_fluorescence_black.png"))
    comp = np.clip(base_dark + glow*1.2, 0,1)
    Image.fromarray(np.uint8(comp*255)).save(os.path.join(outdir,"11_fluorescence_on_mosaic.png"))

    # simple debug plots
    plt.figure(); plt.imshow(im); plt.axis("off"); plt.title("Original (rescaled)")
    plt.savefig(os.path.join(outdir,"plot_01_original.png"), bbox_inches="tight")
    plt.figure(); plt.imshow(cand.astype(np.uint8)*255); plt.axis("off"); plt.title("Candidate (raw)")
    plt.savefig(os.path.join(outdir,"plot_02_candidate_raw.png"), bbox_inches="tight")
    plt.figure(); plt.imshow(cand2.astype(np.uint8)*255); plt.axis("off"); plt.title("Candidate (closed)")
    plt.savefig(os.path.join(outdir,"plot_03_candidate_closed.png"), bbox_inches="tight")

    return im_np, base_dark, glow  # arrays in 0..1, shape (h,w,[3])

# ----------------------- GIF builder -----------------------
def downscale_for_gif(imgs, max_long=MAX_GIF_LONG):
    out=[]
    for im in imgs:
        w,h = im.size
        s = min(1.0, max_long/max(w,h))
        out.append(im.resize((int(w*s), int(h*s)), Image.Resampling.LANCZOS) if s<1.0 else im)
    return out

def to_gif_palette_seq(frames):
    first = frames[0].convert("P", palette=Image.Palette.ADAPTIVE, colors=256)
    pal = first.getpalette()
    seq=[first]
    for im in frames[1:]:
        p = im.convert("P"); p.putpalette(pal); seq.append(p)
    return seq

def build_gifs(base_dark, glow, outdir, num_frames=NUM_FRAMES, frame_ms=FRAME_MS, flicker_sigma=FLICKER_SIGMA):
    h,w,_ = glow.shape
    frames_on=[]; frames_black=[]
    glow = np.clip(glow,0,1).astype(np.float32)
    base_dark = np.clip(base_dark,0,1).astype(np.float32)

    for t in range(num_frames):
        phase = 2*math.pi*(t/num_frames)
        gain = 0.15 + 1.15*max(0.0, math.sin(phase))   # no negative light
        rng = np.random.default_rng(1234+t)
        flicker = rng.normal(1.0, flicker_sigma, size=(h,w)).astype(np.float32)
        flicker = np.clip(flicker, 0.92, 1.08)[...,None]

        Em = np.clip(glow * gain * flicker, 0,1)
        on = np.clip(base_dark + Em, 0,1)
        frames_on.append(Image.fromarray(np.uint8(on*255)))
        frames_black.append(Image.fromarray(np.uint8(Em*255)))

    frames_on  = downscale_for_gif(frames_on)
    frames_black = downscale_for_gif(frames_black)

    gif_on_seq  = to_gif_palette_seq(frames_on)
    gif_blk_seq = to_gif_palette_seq(frames_black)
    p_on  = os.path.join(outdir, "fluorescence_pulse_on_mosaic.gif")
    p_blk = os.path.join(outdir, "fluorescence_pulse_black.gif")
    gif_on_seq[0].save(p_on,  save_all=True, append_images=gif_on_seq[1:], duration=frame_ms, loop=0, disposal=2)
    gif_blk_seq[0].save(p_blk, save_all=True, append_images=gif_blk_seq[1:], duration=frame_ms, loop=0, disposal=2)
    print("GIFs saved:", p_on, "and", p_blk)

# ----------------------- Main -----------------------
def main(src, outdir):
    _, base_dark, glow = run_pipeline(src, outdir)
    build_gifs(base_dark, glow, outdir)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="Path to mosaic image")
    ap.add_argument("--out", default="uranium_pipeline_out", help="Output directory")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    main(args.src, args.out)