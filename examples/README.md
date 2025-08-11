# Example Results

This folder contains sample input and output files to demonstrate the capabilities of the Roman Mosaic Fluorescence Simulator.

## Input Image

**File**: `input_images/kaliumkalziumglas01w.jpg`
- A Roman mosaic with greenish tesserae containing potassium-calcium glass
- This type of glass is known to fluoresce under UV light due to manganese content

## Sample Results

The `sample_results/` folder contains the complete output from running:
```bash
python mosaic_fluorescence_gif.py --src examples/input_images/kaliumkalziumglas01w.jpg --out examples/sample_results
```

### Output Files Explained

#### Segmentation and Analysis
- **01_candidate_mask.png** - Initial detection of greenish pixels
- **02_candidate_mask_closed.png** - Cleaned mask after morphological closing
- **03_tiles_bbox_overlay_RED.png** - Red bounding boxes showing detected tiles
- **04_tiles_class_overlay.png** - Color-coded classification overlay
- **tiles_classification.csv** - Detailed per-tile analysis data

#### Fluorescence Renders
- **10_fluorescence_black.png** - Fluorescence on black background
- **11_fluorescence_on_mosaic.png** - Fluorescence overlaid on original image
- **fluorescence_pulse_black.gif** - Animated fluorescence on black (24 frames)
- **fluorescence_pulse_on_mosaic.gif** - Animated fluorescence on mosaic (24 frames)

#### Debug Visualizations
- **plot_01_original.png** - Original image (rescaled)
- **plot_02_candidate_raw.png** - Raw candidate mask
- **plot_03_candidate_closed.png** - Processed candidate mask

## Understanding the Results

### Tile Classification
The simulator classifies each detected tile into three categories:
- **Bright Green** (strong fluorescence) - Likely contains manganese-doped glass
- **Pale Green** (weak fluorescence) - May contain trace amounts of manganese
- **Bluish** (no fluorescence) - Different glass composition or other materials

### CSV Data
The `tiles_classification.csv` file contains detailed metrics for each tile:
- Position (x0, y0, x1, y1)
- Area in pixels
- Mean HSV values (Hmean, Smean, Vmean)
- Classification label
- Fluorescence intensity (0-1)

### Animation Features
The GIF animations demonstrate:
- Pulsing fluorescence intensity (sinusoidal modulation)
- Spatial flicker effects (realistic noise)
- Soft-edged emission from whole tiles
- Multi-scale bloom effects

## File Sizes
Note: The generated files can be quite large:
- Static images: ~1-8 MB each
- Animated GIFs: ~400KB - 11MB
- Total output: ~30MB for this example

## Usage Tips
1. Use the static images for documentation and analysis
2. Use the animated GIFs for presentations and demonstrations
3. Examine the CSV file for quantitative analysis
4. Adjust parameters in the script for different materials or conditions
