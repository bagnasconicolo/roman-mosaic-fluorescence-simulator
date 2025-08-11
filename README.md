# Roman Mosaic Fluorescence Simulator

A Python tool for simulating green tesserae fluorescence in Roman mosaics, with support for both static images and animated GIFs.

## Features

- **Tile-wise fluorescence simulation**: Segments individual greenish tesserae using region-growing on HSV color space
- **Classification system**: Classifies tiles as bright-green (strong fluorescence), pale-green (weak), or bluish (none)
- **Static rendering**: Generates fluorescence images on black background and overlaid on the original mosaic
- **Animated GIFs**: Creates pulsing fluorescence animations with spatial flicker effects
- **Debug outputs**: Comprehensive visualization and analysis tools including histograms and scatter plots

## Requirements

- Python 3.7+
- pillow
- numpy
- matplotlib
- pandas

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/roman-mosaic-fluorescence-simulator.git
cd roman-mosaic-fluorescence-simulator
```

2. Create and activate a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install pillow numpy matplotlib pandas
```

## Usage

### Static Fluorescence Simulation

```bash
python mosaic_fluorescence.py --src path/to/image.jpg --out output_directory
```

### Animated GIF Generation

```bash
python mosaic_fluorescence_gif.py --src path/to/image.jpg --out output_directory
```

## Output Files

The simulator generates various output files:

- `01_candidate_mask.png` - Initial greenish pixel candidates
- `02_candidate_mask_closed.png` - Cleaned candidate mask
- `03_tiles_bbox_overlay_RED.png` - Tile segmentation visualization
- `04_tiles_class_overlay.png` - Classification overlay
- `10_fluorescence_black.png` - Fluorescence on black background
- `11_fluorescence_on_mosaic.png` - Fluorescence overlaid on original
- `fluorescence_pulse_*.gif` - Animated fluorescence (GIF version only)
- `tiles_classification.csv` - Per-tile analysis data
- `plot_*.png` - Debug visualizations

## Parameters

Key parameters can be adjusted in the script:

- `LONG_SIDE`: Working resolution (default: 2200px)
- `H_MIN, H_MAX`: HSV hue range for greenish candidates
- `S_MIN, V_MIN`: Minimum saturation and value thresholds
- `TOL_H, TOL_S, TOL_V`: Region-growing tolerances
- `MIN_AREA, MAX_AREA`: Tile size limits
- `TINT`: Fluorescence color (~525nm green)

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
