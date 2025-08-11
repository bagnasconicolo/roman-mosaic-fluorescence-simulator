#!/usr/bin/env python3
"""
Demo script to run the fluorescence simulator on the example image.
This generates the same results as shown in examples/sample_results/
"""

import os
import sys
import subprocess

def main():
    # Check if we're in the right directory
    if not os.path.exists('mosaic_fluorescence_gif.py'):
        print("Error: Please run this script from the repository root directory")
        sys.exit(1)
    
    # Check if example image exists
    example_image = 'examples/input_images/kaliumkalziumglas01w.jpg'
    if not os.path.exists(example_image):
        print(f"Error: Example image not found at {example_image}")
        sys.exit(1)
    
    # Create output directory
    output_dir = 'demo_output'
    os.makedirs(output_dir, exist_ok=True)
    
    print("Running Roman Mosaic Fluorescence Simulator demo...")
    print(f"Input: {example_image}")
    print(f"Output: {output_dir}/")
    print()
    
    # Run the simulator
    try:
        cmd = [sys.executable, 'mosaic_fluorescence_gif.py', 
               '--src', example_image, '--out', output_dir]
        subprocess.run(cmd, check=True)
        
        print()
        print("✅ Demo completed successfully!")
        print(f"Check the '{output_dir}/' folder for results including:")
        print("  • Static fluorescence images")
        print("  • Animated GIF files")
        print("  • Analysis plots and data")
        print()
        print("Compare with the reference results in examples/sample_results/")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running simulator: {e}")
        print("Make sure you have installed the requirements:")
        print("  pip install -r requirements.txt")
        sys.exit(1)
    except FileNotFoundError:
        print("❌ Python not found. Make sure Python is installed and in your PATH.")
        sys.exit(1)

if __name__ == "__main__":
    main()
