"""
Converts a directory of PNG images (potentially 16-bit grayscale) to JPEG format,
applying resizing, windowing (for 16-bit sources), and high-quality settings.
Uses multiprocessing for parallel execution.
"""

import multiprocessing as mp
import os
import argparse
import time
from pprint import pprint

from medvqa.datasets.image_processing import resize_image

# ==============================================================================
# Argument Parsing and Main Execution Logic
# ==============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Convert PNG images (including 16-bit) to JPEG with resizing and windowing.")
    parser.add_argument('--source_folder', type=str, required=True,
                        help='Folder containing the source PNG images.')
    parser.add_argument('--target_folder', type=str, required=True,
                        help='Folder where the output JPEG images will be saved.')
    parser.add_argument('--num_workers', type=int, default=mp.cpu_count(),
                        help='Number of parallel processes to use (default: number of CPU cores).')
    parser.add_argument('--jpeg_quality', type=int, default=95, choices=range(0, 101), metavar='[0-100]',
                        help='Quality for the output JPEG images (default: 95).')
    parser.add_argument('--p_min', type=float, default=1.0,
                        help='Lower percentile for 16-bit windowing (default: 1.0).')
    parser.add_argument('--p_max', type=float, default=99.0,
                        help='Upper percentile for 16-bit windowing (default: 99.0).')

    # --- Size arguments ---
    group_size = parser.add_argument_group('Resizing Options')
    group_mode = group_size.add_mutually_exclusive_group(required=True)
    group_mode.add_argument('--keep_aspect_ratio', action='store_true',
                             help='Maintain aspect ratio. Requires --size.')
    group_mode.add_argument('--fixed_size', action='store_true',
                             help='Resize to exact dimensions. Requires --width and --height.')

    group_size.add_argument('--size', type=int,
                             help='Target size for the *smallest* side (used with --keep_aspect_ratio).')
    group_size.add_argument('--width', type=int,
                             help='Target width (used with --fixed_size).')
    group_size.add_argument('--height', type=int,
                             help='Target height (used with --fixed_size).')

    args = parser.parse_args()

    # --- Validate size arguments ---
    if args.keep_aspect_ratio:
        if args.size is None or args.size <= 0:
            parser.error("--size must be a positive integer when --keep_aspect_ratio is used.")
        if args.width is not None or args.height is not None:
            parser.error("--width and --height cannot be used with --keep_aspect_ratio.")
        args.parsed_new_size = args.size
        args.parsed_keep_aspect_ratio = True
    elif args.fixed_size:
        if args.width is None or args.width <= 0 or args.height is None or args.height <= 0:
            parser.error("--width and --height must be positive integers when --fixed_size is used.")
        if args.size is not None:
            parser.error("--size cannot be used with --fixed_size.")
        args.parsed_new_size = (args.width, args.height)
        args.parsed_keep_aspect_ratio = False
    # The mutually exclusive group ensures one mode is chosen

    return args

def worker_task(src_path, tgt_path, new_size_arg, keep_aspect_ratio_arg, quality_arg, p_min_arg, p_max_arg):
    """Wrapper function for multiprocessing to call the main resize function."""
    # Add any setup needed per worker, if necessary
    resize_image(
        src_image_path=src_path,
        tgt_image_path=tgt_path,
        new_size=new_size_arg,
        keep_aspect_ratio=keep_aspect_ratio_arg,
        jpeg_quality=quality_arg,
        p_min=p_min_arg,
        p_max=p_max_arg,
        # Interpolation args use defaults from resize_image
    )


if __name__ == '__main__':
    args = parse_args()

    print("--- Configuration ---")
    print(f"Source Folder:      {args.source_folder}")
    print(f"Target Folder:      {args.target_folder}")
    print(f"Keep Aspect Ratio:  {args.parsed_keep_aspect_ratio}")
    print(f"Target Size Param:  {args.parsed_new_size}")
    print(f"JPEG Quality:       {args.jpeg_quality}")
    print(f"16-bit Windowing:   {args.p_min}% - {args.p_max}%")
    print(f"Num Workers:        {args.num_workers}")
    print("---------------------\n")

    os.makedirs(args.target_folder, exist_ok=True)

    # --- Prepare file lists ---
    source_image_filepaths = []
    target_image_filepaths = []
    skipped_files = []

    print("Scanning source folder...")
    try:
        for filename in os.listdir(args.source_folder):
            if filename.lower().endswith('.png'):
                base, _ = os.path.splitext(filename)
                source_path = os.path.join(args.source_folder, filename)
                target_path = os.path.join(args.target_folder, f"{base}.jpg") # Ensure .jpg extension
                source_image_filepaths.append(source_path)
                target_image_filepaths.append(target_path)
            else:
                skipped_files.append(filename)
    except FileNotFoundError:
        print(f"Error: Source folder not found: {args.source_folder}")
        exit(1)
    except Exception as e:
        print(f"Error scanning source folder: {e}")
        exit(1)

    if not source_image_filepaths:
        print("Error: No PNG files found in the source folder.")
        exit(0)

    print(f"Found {len(source_image_filepaths)} PNG images to process.")
    if skipped_files:
        print(f"Skipped {len(skipped_files)} non-PNG files (showing first 5): {skipped_files[:5]}{'...' if len(skipped_files)>5 else ''}")

    print("\n--- Sample Paths ---")
    pprint(source_image_filepaths[:5])
    print("...")
    pprint(target_image_filepaths[:5])
    print("--------------------\n")

    # --- Prepare arguments for multiprocessing ---
    tasks = [
        (src, tgt, args.parsed_new_size, args.parsed_keep_aspect_ratio, args.jpeg_quality, args.p_min, args.p_max)
        for src, tgt in zip(source_image_filepaths, target_image_filepaths)
    ]

    # --- Run multiprocessing ---
    print(f"Starting conversion with {args.num_workers} workers...")
    start_time = time.time()
    try:
        with mp.Pool(args.num_workers) as pool:
            pool.starmap(worker_task, tasks)
    except Exception as e:
        print(f"\n--- Multiprocessing pool encountered an error: {e} ---")
        # Potential errors during pool setup or task distribution
        # Individual task errors are handled within worker_task/resize_image_enhanced

    end_time = time.time()
    print(f"\n--- Processing Complete ---")
    print(f"Finished in {end_time - start_time:.2f} seconds.")