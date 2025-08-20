import os
import time
from pathlib import Path

# --- IMPORTANT ---
# This script assumes it is in the same directory as your quality_checks.py file.
try:
    import quality_checks
except ImportError:
    print("ERROR: Could not find 'quality_checks.py'.")
    print("Please make sure this test script is in the same folder as your main project files.")
    exit()

# --- CONFIGURATION ---
TEMP_VIDEO_DIR = Path("temp_videos")

def main():
    """
    Finds all videos in the temp_videos directory and runs a suite of
    quality check functions on each one, printing the results and execution time.
    """
    print("--- Starting Video Quality Check Performance Test ---")
    print("Note: The first run of each AI model includes a one-time loading cost from disk.")
    print("Subsequent runs on other videos will show the true inference speed.")

    if not TEMP_VIDEO_DIR.is_dir():
        print(f"\nERROR: The directory '{TEMP_VIDEO_DIR}' was not found.")
        print("Please create it and add some sample videos to test.")
        return

    video_files = list(TEMP_VIDEO_DIR.glob('*.mp4'))
    if not video_files:
        print(f"\nNo .mp4 videos found in the '{TEMP_VIDEO_DIR}' directory.")
        return

    print(f"Found {len(video_files)} videos to test.\n")

    # Loop through each video file
    for video_path in video_files:
        print(f"\n{'='*20} Timing Video: {video_path.name} {'='*20}")

        # --- Define the functions to test ---
        functions_to_test = [
            
            # --- Historical Functions for Comparison ---
            {"name": "Hybrid (Shape + OCR)", "func": "check_for_text_hybrid_final"},
            {"name": "Final Pre-Processed OCR", "func": "check_for_text_final"},
            {"name": "Multi-Zone OCR", "func": "check_for_text_ocr_multi_zone"}

        ]

        for test in functions_to_test:
            func_name = test["func"]
            display_name = test["name"]
            
            try:
                if hasattr(quality_checks, func_name):
                    test_function = getattr(quality_checks, func_name)
                    
                    # --- TIMING LOGIC ---
                    start_time = time.perf_counter()
                    result = test_function(str(video_path))
                    end_time = time.perf_counter()
                    duration = end_time - start_time
                    # --- END TIMING LOGIC ---
                    
                    # Print the result along with the time it took
                    print(f"  - {display_name}: {result} (took {duration:.4f} seconds)")
                else:
                    pass # Silently skip functions that don't exist in the current quality_checks.py
            except Exception as e:
                print(f"  - {display_name}: ERROR - {e}")

    print(f"\n{'='*20} Performance Test Finished {'='*20}")


if __name__ == "__main__":
    main()