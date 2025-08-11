# quality_checks.py

import cv2
import numpy as np
from ultralytics import YOLO

# Load the YOLO model once to be reused.
# It will be downloaded automatically on the first run.
try:
    YOLO_MODEL = YOLO('yolov8n.pt')
    print("AI Model (YOLOv8) loaded successfully.")
except Exception as e:
    YOLO_MODEL = None
    print(f"CRITICAL: Failed to load YOLO model. Subject clarity check will be disabled. Error: {e}")


# --- NEW VERIFICATION FUNCTION ---
def verify_video_file(video_path: str) -> bool:
    """
    Quickly checks if a video file is valid and playable by trying to open it and read a frame.
    Returns True if valid, False if corrupted or unplayable.
    """
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"  [QA FAIL] Verification failed: Could not open video file.")
            return False
        
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            print(f"  [QA FAIL] Verification failed: Could not read a frame from the video.")
            return False
        
        # print("  [QA PASS] Video file verified successfully.")
        return True
    except Exception as e:
        print(f"  [QA ERROR] Verification check raised an exception: {e}")
        return False


def check_for_hardcoded_subtitles(video_path: str, brightness_thresh=215, area_thresh=0.1554, frames_to_check=5) -> bool:
    """
    Checks for large, hardcoded subtitles in the video.
    Returns True if the video is clean, False if subtitles are likely present.
    """
    print(f"  [QA Check] Analyzing for hardcoded subtitles...")
    try:
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames < frames_to_check: return True

        # Check a few frames spread across the video
        frame_indices = np.linspace(0, total_frames - 1, frames_to_check, dtype=int)
        
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret: continue

            # --- Analysis Zone ---
            # Focus on the bottom 40% of the frame where subtitles usually are.
            h, w, _ = frame.shape
            subtitle_zone_y_start = int(h * 0.60)
            subtitle_zone = frame[subtitle_zone_y_start:, :]
            
            # Convert to grayscale to focus on brightness
            gray_zone = cv2.cvtColor(subtitle_zone, cv2.COLOR_BGR2GRAY)
            
            # Apply a high threshold to isolate only very bright pixels (like text)
            _, threshold_zone = cv2.threshold(gray_zone, brightness_thresh, 255, cv2.THRESH_BINARY)
            
            # --- Detection Logic ---
            # Calculate the percentage of white pixels in the zone
            zone_area = threshold_zone.shape[0] * threshold_zone.shape[1]
            white_pixels = cv2.countNonZero(threshold_zone)
            white_pixel_percentage = white_pixels / zone_area
            
            # If the percentage of bright pixels is too high, it's likely text
            if white_pixel_percentage > area_thresh:
                print(f"  [QA FAIL] Potential subtitles found. White pixel ratio: {white_pixel_percentage:.4f}")
                cap.release()
                return False

        print("  [QA PASS] No significant hardcoded subtitles detected.")
        cap.release()
        return True
    except Exception as e:
        print(f"  [QA ERROR] Subtitle check failed: {e}. Passing by default.")
        return True
def check_scene_cuts(video_path: str, cut_threshold: float = 2.5) -> bool:
    """
    Analyzes video for rapid scene changes.
    Returns True if the video is OK, False if it's likely over-edited.
    """
    print(f"  [QA Check] Analyzing scene cut frequency...")
    try:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0: return True

        cuts = 0
        ret, prev_frame = cap.read()
        if not ret: return True

        prev_hist = cv2.calcHist([prev_frame], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        cv2.normalize(prev_hist, prev_hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

        while True:
            ret, frame = cap.read()
            if not ret: break
            hist = cv2.calcHist([frame], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
            cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
            diff = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
            if diff < 0.7:
                cuts += 1
            prev_hist = hist

        duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
        cuts_per_second = cuts / duration if duration > 0 else 0
        cap.release()
        
        if cuts_per_second > cut_threshold:
            print(f"  [QA FAIL] Scene cuts too frequent ({cuts_per_second:.2f} per second).")
            return False
        else:
            print(f"  [QA PASS] Scene cut frequency is acceptable.")
            return True
    except Exception as e:
        print(f"  [QA ERROR] Scene cut check failed: {e}. Passing by default.")
        return True

def check_subject_clarity(video_path: str, min_person_area: float = 0.05, frames_to_check: int = 5) -> bool:
    """
    Analyzes video to ensure a person is a clear, primary subject.
    Returns True if the subject is clear, False otherwise.
    """
    if not YOLO_MODEL:
        print("[QA Check] Skipping subject clarity: YOLO model not available.")
        return True

    print(f"  [QA Check] Analyzing subject clarity...")
    try:
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames < frames_to_check: return True

        frame_indices = np.linspace(0, total_frames - 1, frames_to_check, dtype=int)
        
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret: continue

            frame_height, frame_width, _ = frame.shape
            frame_area = frame_height * frame_width
            results = YOLO_MODEL(frame, verbose=False)
            
            for result in results:
                for box in result.boxes:
                    if int(box.cls[0]) == 0:  # Class 0 is 'person'
                        xywh = box.xywh[0]
                        box_area = xywh[2] * xywh[3]
                        if (box_area / frame_area) > min_person_area:
                            print(f"  [QA PASS] Clear subject found.")
                            cap.release()
                            return True
        
        print(f"  [QA FAIL] No clear, close-up subject detected.")
        cap.release()
        return False
    except Exception as e:
        print(f"  [QA ERROR] Subject clarity check failed: {e}. Passing by default.")
        return True