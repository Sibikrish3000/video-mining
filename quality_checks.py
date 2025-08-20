from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO
import easyocr

OCR_READER_CJK = None
OCR_READER_DEVANAGARI = None
OCR_READER = None


try:
    YOLO_MODEL = YOLO('yolov8n.pt')
    print("AI Model (YOLOv8) loaded successfully.")
except Exception as e:
    YOLO_MODEL = None
    print(f"CRITICAL: Failed to load YOLO model. Subject clarity check will be disabled. Error: {e}")

try:
    OCR_READER = easyocr.Reader(["en"])
    print("  - English OCR Reader loaded successfully.")
except Exception as e:
    OCR_READER = None
    print(f"  - CRITICAL: Failed to load EasyOCR model. Subtitle check will be disabled. Error: {e}")

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



def check_for_text_ocr_multi_zone(
    video_path: str,
    word_threshold: int = 2,
    frames_to_check: int = 3,
    debug_save: bool = True,
    debug_dir: str = "debug_frames"
) -> bool:
    """
    Uses OCR to detect significant text in the TOP, CENTER, and BOTTOM zones of the video.
    This method is color and brightness independent.
    """
    if not OCR_READER:
        print("  [QA Check] Skipping text check: OCR models failed to load.")
        return True

    print(f"  [QA Check] Analyzing for text with OCR in all zones (Threshold: >{word_threshold} words)...")
    try:
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames < frames_to_check: return True

        frame_indices = np.linspace(0, total_frames - 1, frames_to_check, dtype=int)

        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret: continue

            h, w, _ = frame.shape
            
            # --- Define Top, Center, and Bottom Analysis Zones ---
            top_zone = frame[0:int(h*0.30), :]
            center_zone = frame[int(h*0.30):int(h*0.70), :]
            bottom_zone = frame[int(h*0.70):h, :]
            
            zones = {
                "TOP": {"frame": top_zone, "offset": 0},
                "CENTER": {"frame": center_zone, "offset": int(h*0.30)},
                "BOTTOM": {"frame": bottom_zone, "offset": int(h*0.70)}
            }
            
            for zone_name, zone_data in zones.items():
                # Run both OCR readers on the zone
                ocr_results = OCR_READER.readtext(zone_data["frame"])

                if len(ocr_results) >= word_threshold:
                    detected_text = " ".join([res[1] for res in ocr_results])
                    print(f"  [QA FAIL] Excessive text found in {zone_name} zone. Text: '{detected_text}'")
                    
                    if debug_save:
                        # Draw boxes on the original full frame for context
                        for (bbox, text, prob) in ocr_results:
                            (tl, tr, br, bl) = bbox
                            # Add the zone's y-offset to draw the box in the correct position
                            offset = zone_data["offset"]
                            tl = (int(tl[0]), int(tl[1]) + offset)
                            br = (int(br[0]), int(br[1]) + offset)
                            cv2.rectangle(frame, tl, br, (0, 0, 255), 3)
                        
                        debug_filename = f"{Path(video_path).stem}_FAIL_OCR_{zone_name}.jpg"
                        debug_filepath = Path(debug_dir) / debug_filename
                        cv2.imwrite(str(debug_filepath), frame)
                        print(f"  [Debug] Saved OCR failure analysis frame to: {debug_filepath}")

                    cap.release()
                    return False

        print("  [QA PASS] No significant text detected in any zone.")
        cap.release()
        return True
    except Exception as e:
        print(f"  [QA ERROR] OCR check failed: {e}. Passing by default.")
        return True
    
def check_for_text_final(
    video_path: str,
    word_threshold: int = 2, # Tuned to be slightly more lenient
    frames_to_check: int = 5,
    debug_save: bool = True,
    debug_dir: str = "debug_frames"
) -> bool:
    """
    Uses advanced image pre-processing (CLAHE) before running multi-lingual OCR
    on the top, center, and bottom zones of video frames for maximum accuracy.
    """
    if not OCR_READER:
        print("  [QA Check] Skipping text check: OCR model not loaded.")
        return True

    print(f"  [QA Check] Analyzing for text with Pre-Processing + OCR (Threshold: >{word_threshold} words)...")
    try:
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames < frames_to_check: return True

        frame_indices = np.linspace(0, total_frames - 1, frames_to_check, dtype=int)

        def _preprocess_for_ocr(frame_zone):
            """Applies Grayscale and CLAHE to enhance text for OCR."""
            # 1. Convert to grayscale
            gray = cv2.cvtColor(frame_zone, cv2.COLOR_BGR2GRAY)
            # 2. Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced_gray = clahe.apply(gray)
            return enhanced_gray

        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret: continue

            h, w, _ = frame.shape
            zones = {
                "TOP": {"frame": frame[0:int(h*0.33), :], "offset": 0},
                "CENTER": {"frame": frame[int(h*0.33):int(h*0.67), :], "offset": int(h*0.33)},
                "BOTTOM": {"frame": frame[int(h*0.67):h, :], "offset": int(h*0.67)}
            }
            
            for zone_name, zone_data in zones.items():
                # Pre-process the zone to make text clearer
                processed_zone = _preprocess_for_ocr(zone_data["frame"])
                
                # Run OCR on the enhanced image
                ocr_results = OCR_READER.readtext(processed_zone)

                if len(ocr_results) >= word_threshold:
                    detected_text = " ".join([res[1] for res in ocr_results])
                    print(f"  [QA FAIL] Excessive text found in {zone_name} zone. Text: '{detected_text}'")
                    
                    if debug_save:
                        for (bbox, text, prob) in ocr_results:
                            offset = zone_data["offset"]
                            tl = (int(bbox[0][0]), int(bbox[0][1]) + offset)
                            br = (int(bbox[2][0]), int(bbox[2][1]) + offset)
                            cv2.rectangle(frame, tl, br, (0, 0, 255), 3)
                        
                        debug_filename = f"{Path(video_path).stem}_FAIL_FINAL_OCR_{zone_name}.jpg"
                        debug_filepath = Path(debug_dir) / debug_filename
                        cv2.imwrite(str(debug_filepath), frame)
                        print(f"  [Debug] Saved OCR failure analysis frame to: {debug_filepath}")

                    cap.release()
                    return False

        print("  [QA PASS] No significant text detected in any zone.")
        cap.release()
        return True
    except Exception as e:
        print(f"  [QA ERROR] Final OCR check failed: {e}. Passing by default.")
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
def check_for_text_hybrid_final(
    video_path: str,
    shape_contour_threshold: int = 15,     # Threshold for the FAST shape detector
    ocr_word_threshold: int = 2,           # Threshold for the SMART OCR confirmation
    frames_to_check: int = 5,
    debug_save: bool = False,
    debug_dir: str = "debug_frames"
) -> bool:
    """
    Uses a fast, shape-based text region detector as a pre-screener, then runs
    accurate OCR only on suspicious frames for the most robust detection.
    """
    if not OCR_READER:
        print("  [QA Check] Skipping text check: OCR models not loaded.")
        return True

    print(f"  [QA Check] Analyzing for text with Shape Detection + OCR...")
    try:
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames < frames_to_check: return True

        frame_indices = np.linspace(0, total_frames - 1, frames_to_check, dtype=int)

        def _check_zone_for_shapes(zone_frame):
            """Stage 1: Fast, shape-based text region detection."""
            # Pre-processing for contour detection
            gray = cv2.cvtColor(zone_frame, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edged = cv2.Canny(blurred, 30, 150)
            
            # Find contours (outlines of shapes)
            contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            text_like_contours = 0
            for c in contours:
                # Get the bounding box of the contour
                (x, y, w, h) = cv2.boundingRect(c)
                
                # Filter based on geometric properties of letters
                # Letters are usually not too wide or too tall, and have a minimum size
                aspect_ratio = w / float(h)
                if (h > 8 and w > 3) and (h < 100 and w < 100) and (aspect_ratio < 3.0 and aspect_ratio > 0.1):
                    text_like_contours += 1
            
            return text_like_contours > shape_contour_threshold

        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret: continue

            h, w, _ = frame.shape
            zones = {
                "TOP": {"frame": frame[0:int(h*0.33), :], "offset": 0},
                "CENTER": {"frame": frame[int(h*0.33):int(h*0.67), :], "offset": int(h*0.33)},
                "BOTTOM": {"frame": frame[int(h*0.67):h, :], "offset": int(h*0.67)}
            }
            
            for zone_name, zone_data in zones.items():
                # --- STAGE 1: Check for text-like shapes ---
                is_suspicious = _check_zone_for_shapes(zone_data["frame"])
                
                if is_suspicious:
                    print(f"  - Suspicious shapes found in {zone_name} zone. Running OCR for confirmation...")
                    # --- STAGE 2: Confirm with OCR ---
                    ocr_results = OCR_READER.readtext(zone_data["frame"])

                    print(f"  - OCR found {len(ocr_results)} {ocr_word_threshold} words in {zone_name} zone.")
                    if len(ocr_results) >= ocr_word_threshold:
                        detected_text = " ".join([res[1] for res in ocr_results])
                        print(f"  [QA FAIL] OCR confirmed excessive text in {zone_name} zone. Text: '{detected_text}'")
                        
                        if debug_save:
                            # Debug drawing logic remains the same
                            for (bbox, text, prob) in ocr_results:
                                offset = zone_data["offset"]
                                tl = (int(bbox[0][0]), int(bbox[0][1]) + offset)
                                br = (int(bbox[2][0]), int(bbox[2][1]) + offset)
                                cv2.rectangle(frame, tl, br, (0, 0, 255), 3)
                            
                            debug_filename = f"{Path(video_path).stem}_FAIL_HYBRID_OCR_{zone_name}.jpg"
                            debug_filepath = Path(debug_dir) / debug_filename
                            cv2.imwrite(str(debug_filepath), frame)
                            print(f"  [Debug] Saved failure analysis frame to: {debug_filepath}")

                        cap.release()
                        return False

        print("  [QA PASS] No significant text detected in any zone.")
        cap.release()
        return True
    except Exception as e:
        print(f"  [QA ERROR] Hybrid OCR check failed: {e}. Passing by default.")
        return True
    
if __name__ == "__main__":
    video_path = "temp_videos/video1.mp4"

    result = check_for_text_ocr_multi_zone(video_path)
    print(f"Text OCR multi-zone check result: {result}")
    result = check_for_text_final(video_path)
    print(f"Text final check result: {result}")
    result = check_for_text_hybrid_final(video_path)
    print(f"Text hybrid final check result: {result}")
