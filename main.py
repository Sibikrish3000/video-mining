# run_pipeline.py

import os
import shutil
import sqlite3
from datetime import datetime
from pathlib import Path
import isodate
import yt_dlp
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from dotenv import load_dotenv
import os
# Import our custom quality check functions
import quality_checks
import subprocess 
import cv2
import glob

# Load environment variables from .env file
load_dotenv()

# --- CONFIGURATION ---
# IMPORTANT: Fill these in
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
# Create a folder for pending reviews and put its ID here
PENDING_REVIEW_FOLDER_ID = os.getenv("PENDING_REVIEW_FOLDER_ID")
MAX_VIDEO_DURATION_FOR_SINGLE_CLIP = 45  # Videos longer than this will be chunked
MAX_SUCCESSFUL_CHUNKS_PER_VIDEO = 5 # Set the desired limit here
MAX_CONSECUTIVE_FAILURES = 7
OWNER_NAME = "sibi_krishnamoorthy"
# Files and Folders
CREDENTIALS_FILE = "credentials.json" # Downloaded from Google Cloud
TOKEN_FILE = "token.json"              # Will be created automatically
DB_FILE = "crawled_videos.db"
TEMP_VIDEO_DIR = "temp_videos"
SCOPES = ['https://www.googleapis.com/auth/youtube.readonly', 'https://www.googleapis.com/auth/drive.file']

COMPRESSION_THRESHOLD_MB = 10
FILE_SIZE_THRESHOLD = COMPRESSION_THRESHOLD_MB * 1024 * 1024

# CRF (Constant Rate Factor) for H.264. 23 is good, 28 is smaller.
COMPRESSION_CRF = 28

# `fast` or `medium`. Faster preset = slightly larger file.
COMPRESSION_PRESET = 'fast'

QA_MIN_PERSON_AREA=0.05

TARGET_SIZE_MB = 5
# A small buffer to ensure the final file is under the target size.
TARGET_SIZE_BUFFER_MB = 4.8
FILE_SIZE_THRESHOLD = TARGET_SIZE_MB * 1024 * 1024

def compress_to_target_size(video_path: Path) -> Path:
    """
    Checks video file size. If it exceeds the target, it performs a two-pass
    encode with FFmpeg to compress the file to just under the target size.
    """
    try:
        file_size = video_path.stat().st_size
        if file_size <= FILE_SIZE_THRESHOLD:
            return video_path

        print(f"  [Compress] Size ({file_size / 1024**2:.2f}MB) exceeds {TARGET_SIZE_MB}MB. Starting two-pass compression...")
        
        # 1. Get Video Duration (needed to calculate bitrate)
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print("  [Compress FAIL] Could not open video to get duration.")
            return video_path
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 30 # Default to 30s if duration fails
        cap.release()
        if duration <= 0:
            print("  [Compress FAIL] Could not determine video duration.")
            return video_path

        # 2. Calculate Target Bitrate
        # Formula: bitrate = (target_size_in_bits) / duration_in_seconds
        target_bits = TARGET_SIZE_BUFFER_MB * 1024 * 1024 * 8
        target_bitrate = int(target_bits / duration)
        
        output_path = video_path.with_name(f"{video_path.stem}_compressed.mp4")
        ffmpeg_log_files = glob.glob("ffmpeg2pass-*.log")
        
        # Use a try...finally block to ensure log files are always cleaned up
        try:
            # 3. Two-Pass Encoding
            # Pass 1: Analyze the video and create a log file
            print("  [Compress] Starting Pass 1...")
            pass1_command = [
                'ffmpeg', '-y', '-i', str(video_path),
                '-c:v', 'libx264', '-b:v', str(target_bitrate),
                '-pass', '1', '-preset', 'medium', '-an', '-f', 'mp4',
                os.devnull # Discard the video output, we only need the log
            ]
            result1 = subprocess.run(pass1_command, capture_output=True, text=True)
            if result1.returncode != 0:
                print(f"  [Compress FAIL] FFmpeg Pass 1 failed. Error: {result1.stderr}")
                return video_path

            # Pass 2: Use the log file to encode the final video
            print("  [Compress] Starting Pass 2...")
            pass2_command = [
                'ffmpeg', '-i', str(video_path),
                '-c:v', 'libx264', '-b:v', str(target_bitrate),
                '-pass', '2', '-preset', 'medium', '-an',
                str(output_path)
            ]
            result2 = subprocess.run(pass2_command, capture_output=True, text=True)
            if result2.returncode != 0:
                print(f"  [Compress FAIL] FFmpeg Pass 2 failed. Error: {result2.stderr}")
                return video_path

        finally:
            # 4. Clean up FFmpeg log files
            for log_file in glob.glob("ffmpeg2pass-*.log*"):
                os.remove(log_file)

        compressed_size = output_path.stat().st_size
        print(f"  [Compress SUCCESS] New size: {compressed_size / 1024**2:.2f}MB.")
        
        # 5. Replace original file with the compressed one
        original_path_str = str(video_path)
        video_path.unlink()
        output_path.rename(original_path_str)
        return Path(original_path_str)

    except Exception as e:
        print(f"  [Compress ERROR] An exception occurred: {e}")
        return video_path

def process_and_upload_clip(local_filepath: Path, db_conn, drive_service, video_id: str, part_num: int = 0):
    """
    Compresses, runs QA checks, and uploads a local video file.
    """
    print(f"\n--- Processing {local_filepath.name} ---")

    # === NEW STEP 1: COMPRESS THE FILE IF NEEDED ===
    processed_filepath = compress_to_target_size(local_filepath)

    # === STEP 2: RUN QA CHECKS ON THE FINAL FILE (original or compressed) ===
    print(f"--- Running QA on {processed_filepath.name} ---")
    if not quality_checks.verify_video_file(str(processed_filepath)):
        print(f"REJECTED CLIP: File is corrupted or unplayable.")
        return False

    if not quality_checks.check_for_hardcoded_subtitles(str(processed_filepath)):
        print(f"REJECTED CLIP: Failed hardcoded subtitle check.")
        return False
        
    if not quality_checks.check_scene_cuts(str(processed_filepath)):
        print(f"REJECTED CLIP: Failed scene cut check.")
        return False

    if not quality_checks.check_subject_clarity(str(processed_filepath), min_person_area=QA_MIN_PERSON_AREA):
        print(f"REJECTED CLIP: Failed subject clarity check.")
        return False

    # === STEP 3: UPLOAD THE FINAL FILE ===
    print(f"PASSED all checks. Uploading {processed_filepath.name} for human review...")
    try:
        part_suffix = f"_part_{part_num}" if part_num > 0 else ""
        file_metadata = { 'name': f"{video_id}{part_suffix}.mp4", 'parents': [PENDING_REVIEW_FOLDER_ID] }
        media = MediaFileUpload(processed_filepath, mimetype='video/mp4') # Use the final file path
        drive_service.files().create(body=file_metadata, media_body=media, fields='id').execute()
        print("SUCCESS: Uploaded to Google Drive.")
        return True
    except Exception as e:
        print(f"ERROR: Failed to upload clip to Drive. Reason: {e}")
        return False

# --- AUTHENTICATION & SERVICES ---
def get_google_services():
    """Handles authentication and returns YouTube and Drive service objects."""
    creds = None
    if os.path.exists(TOKEN_FILE):
        creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_FILE, SCOPES)
            creds = flow.run_local_server(port=0)
        with open(TOKEN_FILE, 'w') as token:
            token.write(creds.to_json())
    
    youtube_service = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)
    drive_service = build('drive', 'v3', credentials=creds)
    return youtube_service, drive_service

# --- DATABASE HELPERS ---
def setup_database():
    """Initializes a more sophisticated two-table database."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    # Table 1: Tracks parent videos and their overall status
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS parent_videos (
        video_id TEXT PRIMARY KEY,
        status TEXT, -- e.g., 'REJECTED_METADATA', 'FULLY_PROCESSED', 'PARTIALLY_PROCESSED'
        last_processed_at TIMESTAMP
    )""")
    # Table 2: Tracks individual chunks of long videos
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS processed_chunks (
        chunk_id TEXT PRIMARY KEY, -- e.g., 'hHSan_NYZtw_part_3'
        parent_video_id TEXT,
        status TEXT, -- 'PASSED', 'FAILED'
        FOREIGN KEY(parent_video_id) REFERENCES parent_videos(video_id)
    )""")
    conn.commit()
    return conn

def update_parent_video_status(conn, video_id, status):
    """Updates the status of a parent video."""
    cursor = conn.cursor()
    cursor.execute("INSERT OR REPLACE INTO parent_videos (video_id, status, last_processed_at) VALUES (?, ?, ?)",
                   (video_id, status, datetime.now()))
    conn.commit()

def log_chunk_status(conn, parent_video_id, part_num, status):
    """Logs the result of a single chunk processing attempt."""
    chunk_id = f"{parent_video_id}_part_{part_num}"
    cursor = conn.cursor()
    cursor.execute("INSERT OR REPLACE INTO processed_chunks (chunk_id, parent_video_id, status) VALUES (?, ?, ?)",
                   (chunk_id, parent_video_id, status))
    conn.commit()

def has_parent_video_been_processed(conn, video_id):
    """Checks if a parent video has been rejected or fully processed already."""
    cursor = conn.cursor()
    cursor.execute("SELECT status FROM parent_videos WHERE video_id = ?", (video_id,))
    result = cursor.fetchone()
    if result and result[0] in ['REJECTED_METADATA', 'FULLY_PROCESSED']:
        return True
    return False

def was_chunk_processed(conn, parent_video_id, part_num):
    """Checks if a specific chunk has been processed before, regardless of outcome."""
    chunk_id = f"{parent_video_id}_part_{part_num}"
    cursor = conn.cursor()
    cursor.execute("SELECT chunk_id FROM processed_chunks WHERE chunk_id = ?", (chunk_id,))
    return cursor.fetchone() is not None

# --- MAIN WORKFLOW ---
def main():
    print("--- Starting Pipeline ---")
    
    if os.path.exists(TEMP_VIDEO_DIR):
        shutil.rmtree(TEMP_VIDEO_DIR)
    os.makedirs(TEMP_VIDEO_DIR)
    
    db_conn = setup_database()
    youtube, drive = get_google_services()

    # --- Level 1: Search & Metadata Filtering (Unchanged) ---
    print("\n--- Level 1: Searching YouTube & Filtering Metadata ---")
    queries = ["swimming sports -edit -vlog", "diving sports -compilation", "kayak roll sports -fail", "water polo sports","water dive sports"]
    bad_keywords = ['edit', 'vlog', 'cinematic', 'fire edit', 'compilation', 'fails', 'music video']
    
    # ... (This whole search and metadata filter block is identical to the previous version)
    candidate_ids = set()
    for query in queries:
        request = youtube.search().list(q=query, part="id", type="video", maxResults=120).execute()
        candidate_ids.update(item['id']['videoId'] for item in request.get('items', []))
    
    videos_to_process = []
    for video_id in candidate_ids:
        if has_parent_video_been_processed(db_conn, video_id):
            continue
        try:
            details = youtube.videos().list(part="snippet,contentDetails", id=video_id).execute()['items'][0]
            title = details['snippet']['title'].lower()
            tags = [tag.lower() for tag in details['snippet'].get('tags', [])]
            duration_iso = details['contentDetails']['duration']
            duration = isodate.parse_duration(duration_iso).total_seconds()

            if any(word in title for word in bad_keywords) or any(word in tag for tag in tags for word in bad_keywords):
                update_parent_video_status(db_conn, video_id, "REJECTED_METADATA")
            else:
                videos_to_process.append({'id': video_id, 'duration': duration})
        except Exception as e:
            print(f"Could not process {video_id}: {e}")
    print(f"Found {len(videos_to_process)} videos passing initial checks.")
    
    # --- New Processing Loop ---
    for video in videos_to_process:
        video_id = video['id']
        duration = video['duration']
        print(f"\n{'='*20}\nProcessing Video ID: {video_id} (Duration: {duration:.0f}s)")
        
        # --- PATH A: Short Videos (<= MAX_VIDEO_DURATION_FOR_SINGLE_CLIP) ---
        if duration <= MAX_VIDEO_DURATION_FOR_SINGLE_CLIP:
            try:
                # Download a single clip (full video or middle 30s)
                video_url = f"https://www.youtube.com/watch?v={video_id}"
                output_template = os.path.join(TEMP_VIDEO_DIR, f'{video_id}.%(ext)s')
                ydl_opts = {'format': 'bestvideo[height<=1080][vcodec~=avc][ext=mp4]/bestvideo[height<=1080][ext=mp4]/bestvideo', 'outtmpl': output_template, 'quiet': True ,'postprocessors': [{
                        'key': 'FFmpegVideoConvertor',
                        'preferedformat': 'mp4', # Ensures the output container is mp4
                    }],}
                if duration > 30:
                    start, end = max(0, int((duration/2)-15)), max(0, int((duration/2)-15))+15
                    ydl_opts['download_ranges'] = yt_dlp.utils.download_range_func(None, [(start, end)])
                
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([video_url])

                local_filepath = Path(TEMP_VIDEO_DIR) / f"{video_id}.mp4"
                if local_filepath.exists():
                    process_and_upload_clip(local_filepath, db_conn, drive, video_id)
                else:
                    raise FileNotFoundError("Download did not produce an mp4 file.")
            except Exception as e:
                print(f"REJECTED (Download/Process Error): {e}")
            finally:
                # Mark as fully processed since it's a short video
                update_parent_video_status(db_conn, video_id, "FULLY_PROCESSED")
        # --- PATH B: Long Videos (to be chunked) ---
        else:
            print(f"Identified long video. Attempting to extract up to {MAX_SUCCESSFUL_CHUNKS_PER_VIDEO} clips.")
            # Define content window: ignore first 15% and last 15%
            # ** NEW: Add a counter for successful uploads **
            successful_uploads = 0
            consecutive_failures = 0
            content_start = int(duration * 0.15)
            content_end = int(duration * 0.85)
            current_pos = content_start
            part_num = 1

            # ** NEW: Update the while loop condition **
            while (current_pos + 15 <= content_end and 
                   successful_uploads < MAX_SUCCESSFUL_CHUNKS_PER_VIDEO and 
                   consecutive_failures < MAX_CONSECUTIVE_FAILURES):
                
                if was_chunk_processed(db_conn, video_id, part_num):
                    print(f"\n-- Skipping clip {part_num} (already processed in a previous run) --")
                    part_num += 1
                    current_pos += 15
                    continue # Move to the next iteration
                start_time = current_pos
                end_time = start_time + 15
                clip_filename = f"{video_id}_part_{part_num}"
                print(f"\n-- Attempting clip {part_num} (Success: {successful_uploads}/{MAX_SUCCESSFUL_CHUNKS_PER_VIDEO} | Fails: {consecutive_failures}/{MAX_CONSECUTIVE_FAILURES}) --")

                try:
                    video_url = f"https://www.youtube.com/watch?v={video_id}"
                    output_template = os.path.join(TEMP_VIDEO_DIR, f'{clip_filename}.%(ext)s')
                    ydl_opts = {
                        'format': 'bestvideo[height<=1080][vcodec~=avc][ext=mp4]/bestvideo[height<=1080][ext=mp4]/bestvideo',
                        'outtmpl': output_template, 'quiet': True,
                        'download_ranges': yt_dlp.utils.download_range_func(None, [(start_time, end_time)]),
                        'postprocessors': [{
                            'key': 'FFmpegVideoConvertor',
                            'preferedformat': 'mp4',
                        }],
                    }
                    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                        ydl.download([video_url])
                    
                    local_filepath = Path(TEMP_VIDEO_DIR) / f"{clip_filename}.mp4"
                    if local_filepath.exists():
                        # The function returns True on success
                        was_successful = process_and_upload_clip(local_filepath, db_conn, drive, video_id, part_num)
                        
                        # ** NEW: Reset or increment the failure counter **
                        if was_successful:
                            successful_uploads += 1
                            consecutive_failures = 0 # RESET on success
                            log_chunk_status(db_conn, video_id, part_num, "PASSED")
                        else:
                            consecutive_failures += 1 # INCREMENT on failure
                            log_chunk_status(db_conn, video_id, part_num, "FAILED")
                    else:
                        raise FileNotFoundError("Clip download did not produce an mp4 file.")
                except Exception as e:
                    print(f"Could not process clip {part_num}. Reason: {e}")
                    consecutive_failures += 1 # Also count download errors as failures
                    log_chunk_status(db_conn, video_id, part_num, "DOWNLOAD_ERROR")
                part_num += 1
                current_pos += 15 # Move to the next 30-second segment
            
             # ** NEW: Update parent status based on why the loop stopped **
            if successful_uploads >= MAX_SUCCESSFUL_CHUNKS_PER_VIDEO or current_pos + 30 > content_end:
                # If we achieved our goal or ran out of video, it's fully processed
                print(f"\nFinished processing long video {video_id}. Found {successful_uploads} good clips. Marking as fully processed.")
                update_parent_video_status(db_conn, video_id, "FULLY_PROCESSED")
            else:
                # If we stopped due to failures, it's only partially processed and can be re-evaluated later
                print(f"\nPaused processing long video {video_id} due to consecutive failures. Found {successful_uploads} good clips so far.")
                update_parent_video_status(db_conn, video_id, "PARTIALLY_PROCESSED")

    # --- Final Cleanup ---
    db_conn.close()
    if os.path.exists(TEMP_VIDEO_DIR):
        shutil.rmtree(TEMP_VIDEO_DIR)
    print(f"\n{'='*20}\n--- Pipeline Finished ---")
if __name__ == "__main__":
    main()
