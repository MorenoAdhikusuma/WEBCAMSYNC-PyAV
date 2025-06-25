import os
import cv2
import time
import av
import datetime
import subprocess
import json
from datetime import datetime, timedelta

# ---- Get metadata using ffprobe ----
def get_video_metadata(path: str) -> dict:
    cmd = [
        'ffprobe',
        '-v', 'quiet',
        '-print_format', 'json',
        '-show_format',
        '-show_streams',
        path
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    metadata = json.loads(result.stdout)

    format_tags = metadata.get("format", {}).get("tags", {})
    streams = metadata.get("streams", [])
    stream_tags = streams[0].get("tags", {}) if streams else {}

    creation_time = (
        format_tags.get("creation_time") or
        stream_tags.get("creation_time")
    )
    
    # Convert to Unix timestamp
    unix_timestamp = None
    if creation_time:
        try:
            if creation_time.endswith('Z'):
                dt = datetime.fromisoformat(creation_time[:-1] + '+00:00')
            else:
                dt = datetime.fromisoformat(creation_time)
            unix_timestamp = int(dt.timestamp())
        except ValueError:
            pass

    return {
        "duration": metadata.get("format", {}).get("duration"),
        "bit_rate": metadata.get("format", {}).get("bit_rate"),
        "filename": metadata.get("format", {}).get("filename"),
        "start_time": metadata.get("format", {}).get("start_time"),
        "creation_time": creation_time,
        "unix_timestamp": unix_timestamp
    }

# ---- Extract frames with Unix timestamps ----
def extract_frames_with_unixtime(video_path, output_dir, fps=30):
    os.makedirs(output_dir, exist_ok=True)
    container = av.open(video_path)
    stream = container.streams.video[0]

    metadata = get_video_metadata(video_path)
    base_unix = metadata.get("unix_timestamp") or int(time.time())

    print(f"Extracting from: {os.path.basename(video_path)}")
    print(f"Base Unix timestamp: {base_unix}")
    print(f"Equivalent datetime: {datetime.fromtimestamp(base_unix).isoformat()}")

    count = 0
    for frame in container.decode(video=0):
        if frame.pts is None:
            continue

        # Calculate exact timestamp
        time_offset = float(frame.pts * stream.time_base)
        exact_unix = base_unix + time_offset
        unix_seconds = int(exact_unix)
        milliseconds = int((exact_unix - unix_seconds) * 1000)

        filename = f"{unix_seconds}_{milliseconds:03d}.jpg"
        output_path = os.path.join(output_dir, filename)
        
        img = frame.to_ndarray(format="bgr24")
        cv2.imwrite(output_path, img)
        count += 1

    print(f"Done: {count} frames extracted with Unix timestamps.")
    container.close()

# ---- Convert existing numbered frames to Unix timestamps ----
def convert_numbered_to_unix(input_dir, output_dir, base_unix_time=None, fps=30):
    """
    Convert existing numbered frames (00000.jpg, 00001.jpg) to Unix timestamp format
    """
    os.makedirs(output_dir, exist_ok=True)
    
    base_time = base_unix_time or int(time.time())
    print(f"Base Unix timestamp: {base_time}")
    print(f"Equivalent datetime: {datetime.fromtimestamp(base_time).isoformat()}")

    frame_files = sorted([f for f in os.listdir(input_dir) if f.endswith(('.jpg', '.png'))])
    
    for i, filename in enumerate(frame_files):
        try:
            frame_num = int(os.path.splitext(filename)[0])
        except ValueError:
            continue
            
        time_offset = frame_num / fps
        exact_unix = base_time + time_offset
        unix_seconds = int(exact_unix)
        milliseconds = int((exact_unix - unix_seconds) * 1000)
        
        new_filename = f"{unix_seconds}_{milliseconds:03d}.jpg"
        old_path = os.path.join(input_dir, filename)
        new_path = os.path.join(output_dir, new_filename)
        
        # Copy file to preserve original
        img = cv2.imread(old_path)
        cv2.imwrite(new_path, img)
        
        print(f"Converted {filename} → {new_filename}")
    
    print(f"\nDone! Converted {len(frame_files)} frames to Unix timestamp format.")

# ---- Process all videos ----
def process_all_videos(input_dir, output_dir, mode="unixtime", fps=30):
    os.makedirs(output_dir, exist_ok=True)

    for file in os.listdir(input_dir):
        if file.lower().endswith((".mp4", ".mts", ".mov", ".avi", ".mpi")):
            video_path = os.path.join(input_dir, file)
            metadata = get_video_metadata(video_path)

            print(f"\n[Metadata] {file}:")
            print(f"  Duration: {float(metadata['duration']):.2f}s" if metadata['duration'] else "  Duration: N/A")
            print(f"  Bitrate: {metadata['bit_rate']} bps" if metadata['bit_rate'] else "  Bitrate: N/A")
            print(f"  Creation time (Unix): {metadata['unix_timestamp']}" if metadata['unix_timestamp'] else "  Creation time: N/A")

            output_subdir = os.path.join(
                output_dir, 
                f"unix_{metadata['unix_timestamp'] or int(time.time())}_{os.path.splitext(file)[0]}"
            )

            if mode == "unixtime":
                extract_frames_with_unixtime(video_path, output_subdir, fps)
            elif mode == "convert":
                convert_numbered_to_unix(video_path, output_subdir, metadata.get("unix_timestamp"), fps)
            else:
                print(f"Unknown mode: {mode}")

# ---- Entry point ----
if __name__ == "__main__":
    input_folder = r"C:\Users\moreno\programming\Scientific_Works\senyum\Not_experiment\OUTPUT_VID"
    output_folder = r"C:\Users\moreno\programming\Scientific_Works\senyum\Not_experiment\OUTPUT_VID\frames"
    
    # Choose mode:
    # "unixtime" - extract frames directly with Unix timestamps
    # "convert" - convert existing numbered frames to Unix timestamps
    extract_mode = "convert"
    frames_per_second = 30  # Adjust based on your video's FPS
    
    process_all_videos(input_folder, output_folder, mode=extract_mode, fps=frames_per_second)