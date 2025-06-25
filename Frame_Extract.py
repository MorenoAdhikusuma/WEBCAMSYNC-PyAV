import os
import cv2
import time
import av
import datetime
import subprocess
import json

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

    # Prefer creation_time from format tags or stream tags
    format_tags = metadata.get("format", {}).get("tags", {})
    streams = metadata.get("streams", [])
    stream_tags = streams[0].get("tags", {}) if streams else {}

    creation_time = (
        format_tags.get("creation_time") or
        stream_tags.get("creation_time")
    )

    return {
        "duration": metadata.get("format", {}).get("duration"),
        "bit_rate": metadata.get("format", {}).get("bit_rate"),
        "filename": metadata.get("format", {}).get("filename"),
        "start_time": metadata.get("format", {}).get("start_time"),
        "creation_time": creation_time
    }

# ---- Extract frames using ISO 8601 timecode ----
def extract_frames_with_timecode(video_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    container = av.open(video_path)
    stream = container.streams.video[0]

    metadata = get_video_metadata(video_path)
    creation_time_str = metadata.get("creation_time")

    if creation_time_str:
        try:
            base_time = datetime.datetime.fromisoformat(creation_time_str.replace("Z", "+00:00"))
        except ValueError:
            print(f"Invalid creation_time format: {creation_time_str}. Using epoch.")
            base_time = datetime.datetime.fromtimestamp(0, tz=datetime.timezone.utc)
    else:
        # Fallback to start_time or epoch
        start_time_str = metadata.get("start_time")
        if start_time_str:
            try:
                base_time = datetime.datetime.fromtimestamp(float(start_time_str), tz=datetime.timezone.utc)
            except ValueError:
                base_time = datetime.datetime.fromtimestamp(0, tz=datetime.timezone.utc)
        else:
            base_time = datetime.datetime.fromtimestamp(0, tz=datetime.timezone.utc)

    print(f"Extracting from: {os.path.basename(video_path)}")
    print(f"Base time: {base_time.isoformat()}")

    count = 0
    for frame in container.decode(video=0):
        if frame.pts is None:
            continue

        time_offset = float(frame.pts * stream.time_base)
        frame_time = base_time + datetime.timedelta(seconds=time_offset)

        timecode = frame_time.isoformat(timespec='milliseconds').replace(":", "-").replace("T", "_")
        filename = f"{timecode}.jpg"
        output_path = os.path.join(output_dir, filename)

        img = frame.to_ndarray(format="bgr24")
        cv2.imwrite(output_path, img)
        count += 1

    print(f"Done: {count} frames extracted using timecode.")
    container.close()

# ---- Extract frames by index ----
def extract_frames_by_index(video_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Processing {video_path} | Total Frames: {video_length}")
    count = 0
    time_start = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        filename = os.path.join(output_dir, f"{count:05d}.jpg")
        cv2.imwrite(filename, frame)
        count += 1

    cap.release()
    time_end = time.time()
    print(f" Done: {count} frames extracted in {int(time_end - time_start)}s\n")

# ---- Batch process videos ----
def process_all_videos(input_dir, output_dir, mode="index"):
    os.makedirs(output_dir, exist_ok=True)

    for file in os.listdir(input_dir):
        if file.lower().endswith((".mp4", ".mts", ".mov", ".avi")):
            video_path = os.path.join(input_dir, file)
            metadata = get_video_metadata(video_path)

            print(f"\n[Metadata] {file}:")
            for k, v in metadata.items():
                print(f"  {k}: {v}")

            base_time_str = metadata.get("creation_time") or metadata.get("start_time")
            if base_time_str:
                sanitized_time = base_time_str.replace(":", "-").replace("T", "_").replace("Z", "")
                output_subdir = os.path.join(output_dir, f"{sanitized_time}_{os.path.splitext(file)[0]}")
            else:
                output_subdir = os.path.join(output_dir, f"fallback_{os.path.splitext(file)[0]}")

            if mode == "timecode":
                extract_frames_with_timecode(video_path, output_subdir)
            else:
                extract_frames_by_index(video_path, output_subdir)

# ---- Entry point ----
if __name__ == "__main__":
    input_folder = r"C:\Users\moreno\programming\Scientific_Works\senyum\Not_experiment\OUTPUT_VID"
    output_folder = r"C:\Users\moreno\programming\Scientific_Works\senyum\Not_experiment\OUTPUT_VID\frames"
    extract_mode = "timecode"  # Options: "timecode" or "index"

    process_all_videos(input_folder, output_folder, mode=extract_mode)
