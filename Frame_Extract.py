import os
import cv2
import time
import av
import datetime
import subprocess
import json
import piexif
from datetime import datetime, timedelta, timezone

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

    dt_utc = None
    if creation_time:
        try:
            if creation_time.endswith('Z'):
                dt_utc = datetime.fromisoformat(creation_time[:-1] + '+00:00')
            else:
                dt_utc = datetime.fromisoformat(creation_time)
        except ValueError:
            pass

    return {
        "duration": metadata.get("format", {}).get("duration"),
        "bit_rate": metadata.get("format", {}).get("bit_rate"),
        "filename": metadata.get("format", {}).get("filename"),
        "start_time": metadata.get("format", {}).get("start_time"),
        "creation_time": dt_utc.isoformat() if dt_utc else None,
        "datetime_utc": dt_utc
    }

# ---- Convert UTC to UTC+7 ISO ----
def to_utc7(dt: datetime) -> datetime:
    return dt.astimezone(timezone(timedelta(hours=7)))

def to_safe_filename(dt: datetime) -> str:
    return dt.isoformat().replace(':', '-')

def format_exif_datetime(dt: datetime) -> str:
    return dt.strftime("%Y:%m:%d %H:%M:%S")

# ---- Save image with EXIF ----
def save_image_with_exif(img, filepath, dt: datetime):
    exif_dict = {"0th": {}, "Exif": {}, "GPS": {}, "1st": {}, "thumbnail": None}
    exif_time_str = format_exif_datetime(dt)

    # EXIF DateTimeOriginal
    exif_dict["Exif"][piexif.ExifIFD.DateTimeOriginal] = exif_time_str
    exif_dict["Exif"][piexif.ExifIFD.DateTimeDigitized] = exif_time_str
    exif_dict["0th"][piexif.ImageIFD.DateTime] = exif_time_str

    exif_bytes = piexif.dump(exif_dict)

    # Save with OpenCV and then inject EXIF
    temp_path = filepath + ".temp.jpg"
    cv2.imwrite(temp_path, img)
    piexif.insert(exif_bytes, temp_path, filepath)
    os.remove(temp_path)

# ---- Extract frames with ISO timestamps and EXIF ----
def extract_frames_with_iso_exif(video_path, output_dir, fps=30):
    os.makedirs(output_dir, exist_ok=True)
    container = av.open(video_path)
    stream = container.streams.video[0]

    metadata = get_video_metadata(video_path)
    base_dt = metadata.get("datetime_utc") or datetime.now(timezone.utc)

    print(f"Extracting from: {os.path.basename(video_path)}")
    print(f"Base UTC time: {base_dt.isoformat()}")
    print(f"Base UTC+7 time: {to_utc7(base_dt).isoformat()}")

    count = 0
    for frame in container.decode(video=0):
        if frame.pts is None:
            continue

        time_offset = float(frame.pts * stream.time_base)
        frame_dt = to_utc7(base_dt + timedelta(seconds=time_offset))

        filename = f"{to_safe_filename(frame_dt)}.jpg"
        output_path = os.path.join(output_dir, filename)

        img = frame.to_ndarray(format="bgr24")
        save_image_with_exif(img, output_path, frame_dt)

        count += 1

    print(f"Done: {count} frames extracted with ISO timestamps and EXIF.")
    container.close()

# ---- Process all videos ----
def process_all_videos(input_dir, output_dir, fps=30):
    os.makedirs(output_dir, exist_ok=True)

    for file in os.listdir(input_dir):
        if file.lower().endswith((".mp4", ".mts", ".mov", ".avi", ".mkv")):
            video_path = os.path.join(input_dir, file)
            metadata = get_video_metadata(video_path)

            print(f"\n[Metadata] {file}:")
            print(f"  Duration: {float(metadata['duration']):.2f}s" if metadata['duration'] else "  Duration: N/A")
            print(f"  Bitrate: {metadata['bit_rate']} bps" if metadata['bit_rate'] else "  Bitrate: N/A")
            print(f"  Creation time: {metadata['creation_time']}" if metadata['creation_time'] else "  Creation time: N/A")

            creation_dt = metadata.get("datetime_utc")
            output_subdir = os.path.join(
                output_dir,
                f"ISO_{to_safe_filename(to_utc7(creation_dt))}__{os.path.splitext(file)[0]}"
            )

            extract_frames_with_iso_exif(video_path, output_subdir, fps)

# ---- Entry Point ----
if __name__ == "__main__":
    input_folder = r"C:\Users\moreno\programming\Scientific_Works\senyum\Not_experiment\OUTPUT_VID"
    output_folder = r"C:\Users\moreno\programming\Scientific_Works\senyum\Not_experiment\OUTPUT_VID\frames"

    frames_per_second = 30
    process_all_videos(input_folder, output_folder, fps=frames_per_second)
