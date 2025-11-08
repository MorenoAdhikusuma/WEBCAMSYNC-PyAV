import os
import cv2
import av
import subprocess
import json
import piexif
from datetime import datetime, timedelta, timezone


# =========================  Utilities  =========================
def get_video_metadata(path: str) -> dict:
    """Extract video metadata using ffprobe."""
    cmd = [
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_format", "-show_streams", path
    ]
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                text=True, timeout=30, check=False)
        metadata = json.loads(result.stdout)
    except (subprocess.TimeoutExpired, json.JSONDecodeError) as e:
        print(f"⚠️ Could not read metadata from {os.path.basename(path)}: {e}")
        return {}

    fmt = metadata.get("format", {})
    tags = fmt.get("tags", {})
    creation_time = tags.get("creation_time") or fmt.get("creation_time") or fmt.get("date")
    dt_utc = None
    if creation_time:
        try:
            if creation_time.endswith("Z"):
                dt_utc = datetime.fromisoformat(creation_time[:-1] + "+00:00")
            else:
                dt_utc = datetime.fromisoformat(creation_time)
            if dt_utc.tzinfo is None:
                dt_utc = dt_utc.replace(tzinfo=timezone.utc)
        except Exception:
            pass

    return {
        "duration": fmt.get("duration"),
        "bit_rate": fmt.get("bit_rate"),
        "filename": fmt.get("filename", path),
        "creation_time": dt_utc.isoformat() if dt_utc else None,
        "datetime_utc": dt_utc
    }


def to_utc7(dt: datetime) -> datetime:
    """Convert datetime to UTC+7 timezone."""
    return dt.astimezone(timezone(timedelta(hours=7)))


def to_iso_with_milliseconds(dt: datetime) -> str:
    """ISO format with 3 decimal milliseconds (e.g. 2025-11-05T20:30:57.123+07:00)."""
    base = dt.isoformat()
    if "." in base:
        main, frac = base.split(".", 1)
        frac = frac[:6]
        if "+" in frac:
            ms, tz = frac.split("+", 1)
            return f"{main}.{ms[:3]}+{tz}"
        if "-" in frac:
            ms, tz = frac.split("-", 1)
            return f"{main}.{ms[:3]}-{tz}"
        return f"{main}.{frac[:3]}"
    return base


def to_safe_filename(dt: datetime) -> str:
    """Convert datetime to safe filename (replace ':' with '-')."""
    return to_iso_with_milliseconds(dt).replace(":", "-")


def format_exif_datetime(dt: datetime) -> str:
    """Format datetime for EXIF metadata."""
    return dt.strftime("%Y:%m:%d %H:%M:%S")


def save_image_with_exif(img, filepath, dt: datetime):
    """Save image with EXIF timestamp metadata."""
    exif_time = format_exif_datetime(dt)
    exif_dict = {
        "0th": {piexif.ImageIFD.DateTime: exif_time},
        "Exif": {
            piexif.ExifIFD.DateTimeOriginal: exif_time,
            piexif.ExifIFD.DateTimeDigitized: exif_time
        }
    }
    exif_bytes = piexif.dump(exif_dict)

    temp = filepath + ".writing.jpg"
    try:
        ok = cv2.imwrite(temp, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if not ok:
            raise Exception("cv2.imwrite failed")
        piexif.insert(exif_bytes, temp, filepath)
        os.remove(temp)
    except Exception as e:
        if os.path.exists(temp):
            os.replace(temp, filepath)
        print(f"⚠️ Could not insert EXIF for {os.path.basename(filepath)}: {e}")


def load_session_start_ms(session_folder: str):
    """Read host_start_ms.txt and return datetime UTC+7."""
    fpath = os.path.join(session_folder, "host_start_ms.txt")
    if not os.path.exists(fpath):
        return None
    try:
        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                if "epoch_ms" in line:
                    epoch_ms = int(line.strip().split(":")[1])
                    return datetime.fromtimestamp(epoch_ms / 1000, tz=timezone(timedelta(hours=7)))
    except Exception as e:
        print(f"⚠️ Could not read host_start_ms: {e}")
    return None


# =========================  Core extraction  =========================
def extract_frames_with_iso_exif(video_path, output_dir, fps=30, base_dt=None):
    """Extract frames with ISO timestamps + EXIF (aligned to base_dt)."""
    os.makedirs(output_dir, exist_ok=True)
    container = av.open(video_path)
    stream = container.streams.video[0]

    metadata = get_video_metadata(video_path)
    if base_dt is None:
        base_dt = metadata.get("datetime_utc") or datetime.now(timezone.utc)

    base_utc7 = to_utc7(base_dt)
    print(f"\n🎥 {os.path.basename(video_path)}")
    print(f"Base time: {base_utc7.isoformat()} (UTC+7)")

    cnt = 0
    for frame in container.decode(video=0):
        if frame.pts is None:
            continue
        offset = float(frame.pts * stream.time_base)
        frame_dt = base_utc7 + timedelta(seconds=offset)
        fname = f"{to_safe_filename(frame_dt)}.jpg"
        out = os.path.join(output_dir, fname)
        img = frame.to_ndarray(format="bgr24")
        save_image_with_exif(img, out, frame_dt)
        cnt += 1
        if cnt % 100 == 0:
            print(f"  {cnt} frames extracted...")
    container.close()
    print(f"✅ Done {cnt} frames from {os.path.basename(video_path)}")
    return cnt


def process_session_folder(session_folder: str, fps: int = 20):
    """Process videos in a recorded session folder."""
    cameras = os.path.join(session_folder, "cameras")
    frames_root = os.path.join(session_folder, "frames")
    os.makedirs(frames_root, exist_ok=True)
    if not os.path.exists(cameras):
        print(f"❌ Cameras folder not found: {cameras}")
        return

    videos = [f for f in os.listdir(cameras)
              if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv", ".mts"))]
    if not videos:
        print("⚠️ No videos found.")
        return

    print(f"\n{'='*60}\n📁 Session: {session_folder}\nFound {len(videos)} video(s)\n{'='*60}")
    session_start = load_session_start_ms(session_folder)
    if session_start:
        print(f"🕒 Using synchronized start: {session_start.isoformat()} (UTC+7)")
    else:
        print("⚠️ host_start_ms.txt not found — fallback to file metadata")

    total = 0
    for i, vf in enumerate(videos, 1):
        vpath = os.path.join(cameras, vf)
        vname = os.path.splitext(vf)[0]
        out_dir = os.path.join(frames_root, vname)
        print(f"\n[{i}/{len(videos)}] Extracting {vf} → {out_dir}")
        try:
            frames = extract_frames_with_iso_exif(vpath, out_dir, fps,
                                                  base_dt=session_start.astimezone(timezone.utc) if session_start else None)
            total += frames
        except Exception as e:
            print(f"❌ Error processing {vf}: {e}")

    print(f"\n{'='*60}\n🎉 Extraction complete — {total} frames total\nOutput: {frames_root}\n{'='*60}")


def process_all_videos(input_dir, output_dir, fps=30):
    """Legacy fallback: extract all videos in folder (no session sync)."""
    os.makedirs(output_dir, exist_ok=True)
    for file in os.listdir(input_dir):
        if file.lower().endswith((".mp4", ".mts", ".mov", ".avi", ".mkv")):
            vpath = os.path.join(input_dir, file)
            vout = os.path.join(output_dir, os.path.splitext(file)[0])
            print(f"\nLegacy extract: {file}")
            extract_frames_with_iso_exif(vpath, vout, fps)


# =========================  Entrypoint  =========================
if __name__ == "__main__":
    folder = r"./sessions"
    cameras = os.path.join(folder, "cameras")
    if os.path.exists(cameras):
        print("Detected session folder → synchronized extraction mode")
        process_session_folder(folder, fps=20)
    else:
        print("No session folder found → legacy mode")
        out = os.path.join(folder, "frames")
        process_all_videos(folder, out, fps=30)
