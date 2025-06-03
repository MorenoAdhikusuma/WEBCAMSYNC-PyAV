import os
import cv2
import time
import av

def extract_frames_with_timecode(video_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    container = av.open(video_path)
    stream = container.streams.video[0]
    fps = float(stream.average_rate)
    count = 0

    for frame in container.decode(video=0):
        time_offset = float(frame.pts * stream.time_base)
        total_seconds = int(time_offset)
        frames = int((time_offset - total_seconds) * fps)

        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60

        timecode = f"{hours:02d}_{minutes:02d}_{seconds:02d}_{frames:02d}.jpg"
        output_path = os.path.join(output_dir, timecode)

        img = frame.to_ndarray(format="bgr24")
        cv2.imwrite(output_path, img)
        count += 1

    print(f"Done: {count} frames extracted using timecode from {os.path.basename(video_path)}\n")
    container.close()


def extract_frames_by_index(video_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Processing {video_path} | Frames: {video_length}")
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
    print(f"Done: {count} frames extracted in {int(time_end - time_start)}s.\n")


def process_all_videos(input_dir, output_dir, mode="index"):
    """
    mode: 'index' for 00000.jpg naming (OpenCV),
          'timecode' for hh_mm_ss_ff.jpg naming (PyAV)
    """
    os.makedirs(output_dir, exist_ok=True)
    for file in os.listdir(input_dir):
        if file.lower().endswith((".mp4", ".mts", ".mov", ".avi")):
            video_path = os.path.join(input_dir, file)
            output_subdir = os.path.join(output_dir, os.path.splitext(file)[0])
            if mode == "timecode":
                extract_frames_with_timecode(video_path, output_subdir)
            else:
                extract_frames_by_index(video_path, output_subdir)


if __name__ == "__main__":
    input_folder = "/path/to/videos"
    output_folder = "/path/to/ExtractedFrames"
    extract_mode = "timecode"  # or "index"

    process_all_videos(input_folder, output_folder, mode=extract_mode)

# OUTPUT DIR STRUCTURE
#     C:\Path\To\ExtractedFrames\
# ├── Subject1\
# │   └── test1\
# │       └── 00000.jpg ...
# ├── Subject2\
# │   └── sessionB\
# │       └── 00000.jpg ...