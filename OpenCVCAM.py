import cv2
import threading
import time
import av
import numpy as np
import os
import queue
from datetime import datetime, timedelta, timezone
from pygrabber.dshow_graph import FilterGraph
import subprocess
from fractions import Fraction   

# --- Add metadata to MP4 with proper UTC+7 timestamp and EXIF-style tag ---
def add_start_time_metadata(mp4_path, timestamp_utc7):
    base, ext = os.path.splitext(mp4_path)
    output_path = base + "_withmeta" + ext

    cmd = [
        'ffmpeg',
        '-y',
        '-i', mp4_path,
        '-map_metadata', '0',
        '-metadata', f'creation_time={timestamp_utc7}',
        '-metadata', f'com.apple.quicktime.creationdate={timestamp_utc7}',
        '-codec', 'copy',
        output_path
    ]

    try:
        subprocess.run(cmd, check=True)
        os.replace(output_path, mp4_path)
        print(f"[FFMPEG] Metadata added to {mp4_path}")
    except subprocess.CalledProcessError as e:
        print(f"[FFMPEG] Failed to add metadata: {e}")

# --- Directory Setup ---
OUTPUT_DIR = r"D:\Machine learning code\riset\OUTPUT"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Camera Detection ---
graph = FilterGraph()
devices = graph.get_input_devices()
max_cams_to_check = 10
cameras_identifier = []

for real_index in range(max_cams_to_check):
    cap = cv2.VideoCapture(real_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap.release()
        continue
    try:
        name = devices[real_index]
    except IndexError:
        name = "Unknown"

    if name.startswith("GENERAL - UVC "):
        cameras_identifier.append(real_index)
        print(f"Using camera index {real_index}: {name}")
    else:
        print(f"Skipping camera index {real_index}: {name}")
    cap.release()

# --- Camera Thread Class ---
class CameraWorker(threading.Thread):
    def __init__(self, cam_index):
        super().__init__()
        self.cam_index = cam_index
        self.cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera {cam_index}")

        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = 20  # Force to 20 FPS

        self.frame = None
        self.running = True
        self.recording = False
        self.output = None
        self.stream = None
        self.lock = threading.Lock()
        self.start_time = None
        self.frame_count = 0

        self.frame_queue = queue.Queue(maxsize=30)
        self.encoding_thread = None
        self.preview_scale = 0.5

    def start_recording(self):
        filename = os.path.join(OUTPUT_DIR, f'cam_{self.cam_index}.mp4')
        self.output = av.open(filename, mode='w')
        self.stream = self.output.add_stream('libx264', rate=self.fps)
        self.stream.width = self.width
        self.stream.height = self.height
        self.stream.pix_fmt = 'yuv420p'
        self.stream.options = {'preset': 'ultrafast', 'crf': '23'}

        self.recording = True
        self.start_time = time.time()
        self.frame_count = 0
        self.encoding_thread = threading.Thread(target=self.encoding_worker)
        self.encoding_thread.daemon = True
        self.encoding_thread.start()
        print(f"[Camera {self.cam_index}] Recording started at {self.fps} FPS")

    def encoding_worker(self):
        while self.recording:
            try:
                frame_data = self.frame_queue.get(timeout=0.5)
                if frame_data is None:
                    break
                frame, timestamp = frame_data
                video_frame = av.VideoFrame.from_ndarray(frame, format='bgr24')

                # ✅ Force exact 20 fps with frame counter
                video_frame.pts = self.frame_count
                video_frame.time_base = Fraction(1, self.fps)

                for packet in self.stream.encode(video_frame):
                    self.output.mux(packet)

                self.frame_count += 1
                self.frame_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in encoding thread: {e}")

        # flush encoder saat stop
        try:
            for packet in self.stream.encode(None):
                self.output.mux(packet)
        except Exception as e:
            print(f"Error flushing frames: {e}")

    def stop_recording(self):
        if self.recording:
            self.recording = False
            self.frame_queue.put(None)
            if self.encoding_thread:
                self.encoding_thread.join(timeout=5.0)
            elapsed = time.time() - self.start_time
            actual_fps = self.frame_count / elapsed if elapsed > 0 else 0
            print(f"[Camera {self.cam_index}] Recording stopped. Frames: {self.frame_count}, Duration: {elapsed:.2f}s, Effective FPS: {actual_fps:.2f}")
            try:
                self.output.close()
            except Exception as e:
                print(f"Error closing output: {e}")

            # --- Format timestamp in ISO 8601 with timezone +07:00 ---
            tz_utc_plus_7 = timezone(timedelta(hours=7))
            utc_plus_7 = datetime.fromtimestamp(self.start_time, tz=tz_utc_plus_7)
            iso_time = utc_plus_7.isoformat()

            filename = os.path.join(OUTPUT_DIR, f'cam_{self.cam_index}.mp4')
            add_start_time_metadata(filename, iso_time)

    def run(self):
        frame_interval = 1.0 / self.fps
        last_capture_time = time.time()
        frame_count = 0
        display_interval = 2
        tz_utc_plus_7 = timezone(timedelta(hours=7))

        while self.running:
            current_time = time.time()
            if current_time - last_capture_time < frame_interval:
                time.sleep(max(0.001, frame_interval - (current_time - last_capture_time)))
                continue

            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            last_capture_time = time.time()
            frame_count += 1
            timestamp = time.time()

            human_time = datetime.fromtimestamp(timestamp, tz=tz_utc_plus_7).strftime("%Y-%m-%d %H:%M:%S")

            if frame_count % display_interval == 0:
                preview_frame = cv2.resize(frame, (0, 0), fx=self.preview_scale, fy=self.preview_scale)
                cv2.putText(preview_frame, f'Cam {self.cam_index}: {human_time}',
                            (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                with self.lock:
                    self.frame = preview_frame

            if self.recording and not self.frame_queue.full():
                cv2.putText(frame, f'Cam {self.cam_index}: {human_time}',
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                try:
                    self.frame_queue.put((frame.copy(), timestamp), block=False)
                except queue.Full:
                    pass

        self.stop_recording()
        self.cap.release()

# --- Smart Display Combiner ---
class SmartCombiner:
    def __init__(self, max_width=1920, max_height=1080):
        self.max_width = max_width
        self.max_height = max_height
        self.last_grid = None
        self.last_update = 0
        self.update_interval = 1/60

    def combine_frames(self, frames, cols=2):
        current_time = time.time()
        if self.last_grid is not None and (current_time - self.last_update) < self.update_interval:
            return self.last_grid

        if not frames:
            return None

        while len(frames) < 4:
            h, w = frames[0].shape[:2] if frames else (240, 320)
            blank = np.zeros((h, w, 3), dtype=np.uint8)
            frames.append(blank)

        rows = 2
        h_max = max(f.shape[0] for f in frames)
        w_max = max(f.shape[1] for f in frames)
        grid = np.zeros((h_max * rows, w_max * cols, 3), dtype=np.uint8)

        for idx in range(4):
            frame = frames[idx]
            r = idx // cols
            c = idx % cols
            h, w = frame.shape[:2]
            y_offset = r * h_max + (h_max - h) // 2
            x_offset = c * w_max + (w_max - w) // 2
            grid[y_offset:y_offset+h, x_offset:x_offset+w] = frame

        grid_h, grid_w = grid.shape[:2]
        if grid_w > self.max_width or grid_h > self.max_height:
            scale = min(self.max_width / grid_w, self.max_height / grid_h)
            grid = cv2.resize(grid, (int(grid_w * scale), int(grid_h * scale)), interpolation=cv2.INTER_AREA)

        self.last_grid = grid
        self.last_update = current_time
        return grid

# --- Main Function ---
def main():
    cameras = []
    for i in cameras_identifier:
        try:
            cam = CameraWorker(i)
            cameras.append(cam)
        except RuntimeError as e:
            print(f"Couldn't initialize camera {i}: {e}")

    if not cameras:
        print("No cameras available. Exiting.")
        return

    for cam in cameras:
        cam.start()

    recording = False
    print("Press 'r' to start/stop recording, 'Spacebar' to quit.")
    window_name = 'MULTICAM'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    combiner = SmartCombiner()
    last_ui_update = 0
    ui_refresh_rate = 1/30

    try:
        while True:
            current_time = time.time()

            if current_time - last_ui_update < ui_refresh_rate:
                key = cv2.waitKey(1) & 0xFF
                if key == 32:  # spacebar
                    break
                elif key == ord('r'):
                    recording = not recording
                    for cam in cameras:
                        if recording:
                            cam.start_recording()
                        else:
                            cam.stop_recording()
                time.sleep(0.001)
                continue

            frames = []
            for cam in cameras:
                with cam.lock:
                    if cam.frame is not None:
                        frames.append(cam.frame)
                    else:
                        blank = np.zeros((int(cam.height * cam.preview_scale),
                                          int(cam.width * cam.preview_scale), 3), dtype=np.uint8)
                        frames.append(blank)

            combined = combiner.combine_frames(frames, cols=2)
            if combined is not None:
                cv2.imshow(window_name, combined)
                last_ui_update = current_time

            key = cv2.waitKey(1) & 0xFF
            if key == 32:  # spacebar
                break
            elif key == ord('r'):
                recording = not recording
                for cam in cameras:
                    if recording:
                        cam.start_recording()
                    else:
                        cam.stop_recording()

    except KeyboardInterrupt:
        print("Interrupted by user. Cleaning up...")
    finally:
        for cam in cameras:
            cam.running = False
            cam.join()
        cv2.destroyAllWindows()
        print("Program exited cleanly.")

# --- Run Main ---
if __name__ == "__main__":
    main()
