import cv2 
import threading
import time
import av
import numpy as np
import os
import queue
from datetime import datetime, timedelta, timezone
from fractions import Fraction
import subprocess
import shutil
import platform
from collections import deque
from pygrabber.dshow_graph import FilterGraph

# ----------------- CONFIG -----------------
OUTPUT_DIR = r"C:\Users\moreno\programming\Scientific_Works\senyum\Not_experiment\OUTPUT_VIDEO"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET_FPS = 20
PREVIEW_SCALE = 0.5

# --- Helper: add metadata via ffmpeg (creation_time with timezone +07:00) ---
def add_start_time_metadata(mp4_path, timestamp_utc7_iso):
    base, ext = os.path.splitext(mp4_path)
    tmp = base + "_withmeta" + ext

    ffmpeg_names = ['ffmpeg', 'ffmpeg.exe']
    ffmpeg_path = None
    for name in ffmpeg_names:
        if shutil.which(name):
            ffmpeg_path = name
            break

    if not ffmpeg_path:
        print(f"[FFMPEG] ffmpeg not found in PATH — skipping metadata for {mp4_path}")
        return

    cmd = [
        ffmpeg_path, '-y',
        '-i', mp4_path,
        '-map_metadata', '0',
        '-metadata', f'creation_time={timestamp_utc7_iso}',
        '-metadata', f'com.apple.quicktime.creationdate={timestamp_utc7_iso}',
        '-codec', 'copy',
        tmp
    ]
    print(f"[FFMPEG] Running: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        os.replace(tmp, mp4_path)
        print(f"[FFMPEG] Metadata written into {mp4_path}")
    except subprocess.CalledProcessError as e:
        print(f"[FFMPEG] Failed: {e}; stderr: {e.stderr}")
        if os.path.exists(tmp):
            os.remove(tmp)
    except Exception as e:
        print(f"[FFMPEG] Unexpected error: {e}")
        if os.path.exists(tmp):
            os.remove(tmp)

def detect_cameras():
    graph = FilterGraph()
    try:
        devices = graph.get_input_devices()
    except Exception as e:
        print(f"[detect_cameras] Error listing devices: {e}")
        devices = []

    print("Available video devices:")
    for i, d in enumerate(devices):
        print(f"  {i}: {d}")

    cameras = []
    skip_keywords = ['nvidia', 'virtual', 'obs', 'broadcast', 'asus fhd', 'integrated', 'facetime', ]

    for i, name in enumerate(devices):
        lower_name = name.lower()

        if any(kw in lower_name for kw in skip_keywords):
            continue

        # Keep only devices whose name contains 'uvc'
        if 'uvc' in lower_name:
            # Check if camera index is accessible via OpenCV with DSHOW
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            if cap.isOpened():
                # --- prefer MJPG to save USB bandwidth/CPU ---
                try:
                    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
                except Exception:
                    pass
                ret, _ = cap.read()
                cap.release()
                if ret:
                    cameras.append(i)
                    print(f"[detect] Added UVC camera {i}: {name}")
                else:
                    print(f"[detect] Camera {i} '{name}' failed to capture frame")
            else:
                print(f"[detect] Camera {i} '{name}' not openable")
    return cameras

cameras_identifier = detect_cameras()

# ---------------- CameraWorker ----------------
class CameraWorker(threading.Thread):

    def detect_best_backend(self, cam_index):
        """Test which backend works best for this camera"""
        test_results = {}
        for backend, name in [(cv2.CAP_DSHOW, 'DSHOW')]:
            try:
                cap = cv2.VideoCapture(cam_index, backend)
                if cap.isOpened():
                    ret, frame = cap.read()
                    test_results[backend] = bool(ret and frame is not None)
                    cap.release()
                    print(f"[Backend Test] {name} {'works' if test_results[backend] else 'opens but cannot read'} for camera {cam_index}")
                else:
                    test_results[backend] = False
                    print(f"[Backend Test] {name} cannot open camera {cam_index}")
            except Exception as e:
                test_results[backend] = False
                print(f"[Backend Test] {name} failed with error: {e}")

    def __init__(self, cam_index, cam_name=None):
        super().__init__()
        self.cam_index = cam_index
        self.cam_name = cam_name or f"Cam {cam_index}"
        self.cap = None
        self.open_camera()
        if not self.cap or not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera {self.cam_index}")

        # --- Prefer MJPG to reduce CPU/USB load ---
        try:
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
        except Exception:
            pass
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        except Exception:
            pass

        self.fps = TARGET_FPS
        self.configure_camera_fps()

        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
        self.preview_scale = PREVIEW_SCALE

        self.frame = None
        self.last_valid_frame = None
        self.lock = threading.Lock()
        self.running = True
        self.recording = False
        self.frame_queue = queue.Queue(maxsize=120)
        self.encoding_thread = None
        self.output = None
        self.stream = None
        self.start_time = None
        self.frame_count = 0

        self._ts = deque(maxlen=30)
        self.current_fps = 0.0

    def open_camera(self):
        system = platform.system().lower()
        tried = []
        for backend, name in [(cv2.CAP_DSHOW,'DSHOW')]:
            try:
                cap = cv2.VideoCapture(self.cam_index, backend)
                tried.append((name, self.cam_index))
                if cap.isOpened():
                    self.cap = cap
                    print(f"[{self.cam_name}] opened with {name}")
                    return
                else:
                    try: cap.release()
                    except: pass
            except Exception as e:
                print(f"[{self.cam_name}] backend {name} error: {e}")

        print(f"[{self.cam_name}] tried {tried} - none succeeded")
        raise RuntimeError(f"Cannot open camera {self.cam_index} with any backend")
    
    def close_camera(self):
        if self.cap:
            try:
                for _ in range(5):
                    self.cap.read()
                self.cap.release()
                time.sleep(0.1)
            except Exception:
                pass
            finally:
                self.cap = None
                print(f"[{self.cam_name}] camera closed")

    def configure_camera_fps(self):
        try:
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            actual = self.cap.get(cv2.CAP_PROP_FPS)
            print(f"[{self.cam_name}] requested FPS={self.fps}, CAP_PROP_FPS={actual}")
        except Exception as e:
            print(f"[{self.cam_name}] configure FPS error: {e}")

    def _update_fps(self, ts):
        self._ts.append(ts)
        if len(self._ts) >= 2:
            intervals = [t2 - t1 for t1, t2 in zip(self._ts, list(self._ts)[1:])]
            avg = sum(intervals)/len(intervals)
            self.current_fps = 1.0/avg if avg > 0 else 0.0

    def start_recording(self):
        if self.recording:
            print(f"[{self.cam_name}] already recording")
            return
        filename = os.path.join(OUTPUT_DIR, f"cam_{self.cam_index}.mp4")
        self.output = av.open(filename, mode='w')

        # --- H.264 encoder: multi-thread & low-latency, cocok untuk 3 kamera ---
        self.stream = self.output.add_stream('libx264', rate=self.fps)
        self.stream.width = self.width
        self.stream.height = self.height
        self.stream.pix_fmt = 'yuv420p'
        # GOP sekitar 2 detik (opsional, stabilkan seek)
        try:
            self.stream.gop_size = int(self.fps * 2)
        except Exception:
            pass
        # Opsi encoder (tanpa ubah struktur)
        self.stream.options = {
            'preset': 'ultrafast',     # sangat ringan
            'crf': '23',               # kualitas-bitrate balance
            'tune': 'zerolatency',     # kurangi buffering
            'threads': '0',            # auto (gunakan semua core)
            'bf': '0',                 # non-B-frames (latensi rendah)
        }

        self.recording = True
        self.start_time = time.time()
        self.frame_count = 0
        self.encoding_thread = threading.Thread(target=self._encoding_worker, daemon=True)
        self.encoding_thread.start()
        print(f"[{self.cam_name}] recording started -> {filename}")

    def _encoding_worker(self):
        while self.recording:
            try:
                item = self.frame_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if item is None:
                break
            frame_bgr, ts = item
            video_frame = av.VideoFrame.from_ndarray(frame_bgr, format='bgr24')
            pts = int((ts - self.start_time) * self.fps)
            video_frame.pts = pts
            video_frame.time_base = Fraction(1, self.fps)
            for packet in self.stream.encode(video_frame):
                self.output.mux(packet)
            self.frame_count += 1
            self.frame_queue.task_done()
        # flush encoder
        try:
            for packet in self.stream.encode(None):
                self.output.mux(packet)
        except Exception as e:
            print(f"[{self.cam_name}] flush error: {e}")
        # close file
        try:
            self.output.close()
        except Exception as e:
            print(f"[{self.cam_name}] close error: {e}")
        # write metadata
        try:
            if self.start_time:
                tz = timezone(timedelta(hours=7))
                iso = datetime.fromtimestamp(self.start_time, tz=tz).isoformat()
                filename = os.path.join(OUTPUT_DIR, f"cam_{self.cam_index}.mp4")
                add_start_time_metadata(filename, iso)
        except Exception as e:
            print(f"[{self.cam_name}] metadata add error: {e}")

    def stop_recording(self):
        if not self.recording:
            return
        self.recording = False
        try:
            self.frame_queue.put(None, timeout=1)
        except Exception:
            pass
        if self.encoding_thread:
            self.encoding_thread.join(timeout=5.0)
        print(f"[{self.cam_name}] recording stopped. frames={self.frame_count}")

    def run(self):
        frame_interval = 1.0 / self.fps
        next_time = time.time()
        display_interval = max(1, int(self.fps / 10))

        print(f"[{self.cam_name}] capture loop started")
        while self.running:
            now = time.time()
            sleep_for = next_time - now
            if sleep_for > 0:
                time.sleep(sleep_for)
            next_time += frame_interval

            try:
                ret, frame = self.cap.read()
            except Exception as e:
                print(f"[{self.cam_name}] read error: {e}")
                ret, frame = False, None

            if not ret or frame is None:
                time.sleep(0.01)
                continue

            ts = time.time()
            self._update_fps(ts)
            try:
                self.last_valid_frame = frame.copy()
            except Exception:
                self.last_valid_frame = None

            if (self.frame_count % display_interval) == 0:
                try:
                    preview = cv2.resize(frame, (0, 0), fx=self.preview_scale, fy=self.preview_scale)
                except Exception:
                    preview = frame.copy()
                utc7 = datetime.now(timezone(timedelta(hours=7)))
                time_text = utc7.strftime("%Y-%m-%d %H:%M:%S") + " +07"
                fps_text = f"FPS: {self.current_fps:.2f}"
                cv2.putText(preview, time_text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                cv2.putText(preview, fps_text, (8, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                with self.lock:
                    self.frame = preview

            if self.recording:
                rec_frame = frame.copy()
                utc7 = datetime.now(timezone(timedelta(hours=7)))
                time_text = utc7.strftime("%Y-%m-%d %H:%M:%S") + " +07"
                cv2.putText(rec_frame, time_text, (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(rec_frame, f"FPS: {self.current_fps:.2f}", (8, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
                try:
                    self.frame_queue.put((rec_frame, ts), block=False)
                except queue.Full:
                    pass

            self.frame_count += 1

        self.stop_recording()
        try:
            self.cap.release()
        except:
            pass
        print(f"[{self.cam_name}] capture loop ended")

# ---------------- SmartCombiner ----------------
class SmartCombiner:
    def __init__(self):
        self.last_grid = None
        self.last_update = 0
        self.update_interval = 1.0/30.0

    def combine_frames(self, frames, names=None):
        now = time.time()
        if self.last_grid is not None and (now - self.last_update) < self.update_interval:
            return self.last_grid
        if not frames:
            return None
        while len(frames) < 2:
            h,w = frames[0].shape[:2] if frames else (240,320)
            frames.append(np.zeros((h,w,3), dtype=np.uint8))
        if len(frames) == 2:
            h_max = max(f.shape[0] for f in frames)
            w_max = max(f.shape[1] for f in frames)
            grid = np.zeros((h_max, w_max*2, 3), dtype=np.uint8)
            h,w = frames[0].shape[:2]
            grid[0:h, 0:w] = frames[0]
            h,w = frames[1].shape[:2]
            grid[0:h, w_max:w_max+w] = frames[1]
            if names:
                cv2.putText(grid, names[0], (8, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.putText(grid, names[1], (w_max+8, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        else:
            cols = 2
            rows = (len(frames)+1)//2
            h_max = max(f.shape[0] for f in frames)
            w_max = max(f.shape[1] for f in frames)
            grid = np.zeros((h_max*rows, w_max*cols, 3), dtype=np.uint8)
            for idx, frame in enumerate(frames):
                r = idx//cols
                c = idx%cols
                h,w = frame.shape[:2]
                y = r*h_max + (h_max-h)//2
                x = c*w_max + (w_max-w)//2
                grid[y:y+h, x:x+w] = frame
        gh, gw = grid.shape[:2]
        scale = min(1080/gh, 1920/gw) if gh and gw else 1.0
        if scale < 1.0:
            grid = cv2.resize(grid, (int(gw*scale), int(gh*scale)), interpolation=cv2.INTER_AREA)
        self.last_grid = grid
        self.last_update = now
        return grid

# ---------------- Main ----------------
def main():
    print("Initializing cameras...")
    cameras = []
    camera_names = [f"Cam {i+1}" for i in range(8)]  # up to 8 names

    if not cameras_identifier:
        print("No cameras detected. Exiting.")
        return

    for idx in cameras_identifier:
        name = camera_names[len(cameras)] if len(cameras) < len(camera_names) else f"Cam {idx}"
        try:
            cam = CameraWorker(idx, cam_name=name)
            cameras.append(cam)
            time.sleep(0.5)
        except Exception as e:
            print(f"Failed to init camera {idx}: {e}")

    if not cameras:
        print("No cameras available after initialization. Exiting.")
        return

    for c in cameras:
        c.start()
        time.sleep(0.2)

    combiner = SmartCombiner()
    window_name = "MULTICAM"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    recording = False
    last_health = time.time()

    print("Controls: R = start/stop recording (all cams). SPACE = quit.")

    try:
        while True:
            now = time.time()
            if now - last_health > 5.0:
                for i, cam in enumerate(cameras):
                    if not cam.is_alive() or not cam.cap.isOpened():
                        print(f"[main] reconnecting {cam.cam_name} ...")
                        try:
                            new = CameraWorker(cam.cam_index, cam_name=cam.cam_name)
                            cam.running = False
                            cam.join()
                            cameras[i] = new
                            new.start()
                            print(f"[main] reconnected {cam.cam_name}")
                        except Exception as e:
                            print(f"[main] reconnect failed for {cam.cam_name}: {e}")
                last_health = now

            frames = []
            names = []
            for cam in cameras:
                with cam.lock:
                    if cam.frame is not None:
                        frames.append(cam.frame)
                        names.append(cam.cam_name)
                    elif cam.last_valid_frame is not None:
                        pv = cv2.resize(cam.last_valid_frame, (0,0), fx=cam.preview_scale, fy=cam.preview_scale)
                        utc7 = datetime.now(timezone(timedelta(hours=7)))
                        cv2.putText(pv, utc7.strftime("%Y-%m-%d %H:%M:%S") + " +07", (8,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0),1)
                        cv2.putText(pv, f"FPS: {cam.current_fps:.2f}", (8,40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0),1)
                        frames.append(pv)
                        names.append(cam.cam_name)
                    else:
                        h,w = int(cam.height*cam.preview_scale), int(cam.width*cam.preview_scale)
                        frames.append(np.zeros((h,w,3), dtype=np.uint8))
                        names.append(cam.cam_name)

            combined = combiner.combine_frames(frames, names)
            if combined is not None:
                cv2.imshow(window_name, combined)

            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):
                break
            elif key == ord('r'):
                recording = not recording
                if recording:
                    print("[main] START RECORDING")
                    for cam in cameras:
                        cam.start_recording()
                else:
                    print("[main] STOP RECORDING")
                    for cam in cameras:
                        cam.stop_recording()

    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        for cam in cameras:
            cam.running = False
            cam.join()
        cv2.destroyAllWindows()
        print("Exited cleanly")

if __name__ == "__main__":
    main()
