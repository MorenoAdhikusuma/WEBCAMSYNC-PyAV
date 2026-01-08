import sys, os, time, csv, subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from functools import partial
import requests
import cv2
import numpy as np
import threading
from queue import Queue, Empty

# --- 3rd party
from pygrabber.dshow_graph import FilterGraph
from PyQt5.QtCore import QThread, pyqtSignal, Qt, pyqtSlot, QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QTextEdit, QMessageBox, QComboBox, QLineEdit, QCheckBox
)

# --- COM for device enumeration thread
import pythoncom

# --- Frame extract deps (used by built-in extractor)
import av
import piexif

# ============================ Time Helpers (UTC+7) ============================
TZ_UTC7 = timezone(timedelta(hours=7))

def now_epoch_ms() -> int:
    return int(time.time() * 1000)

def epoch_ms_to_iso_utc7(epoch_ms: int) -> str:
    """
    Return ISO-8601 with milliseconds in UTC+7, e.g. 2025-10-16T01:20:29.901+07:00
    """
    dt_local = datetime.fromtimestamp(epoch_ms / 1000.0, tz=timezone.utc).astimezone(TZ_UTC7)
    base = dt_local.strftime("%Y-%m-%dT%H:%M:%S")
    ms = int(dt_local.microsecond / 1000)
    tz_raw = dt_local.strftime("%z")
    tz_fmt = tz_raw[:3] + ":" + tz_raw[3:]
    return f"{base}.{ms:03d}{tz_fmt}"

def make_session_folder(base_dir="./sessions", custom_name="") -> tuple[str, str, str, str]:
    ts = datetime.now(TZ_UTC7).strftime("%Y%m%d_%H%M%S")
    if custom_name.strip():
        safe = "".join(c for c in custom_name if c.isalnum() or c in (' ', '-', '_')).strip()
        folder = f"{ts}_{safe}"
    else:
        folder = f"session_{ts}"

    main = Path(base_dir) / folder
    main.mkdir(parents=True, exist_ok=False)
    cams = main / "cameras"; imu = main / "imu"; csvp = main / "csv"
    cams.mkdir(exist_ok=True); imu.mkdir(exist_ok=True); csvp.mkdir(exist_ok=True)

    (main / "README.txt").write_text(
        f"Fall Detection Recording Session\nCreated: {datetime.now(TZ_UTC7):%Y-%m-%d %H:%M:%S %z}\n"
        f"{'='*60}\n\n"
        f"📁 {folder}/\n"
        f"  ├── host_start_ms.txt\n"
        f"  ├── cameras/ (cam0.mp4, cam1.mp4, cam2.mp4)\n"
        f"  ├── csv/ (cam0_frames.csv, cam1_frames.csv, cam2_frames.csv)\n"
        f"  └── imu/ (imu.csv)\n",
        encoding="utf-8"
    )
    return str(main), str(cams), str(imu), str(csvp)

def ffmpeg_cmd_nvenc(out_path, w=1280, h=720, fps=20):
    # IMPORTANT:
    # We keep input "-r fps" because we will guarantee we write exactly fps frames/sec to stdin
    return [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
        "-c:v", "h264_nvenc", "-preset", "p1", "-tune", "ull",
        "-rc:v", "vbr", "-b:v", "5M",
        "-pix_fmt", "yuv420p", "-r", str(fps),
        out_path,
    ]

def esp_request(base: str, endpoint: str, label: str, log_cb):
    try:
        r = requests.get(f"{base.rstrip('/')}{endpoint}", timeout=3)
        log_cb(f"{label}: {r.status_code} {r.text}")
    except Exception as e:
        log_cb(f"{label} error: {e}")

def parse_esp_csv(line: str):
    # timestamp,tick_ms,seq,ax,ay,az,gx,gy,gz,temp
    parts = line.strip().split(",", 10)
    if len(parts) < 10:
        return None
    try:
        t, tick, seq, ax, ay, az, gx, gy, gz, temp = parts[:10]
        return {
            "timestamp": t,
            "tick_ms": int(tick),
            "seq": int(seq),
            "ax": float(ax), "ay": float(ay), "az": float(az),
            "gx": float(gx), "gy": float(gy), "gz": float(gz),
            "temp": float(temp)
        }
    except Exception:
        return None

# ============================ Host↔IMU Clock Sync (NO firmware changes) ============================
class ClockSync:
    """
    Map host monotonic time -> ESP tick time.
    Model: dev_ns ≈ a * host_ns + b, where dev_ns = tick_ms * 1e6.
    Updated online with sliding-window linear regression.
    """
    def __init__(self, max_points=300):
        self.max_points = max_points
        self.host = []  # host_ns
        self.dev = []   # dev_ns
        self.a = 1.0
        self.b = 0.0
        self.ready = False
        self._lock = threading.Lock()

    def add(self, host_ns: int, tick_ms: int):
        dev_ns = int(tick_ms) * 1_000_000
        with self._lock:
            self.host.append(int(host_ns))
            self.dev.append(int(dev_ns))
            if len(self.host) > self.max_points:
                self.host.pop(0); self.dev.pop(0)
            if len(self.host) < 30:
                return

            x = np.array(self.host, dtype=np.float64)
            y = np.array(self.dev,  dtype=np.float64)
            x_mean = x.mean(); y_mean = y.mean()
            denom = ((x - x_mean)**2).sum()
            if denom <= 0:
                return
            a = ((x - x_mean) * (y - y_mean)).sum() / denom
            b = y_mean - a * x_mean

            # sanity clamp: a should be close to 1.0
            if 0.95 <= a <= 1.05:
                self.a = float(a)
                self.b = float(b)
                self.ready = True

    def host_to_dev_ns(self, host_ns: int) -> int:
        with self._lock:
            return int(self.a * int(host_ns) + self.b)

# ============================ AI Anonymization (Head Blur via YuNet) ============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
YUNET_MODEL_PATH = os.path.join(BASE_DIR, "face_detection_yunet_2023mar.onnx")

yunet_detector = None
_yunet_input_size = None
yunet_lock = threading.Lock()
_HAS_YUNET = False
_YUNET_DEVICE = "none"

if os.path.exists(YUNET_MODEL_PATH):
    try:
        yunet_detector = cv2.FaceDetectorYN.create(
            model=YUNET_MODEL_PATH,
            config="",
            input_size=(300, 300),
            score_threshold=0.5,
            nms_threshold=0.3,
            top_k=5000
        )
        _YUNET_DEVICE = "cpu"
        print("[AI] YuNet running on CPU (OpenCV pip wheels do not support CUDA backend switching)")
        _HAS_YUNET = True
    except Exception as e:
        print(f"[AI] Failed to load YuNet model: {e}")
        yunet_detector = None
        _HAS_YUNET = False
        _YUNET_DEVICE = "none"
else:
    print("[AI] YuNet model not found. Place 'face_detection_yunet_2023mar.onnx' beside this script.")
    yunet_detector = None
    _HAS_YUNET = False
    _YUNET_DEVICE = "none"

def anonymize_head_region(frame):
    """
    Blur face/head region using YuNet (OpenCV FaceDetectorYN).
    NOTE: In your setup this runs CPU, so it's expensive.
    We'll run it off the capture clock, so it cannot cause speed-up/drift.
    """
    global _yunet_input_size
    if not _HAS_YUNET or yunet_detector is None:
        return frame

    h, w = frame.shape[:2]
    if h == 0 or w == 0:
        return frame

    with yunet_lock:
        if _yunet_input_size != (w, h):
            yunet_detector.setInputSize((w, h))
            _yunet_input_size = (w, h)
        try:
            ok, faces = yunet_detector.detect(frame)
        except Exception:
            return frame

    if not ok or faces is None or len(faces) == 0:
        return frame

    faces = np.array(faces)
    valid = faces[faces[:, 4] >= 0.5]
    if valid.shape[0] == 0:
        return frame

    best = valid[np.argmax(valid[:, 4])]
    x, y, bw, bh, score = best[:5]
    x1 = int(x); y1 = int(y); x2 = int(x + bw); y2 = int(y + bh)

    x1 = max(0, min(x1, w - 1))
    x2 = max(0, min(x2, w - 1))
    y1 = max(0, min(y1, h - 1))
    y2 = max(0, min(y2, h - 1))
    if x2 <= x1 or y2 <= y1:
        return frame

    face_h = y2 - y1
    head_top = max(0, y1 - int(face_h * 0.5))
    head_bottom = min(h, y2 + int(face_h * 0.3))
    y1h = head_top; y2h = head_bottom
    if y2h <= y1h:
        return frame

    roi = frame[y1h:y2h, x1:x2]
    if roi.size == 0:
        return frame

    k = max(31, ((max(x2 - x1, y2h - y1h) // 7) | 1))
    blurred = cv2.GaussianBlur(roi, (k, k), 0)
    frame[y1h:y2h, x1:x2] = blurred
    return frame

# ============================ IMU Serial Logger ============================
import serial

class SerialIMULogger(QThread):
    log = pyqtSignal(str)
    imu_packet = pyqtSignal(dict)

    def __init__(self, port: str, baud: int, csv_path: str, sync_cb=None):
        super().__init__()
        self.port = port
        self.baud = baud
        self.csv_path = csv_path
        self._running = False
        self.ser = None
        self.sync_cb = sync_cb

    def run(self):
        try:
            self.ser = serial.Serial(self.port, self.baud, timeout=1)
            self.log.emit(f"✅ Serial open {self.port} @ {self.baud}")
        except Exception as e:
            self.log.emit(f"❌ Serial open error: {e}")
            return

        try:
            with open(self.csv_path, "w", newline="", encoding="utf-8", buffering=16384) as f:
                w = csv.writer(f)
                # host_ns + firmware fields
                w.writerow(["host_ns","timestamp","tick_ms","seq","ax","ay","az","gx","gy","gz","temp"])
                self._running = True
                pkt = 0; flush_iv = 20; empty = 0
                while self._running:
                    try:
                        line = self.ser.readline().decode("utf-8", errors="replace")
                        host_ns = time.perf_counter_ns()  # monotonic arrival timestamp
                        if not line:
                            empty += 1
                            if empty == 50:
                                self.log.emit("⚠️ No IMU data yet...")
                            continue
                        parsed = parse_esp_csv(line)
                        if not parsed:
                            continue
                        empty = 0

                        # feed clock sync model
                        if self.sync_cb:
                            try:
                                self.sync_cb(host_ns, parsed["tick_ms"])
                            except Exception:
                                pass

                        w.writerow([
                            host_ns,
                            parsed["timestamp"], parsed["tick_ms"], parsed["seq"],
                            parsed["ax"], parsed["ay"], parsed["az"],
                            parsed["gx"], parsed["gy"], parsed["gz"],
                            parsed["temp"]
                        ])
                        pkt += 1
                        if pkt == 1:
                            self.log.emit(f"✅ First IMU seq={parsed['seq']}")
                        if pkt % flush_iv == 0:
                            f.flush()
                        self.imu_packet.emit(parsed)
                    except Exception:
                        continue
        finally:
            if self.ser:
                try: self.ser.close()
                except: pass
            self.log.emit("IMU Serial logger exited")

    def stop(self):
        self._running = False

# ============================ Capture Threads ============================
class PreviewThread(QThread):
    frame_ready = pyqtSignal(QImage)
    log = pyqtSignal(str)
    failed = pyqtSignal(int)
    loading = pyqtSignal(int)

    def __init__(self, device_index: int, width=1280, height=720, fps=20, anonymize: bool=False):
        super().__init__()
        self.idx = device_index; self.w = width; self.h = height; self.fps = fps
        self._cap = None; self._running = False
        self._preview_w = 640; self._preview_h = 360
        self._rgb_stride = None
        self.anonymize = anonymize

    def _open(self) -> bool:
        self.loading.emit(self.idx)
        os.environ['OPENCV_VIDEOIO_DEBUG'] = '0'
        cap = cv2.VideoCapture(self.idx, cv2.CAP_DSHOW)
        if not cap.isOpened():
            self.failed.emit(self.idx); return False
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.h)
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        for _ in range(2):
            cap.read(); time.sleep(0.02)
        self._cap = cap; return True

    def run(self):
        if not self._open(): return
        self._running = True
        frame_period = 1.0 / max(1, int(self.fps))
        next_t = time.perf_counter()
        err = 0; max_err=10
        self.log.emit(f"Preview {self.idx} started")
        try:
            while self._running:
                now = time.perf_counter()
                if now < next_t:
                    time.sleep(next_t - now); continue
                ret, frame = self._cap.read()
                if not ret or frame is None or frame.size == 0:
                    err += 1
                    if err > max_err: break
                    time.sleep(0.01); continue
                err = 0

                if self.anonymize:
                    frame = anonymize_head_region(frame)

                if (frame.shape[1], frame.shape[0]) != (self._preview_w, self._preview_h):
                    frame = cv2.resize(frame, (self._preview_w, self._preview_h), interpolation=cv2.INTER_NEAREST)
                try:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                except cv2.error:
                    continue
                h, w, ch = rgb.shape
                if self._rgb_stride is None:
                    self._rgb_stride = ch * w
                q = QImage(rgb.data, w, h, self._rgb_stride, QImage.Format_RGB888).copy()
                self.frame_ready.emit(q)

                next_t += frame_period
        finally:
            try: self._cap.release()
            except: pass
            self._cap = None

    def stop(self):
        self._running = False


class CameraThread(QThread):
    frame_ready = pyqtSignal(QImage)
    log = pyqtSignal(str)

    def __init__(self, device_index: int, out_path: str, out_csv: str, host_start_ms: int,
                 width=1280, height=720, fps=20, anonymize: bool=False):
        super().__init__()
        self.idx = device_index
        self.out_path = out_path
        self.out_csv = out_csv
        self.host_start_ms = host_start_ms
        self.w = width; self.h = height; self.fps = int(fps)
        self.anonymize = anonymize

        self._running = False
        self.proc = None
        self.frame_index = 0

        # injected by MainWindow
        self.clock_sync: ClockSync | None = None

        # queues and worker state
        self._raw_q = Queue(maxsize=2)    # raw frames (latest-only)
        self._out_q = Queue(maxsize=2)    # anonymized frames (latest-only)
        self._ai_thread = None

    def _put_latest(self, q: Queue, item):
        """
        Latest-only queue:
        - if full, drop the oldest and put the newest
        """
        try:
            q.put_nowait(item)
        except Exception:
            try:
                _ = q.get_nowait()
            except Exception:
                pass
            try:
                q.put_nowait(item)
            except Exception:
                pass

    def _ai_worker(self):
        """
        Heavy step (YuNet) runs here, NOT on the 20fps clock.
        Produces anonymized frames.
        """
        font = cv2.FONT_HERSHEY_SIMPLEX
        fscale=0.7; thick=2
        white=(255,255,255); black=(0,0,0)
        pad=6; tx=10; ty=self.h-10

        while self._running:
            try:
                item = self._raw_q.get(timeout=0.2)
            except Empty:
                continue

            if item is None:
                break

            frame, ts_iso = item

            try:
                if self.anonymize:
                    frame = anonymize_head_region(frame)
            except Exception:
                # if AI fails, still output original frame (privacy risk if model missing)
                # but since user requires anonymization, we at least blur with a big box fallback:
                pass

            # overlay timestamp on output frame (cheap, ok here)
            try:
                (tw, th), _ = cv2.getTextSize(ts_iso, font, fscale, thick)
                cv2.rectangle(frame, (tx-pad, ty-th-pad), (tx+tw+pad, ty+pad), black, cv2.FILLED)
                cv2.putText(frame, ts_iso, (tx,ty), font, fscale, white, thick, cv2.LINE_AA)
            except Exception:
                pass

            self._put_latest(self._out_q, frame)

    def run(self):
        cap = None
        csv_f = None
        try:
            cap = cv2.VideoCapture(self.idx, cv2.CAP_DSHOW)
            if not cap.isOpened():
                self.log.emit(f"Cam {self.idx} failed to open")
                return

            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.h)
            cap.set(cv2.CAP_PROP_FPS, self.fps)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

            cmd = ffmpeg_cmd_nvenc(self.out_path, self.w, self.h, self.fps)
            self.proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=self.w*self.h*3*4
            )

            csv_f = open(self.out_csv, "w", newline="", encoding="utf-8", buffering=16384)
            wcsv = csv.writer(csv_f)
            # include host monotonic + estimated IMU-time (ns)
            wcsv.writerow(["frame_index","host_ns","esp_est_ns","epoch_ms","timestamp_iso_utc7"])

            self._running = True
            self.log.emit(f"Recording cam{self.idx} -> {self.out_path}")

            # start AI worker
            self._ai_thread = threading.Thread(target=self._ai_worker, daemon=True)
            self._ai_thread.start()

            frame_period = 1.0 / max(1, self.fps)
            t0 = time.perf_counter()
            next_t = t0

            last_raw = None           # last captured raw frame (for capture duplication)
            last_out = None           # last anonymized frame (for output duplication)

            # drift monitor
            while self._running:
                now = time.perf_counter()
                if now < next_t:
                    time.sleep(next_t - now)
                    continue

                # ---- CAPTURE (lightweight, on hard 20fps clock) ----
                ret, frame = cap.read()
                host_ns = time.perf_counter_ns()

                if not ret or frame is None:
                    if last_raw is None:
                        # can't produce anything yet
                        next_t += frame_period
                        continue
                    frame = last_raw.copy()
                else:
                    last_raw = frame.copy()

                if frame.shape[1] != self.w or frame.shape[0] != self.h:
                    frame = cv2.resize(frame, (self.w, self.h), interpolation=cv2.INTER_LINEAR)

                epoch = now_epoch_ms()
                ts_iso = epoch_ms_to_iso_utc7(epoch)

                esp_est_ns = -1
                if self.clock_sync and self.clock_sync.ready:
                    try:
                        esp_est_ns = self.clock_sync.host_to_dev_ns(host_ns)
                    except Exception:
                        esp_est_ns = -1

                wcsv.writerow([self.frame_index, host_ns, esp_est_ns, epoch, ts_iso])
                if self.frame_index % 30 == 0:
                    try: csv_f.flush()
                    except: pass

                # send raw frame to AI worker (latest-only)
                # AI worker will blur + draw timestamp
                self._put_latest(self._raw_q, (frame.copy(), ts_iso))

                # ---- ENCODE (also fixed 20fps) ----
                # Try to get newest anonymized frame; else reuse last_out
                try:
                    while True:
                        last_out = self._out_q.get_nowait()
                except Empty:
                    pass

                if last_out is None:
                    # still nothing produced by AI yet; safest behavior:
                    # output a heavily blurred whole-frame fallback to preserve anonymity
                    # until AI starts producing.
                    fallback = cv2.GaussianBlur(frame, (51, 51), 0)
                    out_frame = fallback
                else:
                    out_frame = last_out

                # preview (decimated)
                if self.frame_index % 3 == 0:
                    try:
                        rgb = cv2.cvtColor(out_frame, cv2.COLOR_BGR2RGB)
                        h,w,ch = rgb.shape
                        q = QImage(rgb.data, w, h, ch*w, QImage.Format_RGB888).copy()
                        self.frame_ready.emit(q)
                    except Exception:
                        pass

                try:
                    self.proc.stdin.write(out_frame.tobytes())
                except Exception as e:
                    self.log.emit(f"ffmpeg write err cam{self.idx}: {e}")
                    break

                self.frame_index += 1
                next_t += frame_period

                # drift logging (video time should track real time closely)
                if self.frame_index % 100 == 0:
                    real = time.perf_counter() - t0
                    video = self.frame_index / self.fps
                    self.log.emit(f"[cam{self.idx}] video={video:.2f}s real={real:.2f}s drift={(real-video)*1000:.1f}ms")

        except Exception as e:
            self.log.emit(f"Cam {self.idx} error: {e}")

        finally:
            self._running = False
            # stop AI thread
            try:
                self._put_latest(self._raw_q, None)
            except Exception:
                pass
            if self._ai_thread:
                try: self._ai_thread.join(timeout=2)
                except Exception: pass

            if csv_f:
                try: csv_f.close()
                except: pass

            if self.proc and self.proc.stdin:
                try:
                    self.proc.stdin.close()
                    self.proc.wait(timeout=5)
                except Exception:
                    try: self.proc.kill()
                    except: pass

            if cap:
                try: cap.release()
                except: pass

            self.log.emit(f"Cam {self.idx} stopped (frames: {self.frame_index})")

    def stop(self):
        self._running = False

# ============================ Frame Extraction (built-in) ============================
class FrameExtractThread(QThread):
    log = pyqtSignal(str); progress = pyqtSignal(str); finished = pyqtSignal()
    def __init__(self, session_folder: str, fps: int = 20):
        super().__init__()
        self.session_folder = session_folder; self.fps=fps; self._running=False

    def stop(self): self._running=False

    def run(self):
        self._running=True
        try:
            cams = os.path.join(self.session_folder, "cameras")
            out_root = os.path.join(self.session_folder, "frames")
            if not os.path.exists(cams):
                self.log.emit(f"❌ Cameras folder not found: {cams}"); return
            VIDEO_EXTS = ('.mp4', '.avi', '.mov', '.mkv', '.mts')
            vids = [f for f in os.listdir(cams) if f.lower().endswith(VIDEO_EXTS)]
            if not vids:
                self.log.emit("⚠️ No videos found"); return
            os.makedirs(out_root, exist_ok=True)
            self.log.emit(f"📹 Found {len(vids)} video(s)")

            for i, vf in enumerate(vids, 1):
                if not self._running:
                    self.log.emit("⚠️ Extraction cancelled"); return
                vpath = os.path.join(cams, vf)
                vname = os.path.splitext(vf)[0]
                out_dir = os.path.join(out_root, vname)
                os.makedirs(out_dir, exist_ok=True)
                self.progress.emit(f"[{i}/{len(vids)}] {vf}")
                frames = self._extract_one(vpath, out_dir)
                self.log.emit(f"   ✅ {frames} frames → {out_dir}")
            self.progress.emit("✅ Complete!")
        except Exception as e:
            self.log.emit(f"❌ Extraction error: {e}")
        finally:
            self.finished.emit()

    def _extract_one(self, video_path: str, output_dir: str) -> int:
        cnt=0
        container = av.open(video_path); stream = container.streams.video[0]
        base_dt_utc = datetime.now(timezone.utc)
        try:
            for frame in container.decode(video=0):
                if not self._running: break
                if frame.pts is None: continue
                offset = float(frame.pts * stream.time_base)
                ts_utc7 = (base_dt_utc + timedelta(seconds=offset)).astimezone(TZ_UTC7)
                ms = int(ts_utc7.microsecond/1000)
                tz_raw = ts_utc7.strftime("%z"); tz_fmt = tz_raw[:3] + ":" + tz_raw[3:]
                base = ts_utc7.strftime("%Y-%m-%dT%H-%M-%S")
                fname = f"{base}.{ms:03d}{tz_fmt.replace(':','-')}.jpg"
                out = os.path.join(output_dir, fname)

                img = frame.to_ndarray(format="bgr24")
                self._save_jpeg_with_exif(img, out, ts_utc7)
                cnt += 1
                if cnt % 100 == 0:
                    self.progress.emit(f"   Extracted {cnt} frames...")
        finally:
            container.close()
        return cnt

    def _save_jpeg_with_exif(self, img, filepath: str, dt_local: datetime):
        exif_time = dt_local.strftime("%Y:%m:%d %H:%M:%S")
        exif_dict = {
            "0th": {piexif.ImageIFD.DateTime: exif_time},
            "Exif": {
                piexif.ExifIFD.DateTimeOriginal: exif_time,
                piexif.ExifIFD.DateTimeDigitized: exif_time
            }
        }
        exif_bytes = piexif.dump(exif_dict)
        ok = cv2.imwrite(filepath, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if not ok: raise RuntimeError(f"Write fail: {filepath}")
        try:
            piexif.insert(exif_bytes, filepath)
        except Exception:
            pass

# ============================ MainWindow ============================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FALL DETECTION RECORDER")
        self.setMinimumSize(1300, 900)

        self.clock_sync = ClockSync()  # <<<<<< anchor video to IMU hardware time

        central = QWidget()
        root = QVBoxLayout()
        central.setLayout(root)
        self.setCentralWidget(central)

        # --- Cameras area
        cam_row = QHBoxLayout(); self.cam_inputs=[]; self.preview_labels=[]
        for i in range(3):
            col = QVBoxLayout()
            col.addWidget(QLabel(f"Camera {i}:"))
            cb = QComboBox(); cb.setMaximumWidth(360); col.addWidget(cb); self.cam_inputs.append(cb)
            pv = QLabel(); pv.setFixedSize(420,240); pv.setStyleSheet("background:#111;border:1px solid #444"); pv.setAlignment(Qt.AlignCenter)
            col.addWidget(pv); self.preview_labels.append(pv)
            cam_row.addLayout(col)
        root.addLayout(cam_row)

        # --- Refresh/preview controls
        rr = QHBoxLayout()
        self.refresh_btn = QPushButton("Refresh Cameras")
        self.start_preview_btn = QPushButton("Start Preview")
        self.stop_preview_btn = QPushButton("Stop Preview")
        self.preview_status_label = QLabel("Preview: Stopped")
        self.preview_status_label.setStyleSheet("color:#888;font-weight:bold;")
        for w in [self.refresh_btn, self.start_preview_btn, self.stop_preview_btn, self.preview_status_label]:
            rr.addWidget(w)
        rr.addStretch(); root.addLayout(rr)

        # --- AI Anonymization toggle
        self.enable_anonymization = False
        self.toggle_anonymize = QCheckBox("Enable AI Face Anonymization (head blur, YuNet)")
        self.toggle_anonymize.stateChanged.connect(self.on_toggle_anonymization)
        if not _HAS_YUNET:
            self.toggle_anonymize.setToolTip("YuNet model not found: face_detection_yunet_2023mar.onnx")
        root.addWidget(self.toggle_anonymize)

        # --- IMU + ESP controls
        imu_row = QHBoxLayout()
        imu_row.addWidget(QLabel("IMU (Serial):"))
        self.serial_port_edit = QLineEdit("COM7"); imu_row.addWidget(self.serial_port_edit)
        imu_row.addWidget(QLabel("baud:"))
        self.serial_baud_edit = QLineEdit("115200"); self.serial_baud_edit.setMaximumWidth(100); imu_row.addWidget(self.serial_baud_edit)
        self.btn_imu_only_start = QPushButton("Start IMU Monitor")
        self.btn_imu_only_stop = QPushButton("Stop IMU Monitor"); self.btn_imu_only_stop.setEnabled(False)
        imu_row.addWidget(self.btn_imu_only_start); imu_row.addWidget(self.btn_imu_only_stop)
        root.addLayout(imu_row)

        esp_row = QHBoxLayout()
        esp_row.addWidget(QLabel("ESP32 IP:"))
        self.esp_ip_edit = QLineEdit("http://192.168.4.1"); esp_row.addWidget(self.esp_ip_edit)
        self.btn_esp_start = QPushButton("Start Stream"); self.btn_esp_stop = QPushButton("Stop Stream")
        self.btn_esp_recal = QPushButton("Recalibrate")
        self.esp_delay_edit = QLineEdit("20"); self.esp_delay_edit.setMaximumWidth(60)
        self.btn_esp_delay = QPushButton("Set Delay")
        for w in [self.btn_esp_start, self.btn_esp_stop, self.btn_esp_recal, QLabel("Delay ms:"), self.esp_delay_edit, self.btn_esp_delay]:
            esp_row.addWidget(w)
        root.addLayout(esp_row)

        # --- IMU realtime numeric panel
        grid = QGridLayout(); self.imu_labels={}
        fields = ["seq","tick_ms","ax","ay","az","gx","gy","gz","temp"]
        grid.addWidget(QLabel("IMU (Realtime):"),0,0,1,2)
        for r,k in enumerate(fields, start=1):
            kL=QLabel(k+":"); vL=QLabel("--"); self.imu_labels[k]=vL
            grid.addWidget(kL,r,0); grid.addWidget(vL,r,1)
        root.addLayout(grid)

        # --- Session name
        frow = QHBoxLayout()
        frow.addWidget(QLabel("Session Name (optional):"))
        self.session_name_edit = QLineEdit(); self.session_name_edit.setPlaceholderText("e.g., Fall_Test_01")
        self.session_name_edit.setMaximumWidth(400); frow.addWidget(self.session_name_edit); frow.addStretch()
        root.addLayout(frow)

        # --- Start/Stop/Open
        brow = QHBoxLayout()
        self.start_btn = QPushButton("Start Recording")
        self.stop_btn = QPushButton("Stop Recording"); self.stop_btn.setEnabled(False)
        self.open_btn = QPushButton("Open Session Folder")
        brow.addWidget(self.start_btn); brow.addWidget(self.stop_btn); brow.addWidget(self.open_btn)
        root.addLayout(brow)

        # --- Frame extraction
        exr = QHBoxLayout()
        exr.addWidget(QLabel("Frame Extraction:"))
        self.extract_fps_edit = QLineEdit("20"); self.extract_fps_edit.setMaximumWidth(60); exr.addWidget(QLabel("FPS:")); exr.addWidget(self.extract_fps_edit)
        self.extract_btn = QPushButton("Extract Frames from Session")
        self.extract_status_label = QLabel(""); self.extract_status_label.setStyleSheet("color:#888;font-style:italic;")
        exr.addWidget(self.extract_btn); exr.addWidget(self.extract_status_label); exr.addStretch()
        root.addLayout(exr)

        # --- Log
        self.log = QTextEdit(); self.log.setReadOnly(True); root.addWidget(self.log)

        # --- State
        self.session_folder=None; self.host_start_ms=None
        self.imu_thread=None; self.cam_threads=[]; self.preview_threads=[]
        self.is_previewing=False; self.is_recording=False; self.extract_thread=None
        self.latest_imu_packet=None; self.imu_update_timer=None
        self._cached_device_list=None; self._device_cache_time=0

        # --- Wire
        self.refresh_btn.clicked.connect(self.refresh_cameras)
        self.start_preview_btn.clicked.connect(self.start_preview)
        self.stop_preview_btn.clicked.connect(self.stop_preview)
        self.start_btn.clicked.connect(self.start_session)
        self.stop_btn.clicked.connect(self.stop_session)
        self.open_btn.clicked.connect(self.open_session_folder)
        self.btn_esp_start.clicked.connect(lambda: esp_request(self.esp_base(), "/stream/start", "ESP start", self.log_msg))
        self.btn_esp_stop.clicked.connect(lambda: esp_request(self.esp_base(), "/stream/stop", "ESP stop", self.log_msg))
        self.btn_esp_recal.clicked.connect(lambda: esp_request(self.esp_base(), "/imu/recalibrate", "ESP recal", self.log_msg))
        self.btn_esp_delay.clicked.connect(self.esp_set_delay)
        self.btn_imu_only_start.clicked.connect(self.start_imu_only)
        self.btn_imu_only_stop.clicked.connect(self.stop_imu_only)
        self.extract_btn.clicked.connect(self.extract_frames)

        self.stop_preview_btn.setEnabled(False); self.start_preview_btn.setEnabled(True)

        QTimer.singleShot(200, self.refresh_cameras)
        QTimer.singleShot(400, self.start_preview)

    # ---------- Small utils ----------
    def log_msg(self, s: str):
        ts = datetime.now(TZ_UTC7).strftime("%H:%M:%S")
        self.log.append(f"[{ts}] {s}")

    def esp_base(self) -> str:
        return self.esp_ip_edit.text().strip().rstrip("/")

    def esp_set_delay(self):
        try:
            val=int(self.esp_delay_edit.text())
        except ValueError:
            self.log_msg("ESP delay invalid"); return
        esp_request(self.esp_base(), f"/imu/delay?value={val}", "ESP set delay", self.log_msg)

    # ---------- AI toggle ----------
    def on_toggle_anonymization(self, state: int):
        self.enable_anonymization = (state == Qt.Checked)
        status = "ON" if self.enable_anonymization else "OFF"
        if not _HAS_YUNET:
            self.log_msg("AI Anonymization enabled but YuNet ONNX model is missing. "
                         "Place 'face_detection_yunet_2023mar.onnx' beside the script.")
        else:
            self.log_msg(f"AI Anonymization (YuNet, {_YUNET_DEVICE}) : {status}")

    # ---------- Camera enumeration ----------
    def refresh_cameras(self):
        if self.is_previewing or self.preview_threads:
            self.stop_preview()
        for cb in self.cam_inputs:
            cb.clear(); cb.addItem("Loading cameras...", -1); cb.setEnabled(False)

        self._enum_thread = QThread()
        def worker():
            pythoncom.CoInitialize()
            try:
                now=time.time()
                if self._cached_device_list and (now - self._device_cache_time) < 5:
                    self._temp_device_names = self._cached_device_list; return
                graph = FilterGraph()
                names = graph.get_input_devices()
                self._temp_device_names = names
                self._cached_device_list = names; self._device_cache_time = now
            except Exception:
                self._temp_device_names = []
            finally:
                pythoncom.CoUninitialize()

        self._enum_thread.run = worker
        self._enum_thread.finished.connect(self._on_enum_done)
        self._enum_thread.start()

    def _on_enum_done(self):
        for cb in self.cam_inputs:
            cb.setEnabled(True); cb.blockSignals(True); cb.clear()
        names = getattr(self, "_temp_device_names", [])
        for i, cb in enumerate(self.cam_inputs):
            if names:
                for idx, name in enumerate(names):
                    cb.addItem(f"{name} (Device {idx})", idx)
            else:
                for idx in range(3):
                    cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
                    if cap.isOpened():
                        cb.addItem(f"Device {idx}", idx)
                        cap.release()
            if cb.count()==0: cb.addItem("No camera found", -1)
            cb.setCurrentIndex(min(i, cb.count()-1))
            cb.blockSignals(False)
        self.log_msg("Camera list refreshed")

    # ---------- Preview ----------
    def start_preview(self):
        if self.is_previewing:
            self.log_msg("Preview already running"); return
        if self.is_recording:
            self.log_msg("Cannot start preview while recording"); return
        self.log_msg("Starting preview...")
        for i, cb in enumerate(self.cam_inputs):
            dev = cb.currentData()
            if dev is None or dev < 0:
                self.preview_labels[i].setText("No Camera"); continue
            self.preview_labels[i].setText("Loading camera...")
            t = PreviewThread(device_index=dev, anonymize=self.enable_anonymization)
            t.log.connect(self.log_msg)
            t.frame_ready.connect(partial(self.update_preview, i))
            t.loading.connect(partial(self.on_camera_loading, i))
            t.failed.connect(partial(self.on_camera_failed, i))
            t.start(); self.preview_threads.append(t)

        self.is_previewing=True
        self.preview_status_label.setText("Preview: Running")
        self.preview_status_label.setStyleSheet("color:#00ff00;font-weight:bold;")
        self.start_preview_btn.setEnabled(False); self.stop_preview_btn.setEnabled(True)
        self.log_msg("Preview started")

    def stop_preview(self):
        if not self.is_previewing and not self.preview_threads:
            self.start_preview_btn.setEnabled(True); self.stop_preview_btn.setEnabled(False); return
        self.log_msg("Stopping preview...")
        for t in self.preview_threads:
            try: t.stop(); t.wait(2000)
            except Exception: pass
        self.preview_threads=[]; self.is_previewing=False
        if not self.is_recording:
            for lab in self.preview_labels:
                lab.clear(); lab.setText("Preview Stopped")
                lab.setStyleSheet("background:#222;border:1px solid #444;color:#888")
            self.preview_status_label.setText("Preview: Stopped")
            self.preview_status_label.setStyleSheet("color:#888;font-weight:bold;")
        self.start_preview_btn.setEnabled(True); self.stop_preview_btn.setEnabled(False)
        self.log_msg("Preview stopped")

    def on_camera_loading(self, idx: int, device_index: int):
        if idx < len(self.preview_labels):
            self.preview_labels[idx].setText(f"Loading camera {device_index}...")
            self.log_msg(f"Camera {device_index} initializing...")

    def on_camera_failed(self, idx: int, device_index: int):
        if idx < len(self.preview_labels):
            self.preview_labels[idx].setText(f"Camera {device_index} failed")
            self.preview_labels[idx].setStyleSheet("background:#331111;border:1px solid #881111;color:#ff6666")
            self.log_msg(f"Camera {device_index} failed to open")

    def update_preview(self, idx: int, qimg: QImage):
        if 0 <= idx < len(self.preview_labels):
            lbl = self.preview_labels[idx]
            if not self.is_recording:
                if lbl.styleSheet() != "background:#111;border:1px solid #444":
                    lbl.setStyleSheet("background:#111;border:1px solid #444"); lbl.setText("")
            pix = QPixmap.fromImage(qimg).scaled(lbl.width(), lbl.height(), Qt.KeepAspectRatio)
            lbl.setPixmap(pix)

    # ---------- IMU realtime ----------
    @pyqtSlot(dict)
    def update_imu_preview(self, parsed: dict):
        self.latest_imu_packet = parsed
        if self.imu_update_timer is None:
            self.imu_update_timer = QTimer()
            self.imu_update_timer.timeout.connect(self._update_imu_ui)
            self.imu_update_timer.start(50)

    def _update_imu_ui(self):
        p = self.latest_imu_packet
        if not p: return
        for k in ["seq","tick_ms","ax","ay","az","gx","gy","gz","temp"]:
            lab = self.imu_labels.get(k)
            if not lab: continue
            v = p.get(k)
            if k in ("seq","tick_ms"):
                lab.setText(str(v))
            else:
                try: lab.setText(f"{float(v):.3f}")
                except: lab.setText("--")

    # ---------- IMU-only ----------
    def start_imu_only(self):
        if self.imu_thread:
            self.log_msg("IMU already running."); return
        if self.session_folder is None:
            main, _, imu_dir, _ = make_session_folder(custom_name="IMU_Monitor")
            self.log_msg(f"IMU-only folder: {main}")
            imu_csv = os.path.join(imu_dir, "imu.csv")
            self.session_folder = main
        else:
            imu_csv = os.path.join(self.session_folder, "imu", "imu.csv")

        port = self.serial_port_edit.text().strip()
        try: baud = int(self.serial_baud_edit.text().strip())
        except ValueError:
            QMessageBox.critical(self, "Error", "Invalid baudrate"); return

        self.imu_thread = SerialIMULogger(port, baud, imu_csv, sync_cb=self.clock_sync.add)
        self.imu_thread.log.connect(self.log_msg)
        self.imu_thread.imu_packet.connect(self.update_imu_preview)
        self.imu_thread.start()
        self.btn_imu_only_start.setEnabled(False); self.btn_imu_only_stop.setEnabled(True)
        self.log_msg("IMU Monitor started")

    def stop_imu_only(self):
        if not self.imu_thread:
            self.log_msg("IMU not running."); return
        self.log_msg("Stopping IMU Monitor...")
        self.imu_thread.stop(); self.imu_thread.wait(2000); self.imu_thread=None
        self.btn_imu_only_start.setEnabled(True); self.btn_imu_only_stop.setEnabled(False)
        self.log_msg("IMU Monitor stopped")

    # ---------- Recording ----------
    def start_session(self):
        if self.is_previewing:
            self.stop_preview()

        name = self.session_name_edit.text().strip()
        try:
            main, cams, imu_dir, csv_dir = make_session_folder(custom_name=name)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Cannot create session folder: {e}")
            return

        self.session_folder = main
        self.host_start_ms = now_epoch_ms()

        with open(os.path.join(main, "host_start_ms.txt"), "w", encoding="utf-8") as f:
            f.write(f"epoch_ms: {self.host_start_ms}\n")
            f.write(f"timestamp_iso_utc7: {epoch_ms_to_iso_utc7(self.host_start_ms)}\n")

        self.log_msg(f"Session: {main}\n  → Cameras: {cams}\n  → IMU: {imu_dir}\n  → CSV: {csv_dir}")
        self.log_msg(f"Host start: {self.host_start_ms} ({epoch_ms_to_iso_utc7(self.host_start_ms)})")

        # ESP auto-start (optional)
        try:
            esp_request(self.esp_base(), "/stream/start", "ESP autostart", self.log_msg)
            self.log_msg("⏳ Waiting 2s for ESP to init..."); time.sleep(2)
        except Exception:
            pass

        # Start IMU serial (feeds clock sync)
        imu_csv = os.path.join(imu_dir, "imu.csv")
        port = self.serial_port_edit.text().strip()
        try: baud = int(self.serial_baud_edit.text().strip())
        except ValueError:
            QMessageBox.critical(self, "Error", "Invalid baudrate"); return

        self.imu_thread = SerialIMULogger(port, baud, imu_csv, sync_cb=self.clock_sync.add)
        self.imu_thread.log.connect(self.log_msg)
        self.imu_thread.imu_packet.connect(self.update_imu_preview)
        self.imu_thread.start()

        # Start cameras (video hard-locked to 20fps; anonymization happens without breaking timing)
        self.cam_threads=[]
        for i, cb in enumerate(self.cam_inputs):
            dev = cb.currentData()
            if dev is None or dev < 0:
                self.log_msg(f"Skip cam{i}"); self.preview_labels[i].setText("No Camera"); continue

            self.preview_labels[i].setText("Starting recording...")
            out_video = os.path.join(cams, f"cam{i}.mp4")
            out_csv = os.path.join(csv_dir, f"cam{i}_frames.csv")

            t = CameraThread(dev, out_video, out_csv, self.host_start_ms,
                             anonymize=self.enable_anonymization)
            t.clock_sync = self.clock_sync
            t.log.connect(self.log_msg)
            t.frame_ready.connect(partial(self.update_preview, i))
            t.start(); self.cam_threads.append(t)

        self.is_recording=True
        self.preview_status_label.setText("Status: RECORDING")
        self.preview_status_label.setStyleSheet("color:#ff0000;font-weight:bold;")
        self.start_btn.setEnabled(False); self.stop_btn.setEnabled(True)
        self.start_preview_btn.setEnabled(False); self.stop_preview_btn.setEnabled(False)
        self.btn_imu_only_start.setEnabled(False); self.btn_imu_only_stop.setEnabled(False)
        self.log_msg("Recording started (IMU + cameras)")

    def stop_session(self):
        self.log_msg("Stopping session...")
        try: esp_request(self.esp_base(), "/stream/stop", "ESP autostop", self.log_msg)
        except Exception: pass

        if self.imu_update_timer:
            self.imu_update_timer.stop(); self.imu_update_timer=None

        for t in self.cam_threads:
            try: t.stop(); t.wait(8000)
            except Exception: pass
        self.cam_threads=[]

        if self.imu_thread:
            self.imu_thread.stop(); self.imu_thread.wait(2000); self.imu_thread=None

        self.is_recording=False
        self.start_btn.setEnabled(True); self.stop_btn.setEnabled(False)
        self.btn_imu_only_start.setEnabled(True); self.btn_imu_only_stop.setEnabled(False)
        self.log_msg("Session stopped. Files saved.")
        self.log_msg("Restarting preview...")
        self.start_preview()

    # ---------- Extract ----------
    def extract_frames(self):
        if not self.session_folder:
            QMessageBox.warning(self, "No Session",
                                "Record a session first (needs 'cameras' subfolder).")
            self.log_msg("⚠️ No session for extraction"); return
        if self.extract_thread and self.extract_thread.isRunning():
            QMessageBox.warning(self, "Extraction", "Frame extraction already running."); return
        try:
            fps = int(self.extract_fps_edit.text())
            if fps <= 0 or fps > 120: raise ValueError
        except ValueError:
            QMessageBox.warning(self, "Invalid FPS", "FPS must be 1-120"); return

        cams = os.path.join(self.session_folder, "cameras")
        if not os.path.exists(cams):
            QMessageBox.warning(self, "No Cameras", f"Cameras folder missing:\n{cams}")
            self.log_msg("❌ Cameras folder not found"); return
        video_count = sum(1 for f in os.listdir(cams) if f.lower().endswith(('.mp4','.avi','.mov','.mkv','.mts')))
        if video_count == 0:
            QMessageBox.warning(self, "No Videos", "No compatible videos found."); return

        if QMessageBox.question(self, "Confirm", f"Extract frames from {video_count} video(s)?",
                                QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes) != QMessageBox.Yes:
            self.log_msg("Extraction cancelled"); return

        self.log_msg(f"{'='*60}\n🎬 Frame Extraction\nSession: {self.session_folder}\nVideos: {video_count}\nTarget FPS: {fps}\n{'='*60}")

        self.extract_thread = FrameExtractThread(self.session_folder, fps)
        self.extract_thread.log.connect(self.log_msg)
        self.extract_thread.progress.connect(self.update_extract_status)
        self.extract_thread.finished.connect(self.on_extract_finished)
        self.extract_thread.start()

        self.extract_btn.setEnabled(False); self.extract_btn.setText("Extracting...")
        self.extract_status_label.setText("Processing...")
        self.extract_status_label.setStyleSheet("color:#ff9800;font-weight:bold;font-style:italic;")

    def update_extract_status(self, s: str):
        self.extract_status_label.setText(s)

    def on_extract_finished(self):
        self.extract_btn.setEnabled(True)
        self.extract_btn.setText("Extract Frames from Session")
        self.extract_status_label.setText("✅ Complete!")
        self.extract_status_label.setStyleSheet("color:#4caf50;font-weight:bold;font-style:italic;")
        if self.extract_thread and not self.extract_thread._running and self.session_folder:
            frames = os.path.join(self.session_folder, "frames")
            QMessageBox.information(self, "Done", f"Frames saved to:\n{frames}")
        QTimer.singleShot(5000, lambda: self.extract_status_label.setText(""))

    # ---------- File open ----------
    def open_session_folder(self):
        if not self.session_folder:
            QMessageBox.information(self, "No session", "No session yet"); return
        folder = os.path.abspath(self.session_folder)
        if sys.platform.startswith("win"):
            subprocess.Popen(["explorer", folder])
        else:
            try: subprocess.Popen(["xdg-open", folder])
            except Exception: pass

    # ---------- Close cleanup ----------
    def closeEvent(self, event):
        if self.is_recording or self.imu_thread:
            try: esp_request(self.esp_base(), "/stream/stop", "ESP stop (on exit)", self.log_msg)
            except Exception: pass
        for t in self.cam_threads:
            try: t.stop(); t.wait(2000)
            except Exception: pass
        for t in self.preview_threads:
            try: t.stop(); t.wait(2000)
            except Exception: pass
        if self.imu_thread:
            try: self.imu_thread.stop(); self.imu_thread.wait(2000)
            except Exception: pass
        if self.extract_thread and self.extract_thread.isRunning():
            try: self.extract_thread.stop(); self.extract_thread.wait(2000)
            except Exception: pass
        event.accept()

# ============================ Entrypoint ============================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())
