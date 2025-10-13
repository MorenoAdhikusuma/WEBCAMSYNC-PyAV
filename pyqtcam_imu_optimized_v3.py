# pyqtcam_imu_optimized.py
# === Combined single-file application (optimized IMU) ===
# Camera recorder UI from original pyqtcam.py, with improved IMU ingestion
# inspired by imu_stream_gui.py (robust CSV header resync + tolerant parsing).
#
# - Fixes: make_session_folder syntax, sturdier Serial IMU reading, shared CSV_HEADER
# - Keeps: UDP/HTTP IMU and camera threads, UI layout, file structure & behavior
#
# Requirements:
#   pip install pyqt5 pygrabber opencv-python numpy pyserial requests
#
# Run:
#   python pyqtcam_imu_optimized.py

import sys
import os
import re
import csv
import time
import socket
import subprocess
from functools import partial
from datetime import datetime
from pathlib import Path

import requests
import shutil
import serial
import cv2
import numpy as np
from pygrabber.dshow_graph import FilterGraph

from PyQt5.QtCore import Qt, QThread, pyqtSignal, QObject
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QTextEdit, QMessageBox, QComboBox, QLineEdit
)

# ---------------------------------
# Shared IMU CSV schema (from GUI)
# ---------------------------------
CSV_HEADER = ["timestamp","tick_ms","seq","ax","ay","az","gx","gy","gz","temp"]


# =====================================================
# ============== helpers.py (fixed) ===================
# =====================================================

def now_epoch_ms() -> int:
    return int(time.time() * 1000)

def make_session_folder(base_dir: str = "./sessions") -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = Path(base_dir) / f"session_{ts}"
    path.mkdir(parents=True, exist_ok=False)
    return str(path)

def sanitize_folder_component(name: str) -> str:
    name = (name or "").strip()
    name = re.sub(r'[\\/:*?"<>|]+', '', name)
    name = re.sub(r'\s+', '_', name)
    if not name:
        name = "dataset"
    return name[:60]

def make_session_folder_with_name(custom_name: str, base_dir: str = "./sessions") -> str:
    safe = sanitize_folder_component(custom_name)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = Path(base_dir) / f"{safe}_{ts}"
    path.mkdir(parents=True, exist_ok=False)
    return str(path)

def ffmpeg_cmd_for_windows_output(out_path: str, width: int = 1280, height: int = 720, fps: int = 20) -> list:
    return [
        "ffmpeg",
        "-y",
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{width}x{height}",
        "-r", str(fps),
        "-i", "-",
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-tune", "zerolatency",
        "-pix_fmt", "yuv420p",
        "-r", str(fps),
        out_path,
    ]

def fmt_hms_ms(ms_since_start: int) -> str:
    s, ms = divmod(int(ms_since_start), 1000)
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"

def parse_esp_csv(line: str):
    """
    Accept two formats:
    10 columns: timestamp,tick_ms,seq,ax,ay,az,gx,gy,gz,temp
     9 columns: tick_ms,seq,ax,ay,az,gx,gy,gz,temp  (timestamp filled with host time)
    """
    parts = (line or "").strip().split(",")
    if len(parts) == 10:
        try:
            return {
                "timestamp": parts[0],
                "tick_ms": int(parts[1]),
                "seq": int(parts[2]),
                "ax": float(parts[3]),
                "ay": float(parts[4]),
                "az": float(parts[5]),
                "gx": float(parts[6]),
                "gy": float(parts[7]),
                "gz": float(parts[8]),
                "temp": float(parts[9]),
            }
        except Exception:
            return None
    elif len(parts) == 9:
        try:
            return {
                "timestamp": datetime.now().isoformat(),
                "tick_ms": int(parts[0]),
                "seq": int(parts[1]),
                "ax": float(parts[2]),
                "ay": float(parts[3]),
                "az": float(parts[4]),
                "gx": float(parts[5]),
                "gy": float(parts[6]),
                "gz": float(parts[7]),
                "temp": float(parts[8]),
            }
        except Exception:
            return None
    else:
        return None


# ==================================================
# ===== IMU Readers (optimized / robust) ===========
# ==================================================

class SerialIMULogger(QThread):
    """
    Serial IMU logger with robust header re-sync and tolerant CSV parsing,
    adapted from imu_stream_gui.py's SerialReader (ported to PyQt5 QThread).
    """
    log = pyqtSignal(str)
    imu_packet = pyqtSignal(dict)

    def __init__(self, port: str, baud: int, csv_path: str):
        super().__init__()
        self.port = port
        self.baud = baud
        self.csv_path = csv_path
        self._running = False
        self._ser = None
        self._decoder_errors = "ignore"
        self._have_header = False

    def run(self):
        # Open serial
        try:
            self._ser = serial.Serial(self.port, self.baud, timeout=1)
            try:
                self._ser.reset_input_buffer()
            except Exception:
                pass
            self.log.emit(f"IMU (Serial) connected: {self.port} @ {self.baud}")
        except Exception as e:
            self.log.emit(f"Serial open error: {e}")
            return

        # Open CSV
        try:
            f = open(self.csv_path, "w", newline="", encoding="utf-8")
            w = csv.writer(f)
            w.writerow(CSV_HEADER)
        except Exception as e:
            self.log.emit(f"CSV open error: {e}")
            f = None
            w = None

        self._running = True
        buf = ""
        skipped = 0
        try:
            while self._running:
                try:
                    chunk = self._ser.read(1024)
                    if not chunk:
                        continue
                    try:
                        s = chunk.decode("utf-8", errors=self._decoder_errors)
                    except Exception as e:
                        self.log.emit(f"Decode error: {e}")
                        continue

                    buf += s
                    while "\n" in buf:
                        line, buf = buf.split("\n", 1)
                        line = line.strip()
                        if not line:
                            continue

                        # Header re-sync (accept header anywhere in stream)
                        if not self._have_header:
                            if line.replace(" ", "") == ",".join(CSV_HEADER):
                                self._have_header = True
                                continue  # consume header
                            # else: fall through and try to parse anyway

                        # Tolerant CSV row parsing (pad if short)
                        try:
                            row_list = next(csv.reader([line]))
                        except Exception as e:
                            skipped += 1
                            if skipped % 50 == 0:
                                self.log.emit(f"Parse skip {skipped} (e={e}) last='{line[:80]}'")
                            continue

                        if len(row_list) < len(CSV_HEADER):
                            row_list += [""] * (len(CSV_HEADER) - len(row_list))

                        row = {k: v for k, v in zip(CSV_HEADER, row_list)}

                        # Normalize numeric strings -> floats/ints where possible
                        # and emit
                        parsed = {}
                        try:
                            parsed["timestamp"] = row.get("timestamp") or datetime.now().isoformat()
                            parsed["tick_ms"] = int(float(row.get("tick_ms") or 0))
                            parsed["seq"] = int(float(row.get("seq") or 0))
                            for k in ["ax","ay","az","gx","gy","gz","temp"]:
                                v = row.get(k, "")
                                parsed[k] = float(v) if v not in ("", None) else float("nan")
                        except Exception:
                            # Fallback to legacy parser for lines that aren't CSV-normal
                            parsed = parse_esp_csv(line)

                        if not parsed:
                            skipped += 1
                            if skipped % 50 == 0:
                                self.log.emit(f"Malformed skip {skipped} last='{line[:80]}'")
                            continue
                        skipped = 0

                        if w:
                            try:
                                w.writerow([
                                    parsed.get("timestamp",""),
                                    parsed.get("tick_ms",""),
                                    parsed.get("seq",""),
                                    parsed.get("ax",""),
                                    parsed.get("ay",""),
                                    parsed.get("az",""),
                                    parsed.get("gx",""),
                                    parsed.get("gy",""),
                                    parsed.get("gz",""),
                                    parsed.get("temp",""),
                                ])
                                f.flush()
                            except Exception:
                                pass

                        self.imu_packet.emit(parsed)

                except Exception as e:
                    self.log.emit(f"Serial read error: {e}")
                    time.sleep(0.05)
                    continue
        finally:
            try:
                if self._ser and self._ser.is_open:
                    self._ser.close()
            except Exception:
                pass
            if f:
                try: f.close()
                except Exception: pass
            self.log.emit("IMU Serial logger exiting")

    def stop(self):
        self._running = False


class IMULogger(QThread):
    """UDP IMU logger (unchanged behavior, small cleanups)."""
    log = pyqtSignal(str)
    imu_packet = pyqtSignal(dict)

    def __init__(self, host: str, port: int, csv_path: str):
        super().__init__()
        self.host = host
        self.port = port
        self.csv_path = csv_path
        self._running = False
        self._sock = None

    def run(self):
        try:
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._sock.settimeout(1.0)
            self._sock.bind((self.host, self.port))
        except Exception as e:
            self.log.emit(f"IMU socket bind error: {e}")
            return

        try:
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(CSV_HEADER)
                self._running = True
                self.log.emit(f"IMU (UDP) logging to {self.csv_path}")
                while self._running:
                    try:
                        data, _ = self._sock.recvfrom(4096)
                        raw = data.decode("utf-8", errors="replace").strip()
                        if not raw:
                            continue
                        parsed = parse_esp_csv(raw)
                        if not parsed:
                            continue
                        w.writerow([
                            parsed["timestamp"], parsed["tick_ms"], parsed["seq"],
                            parsed["ax"], parsed["ay"], parsed["az"],
                            parsed["gx"], parsed["gy"], parsed["gz"], parsed["temp"]
                        ])
                        f.flush()
                        self.imu_packet.emit(parsed)
                    except socket.timeout:
                        continue
                    except Exception as e:
                        self.log.emit(f"IMU recv error: {e}")
                        break
        finally:
            if self._sock:
                try: self._sock.close()
                except Exception: pass
            self.log.emit("IMU UDP logger exiting")

    def stop(self):
        self._running = False


class HttpIMULogger(QThread):
    """HTTP stream IMU logger (unchanged behavior, small cleanups)."""
    log = pyqtSignal(str)
    imu_packet = pyqtSignal(dict)

    def __init__(self, url: str, csv_path: str, timeout: float = 5.0):
        super().__init__()
        self.url = url.rstrip("/")
        self.csv_path = csv_path
        self.timeout = timeout
        self._running = False
        self._sess = None
        self._resp = None

    def run(self):
        try:
            self._sess = requests.Session()
            self._resp = self._sess.get(self.url, stream=True, timeout=self.timeout)
            self._resp.raise_for_status()
        except Exception as e:
            self.log.emit(f"HTTP IMU open error: {e}")
            return

        try:
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(CSV_HEADER)
                self._running = True
                self.log.emit(f"IMU (HTTP) logging from {self.url} -> {self.csv_path}")

                for raw in self._resp.iter_lines(decode_unicode=True, chunk_size=256):
                    if not self._running:
                        break
                    if not raw:
                        continue
                    parsed = parse_esp_csv(raw)
                    if not parsed:
                        continue
                    w.writerow([
                        parsed["timestamp"], parsed["tick_ms"], parsed["seq"],
                        parsed["ax"], parsed["ay"], parsed["az"],
                        parsed["gx"], parsed["gy"], parsed["gz"], parsed["temp"]
                    ])
                    f.flush()
                    self.imu_packet.emit(parsed)
        except Exception as e:
            self.log.emit(f"HTTP IMU read error: {e}")
        finally:
            try:
                if self._resp is not None:
                    self._resp.close()
            except Exception:
                pass
            try:
                if self._sess is not None:
                    self._sess.close()
            except Exception:
                pass
            self.log.emit("IMU HTTP logger exiting")

    def stop(self):
        self._running = False
        try:
            if self._resp is not None:
                self._resp.close()
        except Exception:
            pass


# =====================================================
# ===== camera.py (threads preserved) =================
# =====================================================


def _set_prop_safe(cap, prop, value, log_cb=None):
    try:
        if not cap.set(prop, value):
            if log_cb: log_cb(f"Warn: set({int(prop)})={value} not honored")
    except Exception as e:
        if log_cb: log_cb(f"Warn: set({int(prop)}) failed: {e}")

def _warmup(cap, attempts=20, sleep_s=0.02):
    import time
    ok = False
    for _ in range(attempts):
        ret, _ = cap.read()
        if ret:
            ok = True
            break
        time.sleep(sleep_s)
    return ok

def _open_with_backend(backend, idx, width, height, fps, force_mjpg, log_cb):
    cap = cv2.VideoCapture(idx, backend)
    if not cap.isOpened():
        return None
    # Optionally set MJPG
    if force_mjpg:
        try:
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            _set_prop_safe(cap, cv2.CAP_PROP_FOURCC, fourcc, log_cb)
        except Exception:
            pass
    # Size & FPS
    _set_prop_safe(cap, cv2.CAP_PROP_FRAME_WIDTH,  int(width), log_cb)
    _set_prop_safe(cap, cv2.CAP_PROP_FRAME_HEIGHT, int(height), log_cb)
    _set_prop_safe(cap, cv2.CAP_PROP_FPS,          int(fps), log_cb)

    if not _warmup(cap):
        # try without forcing size/fps if not working
        _set_prop_safe(cap, cv2.CAP_PROP_FRAME_WIDTH,  0, log_cb)
        _set_prop_safe(cap, cv2.CAP_PROP_FRAME_HEIGHT, 0, log_cb)
        _set_prop_safe(cap, cv2.CAP_PROP_FPS,          0, log_cb)
        if not _warmup(cap):
            cap.release()
            return None
    return cap

def _open_cap_multi(device_index: int, width: int, height: int, fps: int, log_cb=None):
    # Try a sequence of (backend, force_mjpg) candidates on Windows
    candidates = [
        (cv2.CAP_DSHOW, True,  "DSHOW+MJPG"),
        (cv2.CAP_DSHOW, False, "DSHOW"),
        (cv2.CAP_MSMF,  True,  "MSMF+MJPG"),
        (cv2.CAP_MSMF,  False, "MSMF"),
        (cv2.CAP_ANY,   False, "ANY"),
    ]
    for backend, force_mjpg, label in candidates:
        cap = _open_with_backend(backend, device_index, width, height, fps, force_mjpg, log_cb)
        if cap is not None:
            if log_cb: log_cb(f"Camera {device_index} opened via {label}")
            return cap
        else:
            if log_cb: log_cb(f"Camera {device_index} open failed ({label}).")

    # As a last resort, scan indices 0..10 to suggest a working index
    if log_cb: log_cb("Scanning indices 0..10 to find any openable camera...")
    for i in range(0, 11):
        cap = _open_with_backend(cv2.CAP_MSMF, i, width, height, fps, False, log_cb)
        if cap is None:
            cap = _open_with_backend(cv2.CAP_DSHOW, i, width, height, fps, False, log_cb)
        if cap is None:
            cap = _open_with_backend(cv2.CAP_ANY, i, width, height, fps, False, log_cb)
        if cap is not None:
            if log_cb: log_cb(f"Suggestion: try Device {i} (opened successfully).")
            cap.release()
            break
    return None



class CameraThread(QThread):
    _use_ffmpeg = True
    _vw = None
    frame_ready = pyqtSignal(QImage)
    log = pyqtSignal(str)

    def __init__(self, device_index: int, out_path: str, out_csv: str, host_start_ms: int,
                 width: int = 1280, height: int = 720, fps: int = 20):
        super().__init__()
        self.device_index = device_index
        self.out_path = out_path
        self.out_csv = out_csv
        self.host_start_ms = host_start_ms
        self.width = width
        self.height = height
        self.fps = fps

        self._running = False
        self.proc = None
        self.frame_index = 0

    def run(self):
        cap = _open_cap_multi(self.device_index, self.width, self.height, self.fps, self._emit_log)
        if cap is None:
            self._emit_log(f"Camera {self.device_index} failed to open (check device index & permissions)")
            return

                # Prefer ffmpeg if available
        self._use_ffmpeg = shutil.which('ffmpeg') is not None
        if self._use_ffmpeg:
            cmd = ffmpeg_cmd_for_windows_output(self.out_path, self.width, self.height, self.fps)
            try:
                self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
            except Exception as e:
                self._emit_log(f"FFmpeg not usable for cam{self.device_index} -> falling back to OpenCV VideoWriter (reason: {e})")
                self._use_ffmpeg = False
        if not self._use_ffmpeg:
            # Fallback: OpenCV VideoWriter (mp4v)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self._vw = cv2.VideoWriter(self.out_path, fourcc, float(self.fps), (self.width, self.height))
            if not self._vw.isOpened():
                self._emit_log(f"Failed to open VideoWriter for cam{self.device_index} -> stopping")
                cap.release()
                return

        try:
            csv_f = open(self.out_csv, "w", newline="", encoding="utf-8")
            csv_w = csv.writer(csv_f)
            csv_w.writerow(["frame_index", "epoch_ms"])
        except Exception as e:
            self._emit_log(f"Failed to open frame CSV for cam{self.device_index}: {e}")
            csv_f = None
            csv_w = None

        self._running = True
        self._emit_log(f"Camera {self.device_index} recording -> {self.out_path} ; frames -> {self.out_csv}")

        min_frame_interval = 1.0 / max(1, self.fps)
        last_frame_time = time.time()
        consecutive_fail = 0

        try:
            while self._running:
                ret, frame = cap.read()
                if not ret or frame is None:
                    consecutive_fail += 1
                    if consecutive_fail >= 30:
                        self._emit_log(f"Camera {self.device_index} read failed, reconnecting...")
                        try: cap.release()
                        except Exception: pass
                        cap = _open_cap_multi(self.device_index, self.width, self.height, self.fps, self._emit_log)
                        consecutive_fail = 0
                        if cap is None:
                            self._emit_log(f"Camera {self.device_index} cannot reopen; stopping.")
                            break
                    time.sleep(0.01)
                    continue
                consecutive_fail = 0

                if frame.shape[1] != self.width or frame.shape[0] != self.height:
                    frame = cv2.resize(frame, (self.width, self.height))

                epoch_ms = now_epoch_ms()
                ms_since_start = epoch_ms - self.host_start_ms
                timecode_str = fmt_hms_ms(ms_since_start)
                font = cv2.FONT_HERSHEY_SIMPLEX
                scale, thickness = 0.7, 2
                (tw, th), _ = cv2.getTextSize(timecode_str, font, scale, thickness)
                x, y = 10, self.height - 10
                cv2.rectangle(frame, (x - 6, y - th - 6), (x + tw + 6, y + 6), (0, 0, 0), cv2.FILLED)
                cv2.putText(frame, timecode_str, (x, y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)

                if csv_w:
                    try:
                        csv_w.writerow([self.frame_index, epoch_ms])
                        csv_f.flush()
                    except Exception:
                        pass

                if self._use_ffmpeg and self.proc and self.proc.stdin:
                    try:
                        self.proc.stdin.write(frame.tobytes())
                    except Exception as e:
                        self._emit_log(f"FFmpeg stdin write error cam{self.device_index}: {e}; switching to VideoWriter fallback")
                        # Try switching to fallback at runtime
                        self._use_ffmpeg = False
                        try:
                            if self.proc and self.proc.stdin:
                                self.proc.stdin.close()
                                self.proc.wait(timeout=2)
                        except Exception:
                            pass
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        self._vw = cv2.VideoWriter(self.out_path, fourcc, float(self.fps), (self.width, self.height))
                        if not self._vw.isOpened():
                            self._emit_log(f"Fallback VideoWriter failed for cam{self.device_index}; stopping thread")
                            break
                        # write current frame via fallback
                        try:
                            self._vw.write(frame)
                        except Exception as e2:
                            self._emit_log(f"VideoWriter write error cam{self.device_index}: {e2}")
                            break
                else:
                    # Fallback path
                    try:
                        self._vw.write(frame)
                    except Exception as e:
                        self._emit_log(f"VideoWriter write error cam{self.device_index}: {e}")
                        break

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                qimg = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.shape[1] * 3, QImage.Format_RGB888).copy()
                self.frame_ready.emit(qimg)

                self.frame_index += 1
                elapsed = time.time() - last_frame_time
                to_wait = min_frame_interval - elapsed
                if to_wait > 0:
                    time.sleep(to_wait)
                last_frame_time = time.time()
        finally:
            if csv_f:
                try: csv_f.close()
                except Exception: pass
            if self._vw is not None:
                try:
                    self._vw.release()
                except Exception:
                    pass
            if self.proc and self.proc.stdin:
                try:
                    self.proc.stdin.close()
                    self.proc.wait(timeout=5)
                except Exception:
                    self.proc.kill()
            try: cap.release()
            except Exception: pass
            self._emit_log(f"Camera {self.device_index} stopped (frames: {self.frame_index})")

    def stop(self):
        self._running = False

    def _emit_log(self, msg: str):
        try:
            self.log.emit(msg)
        except Exception:
            pass


class IdlePreviewThread(QThread):
    frame_ready = pyqtSignal(QImage)
    log = pyqtSignal(str)

    def __init__(self, device_index: int, width: int = 1280, height: int = 720, fps: int = 15):
        super().__init__()
        self.device_index = device_index
        self.width = width
        self.height = height
        self.fps = fps

        self._running = False
        self.frame_index = 0

    def run(self):
        cap = _open_cap_multi(self.device_index, self.width, self.height, self.fps, self._emit_log)
        if cap is None:
            self._emit_log(f"[IDLE] Camera {self.device_index} failed to open")
            return

        self._running = True
        min_interval = 1.0 / max(1, self.fps)
        last_t = time.time()

        try:
            while self._running:
                ret, frame = cap.read()
                if not ret or frame is None:
                    time.sleep(0.01)
                    continue

                if frame.shape[1] != self.width or frame.shape[0] != self.height:
                    frame = cv2.resize(frame, (self.width, self.height))

                cv2.putText(frame, "IDLE PREVIEW", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(frame, "IDLE PREVIEW", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                qimg = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.shape[1] * 3, QImage.Format_RGB888).copy()
                self.frame_ready.emit(qimg)

                self.frame_index += 1
                dt = time.time() - last_t
                sleep_for = min_interval - dt
                if sleep_for > 0:
                    time.sleep(sleep_for)
                last_t = time.time()
        finally:
            try: cap.release()
            except Exception: pass
            self._emit_log(f"[IDLE] Camera {self.device_index} stopped at frame {self.frame_index}")

    def stop(self):
        self._running = False

    def _emit_log(self, msg: str):
        try:
            self.log.emit(msg)
        except Exception:
            pass


# =====================================================
# ===== main.py (UI & wiring preserved) ===============
# =====================================================

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FALL DETECTION RECORDER (IMU optimized)")
        self.setMinimumSize(1300, 900)

        central = QWidget()
        root = QVBoxLayout()

        cam_row = QHBoxLayout()
        self.cam_inputs, self.preview_labels = [], []
        for i in range(3):
            col = QVBoxLayout()
            col.addWidget(QLabel(f"Camera {i}:"))
            cb = QComboBox(); cb.setMaximumWidth(360); col.addWidget(cb)
            self.cam_inputs.append(cb)
            preview = QLabel(); preview.setFixedSize(420, 240)
            preview.setStyleSheet("background:#111; border:1px solid #444")
            preview.setAlignment(Qt.AlignCenter)
            col.addWidget(preview)
            self.preview_labels.append(preview)
            cam_row.addLayout(col)
            cb.currentIndexChanged.connect(partial(self.restart_idle_preview, i))
        root.addLayout(cam_row)

        refresh_row = QHBoxLayout()
        self.refresh_btn = QPushButton("Refresh Cameras")
        refresh_row.addWidget(self.refresh_btn)
        root.addLayout(refresh_row)

        imu_mode_row = QHBoxLayout()
        imu_mode_row.addWidget(QLabel("IMU Source:"))
        self.imu_mode = QComboBox()
        self.imu_mode.addItems(["UDP", "Serial", "HTTP"])
        imu_mode_row.addWidget(self.imu_mode)
        root.addLayout(imu_mode_row)

        udp_row = QHBoxLayout()
        udp_row.addWidget(QLabel("UDP host:"))
        self.udp_host_edit = QLineEdit("0.0.0.0"); udp_row.addWidget(self.udp_host_edit)
        udp_row.addWidget(QLabel("port:"))
        self.udp_port_edit = QLineEdit("5005"); self.udp_port_edit.setMaximumWidth(80); udp_row.addWidget(self.udp_port_edit)
        root.addLayout(udp_row)

        serial_row = QHBoxLayout()
        serial_row.addWidget(QLabel("Serial port:"))
        self.serial_port_edit = QLineEdit("COM3"); serial_row.addWidget(self.serial_port_edit)
        serial_row.addWidget(QLabel("baud:"))
        self.serial_baud_edit = QLineEdit("115200"); self.serial_baud_edit.setMaximumWidth(100); serial_row.addWidget(self.serial_baud_edit)
        root.addLayout(serial_row)

        esp_row = QHBoxLayout()
        esp_row.addWidget(QLabel("ESP32 IP:"))
        self.esp_ip_edit = QLineEdit("http://192.168.4.1"); esp_row.addWidget(self.esp_ip_edit)
        self.btn_esp_start = QPushButton("Start Stream")
        self.btn_esp_stop = QPushButton("Stop Stream")
        self.btn_esp_recal = QPushButton("Recalibrate")
        self.esp_delay_edit = QLineEdit("20"); self.esp_delay_edit.setMaximumWidth(60)
        self.btn_esp_delay = QPushButton("Set Delay")
        esp_row.addWidget(self.btn_esp_start); esp_row.addWidget(self.btn_esp_stop)
        esp_row.addWidget(self.btn_esp_recal); esp_row.addWidget(QLabel("Delay ms:"))
        esp_row.addWidget(self.esp_delay_edit); esp_row.addWidget(self.btn_esp_delay)
        root.addLayout(esp_row)

        imu_panel = QGridLayout()
        self.imu_labels = {}
        fields = ["seq","tick_ms","ax","ay","az","gx","gy","gz","temp"]
        imu_panel.addWidget(QLabel("IMU (Realtime):"), 0, 0, 1, 2)
        for row, key in enumerate(fields, start=1):
            key_lab = QLabel(key + ":")
            val_lab = QLabel("--")
            self.imu_labels[key] = val_lab
            imu_panel.addWidget(key_lab, row, 0)
            imu_panel.addWidget(val_lab, row, 1)
        root.addLayout(imu_panel)

        dataset_row = QHBoxLayout()
        dataset_row.addWidget(QLabel("Dataset Folder Name:"))
        self.dataset_name_edit = QLineEdit()
        self.dataset_name_edit.setPlaceholderText("Leave empty for auto-generated name")
        dataset_row.addWidget(self.dataset_name_edit)
        root.addLayout(dataset_row)

        btn_row = QHBoxLayout()
        self.start_btn = QPushButton("Start Recording")
        self.stop_btn = QPushButton("Stop Recording")
        self.open_btn = QPushButton("Open Session Folder")
        btn_row.addWidget(self.start_btn); btn_row.addWidget(self.stop_btn); btn_row.addWidget(self.open_btn)
        root.addLayout(btn_row)

        self.log = QTextEdit(); self.log.setReadOnly(True); root.addWidget(self.log)

        central.setLayout(root)
        self.setCentralWidget(central)

        self.session_folder = None
        self.host_start_ms = None
        self.imu_thread = None
        self.cam_threads = []
        self.idle_threads = []

        self.refresh_btn.clicked.connect(self.refresh_cameras)
        self.start_btn.clicked.connect(self.start_session)
        self.stop_btn.clicked.connect(self.stop_session)
        self.open_btn.clicked.connect(self.open_session_folder)
        self.btn_esp_start.clicked.connect(self.esp_start_stream)
        self.btn_esp_stop.clicked.connect(self.esp_stop_stream)
        self.btn_esp_recal.clicked.connect(self.esp_recalibrate)
        self.btn_esp_delay.clicked.connect(self.esp_set_delay)

        self.stop_btn.setEnabled(False)
        self.refresh_cameras()

    def esp_base(self) -> str:
        return self.esp_ip_edit.text().strip().rstrip("/")

    def esp_start_stream(self):
        base = self.esp_base()
        try:
            r = requests.get(f"{base}/stream/start", timeout=3)
            txt = (r.text or "").strip()
            if r.status_code >= 400 or not txt or "ok" not in txt.lower():
                r = requests.get(f"{base}/stream/toggle", timeout=3)
                txt = (r.text or "").strip()
            self.log_msg(f"ESP start: {txt}")
        except Exception as e:
            self.log_msg(f"ESP start error: {e}")

    def esp_stop_stream(self):
        base = self.esp_base()
        try:
            r = requests.get(f"{base}/stream/stop", timeout=3)
            txt = (r.text or "").strip()
            if r.status_code >= 400 or not txt or "ok" not in txt.lower():
                r = requests.get(f"{base}/stream/toggle", timeout=3)
                txt = (r.text or "").strip()
            self.log_msg(f"ESP stop: {txt}")
        except Exception as e:
            self.log_msg(f"ESP stop error: {e}")

    def esp_recalibrate(self):
        try:
            r = requests.get(f"{self.esp_base()}/imu/recalibrate", timeout=3)
            self.log_msg(f"ESP recalibrate: {r.text}")
        except Exception as e:
            self.log_msg(f"ESP recalibrate error: {e}")

    def esp_set_delay(self):
        try:
            val = int(self.esp_delay_edit.text())
            r = requests.get(f"{self.esp_base()}/imu/delay?value={val}", timeout=3)
            self.log_msg(f"ESP delay set: {r.text}")
        except Exception as e:
            self.log_msg(f"ESP delay error: {e}")

    def log_msg(self, s: str):
        ts = datetime.now().strftime("%H:%M:%S")
        self.log.append(f"[{ts}] {s}")

    def refresh_cameras(self):
        graph = FilterGraph()
        try:
            device_names = graph.get_input_devices()
        except Exception:
            device_names = []

        def good(name: str) -> bool:
            n = (name or "").lower()
            if "virtual" in n or "obs" in n or "ndi" in n:
                return False
            tokens = ("uvc", "usb", "general")
            return any(t in n for t in tokens)

        ordered, others = [], []
        for idx, name in enumerate(device_names):
            (ordered if good(name) else others).append((idx, name))
        listing = ordered + others

        for i, cb in enumerate(self.cam_inputs):
            cb.blockSignals(True)
            cb.clear()
            if listing:
                for idx, name in listing:
                    cb.addItem(f"{name} (Device {idx})", idx)
            else:
                for idx in range(10):
                    cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
                    if cap.isOpened():
                        cb.addItem(f"Device {idx}", idx)
                        cap.release()
            if cb.count() == 0:
                cb.addItem("No camera found", -1)
            cb.setCurrentIndex(0)
            cb.blockSignals(False)

        self.log_msg("Camera list refreshed")
        self.restart_all_idle_previews()

    def start_session(self):
        self.stop_idle_previews()

        try:
            custom = getattr(self, "dataset_name_edit", None)
            custom_name = custom.text().strip() if custom else ""
            if custom_name:
                folder = make_session_folder_with_name(custom_name)
            else:
                folder = make_session_folder()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to create session folder: {e}")
            return
        self.session_folder = folder

        self.videos_dir = os.path.join(folder, 'videos')
        self.camcsv_dir = os.path.join(folder, 'camera_csv')
        self.imu_dir = os.path.join(folder, 'imu')
        for d in (self.videos_dir, self.camcsv_dir, self.imu_dir):
            os.makedirs(d, exist_ok=True)

        self.host_start_ms = now_epoch_ms()
        with open(os.path.join(self.imu_dir, "host_start_ms.txt"), "w") as f:
            f.write(str(self.host_start_ms))
        self.log_msg(f"Session folder: {folder} (host_start_ms: {self.host_start_ms})")
        if custom_name:
            self.log_msg(f"Dataset name: {sanitize_folder_component(custom_name)}")

        imu_csv = os.path.join(self.imu_dir, "imu.csv")
        mode = self.imu_mode.currentText()
        if mode == "UDP":
            try:
                port = int(self.udp_port_edit.text().strip())
            except ValueError:
                QMessageBox.critical(self, "Error", "Invalid UDP port")
                return
            self.imu_thread = IMULogger(self.udp_host_edit.text().strip(), port, imu_csv)

        elif mode == "Serial":
            port_name = self.serial_port_edit.text().strip()
            try:
                baud = int(self.serial_baud_edit.text().strip())
            except ValueError:
                QMessageBox.critical(self, "Error", "Invalid baudrate")
                return
            self.imu_thread = SerialIMULogger(port_name, baud, imu_csv)

        else:
            stream_url = f"{self.esp_base()}/stream"
            self.imu_thread = HttpIMULogger(stream_url, imu_csv)

        self.imu_thread.log.connect(self.log_msg)
        self.imu_thread.imu_packet.connect(self.update_imu_preview)
        self.imu_thread.start()

        self.cam_threads = []
        for i, cb in enumerate(self.cam_inputs):
            dev = cb.currentData()
            if dev is None or dev < 0:
                self.log_msg(f"Skipping Camera {i}, no valid device")
                continue
            out_video = os.path.join(self.videos_dir, f"cam{i}.mp4")
            out_csv = os.path.join(self.camcsv_dir, f"cam{i}.csv")
            t = CameraThread(device_index=dev, out_path=out_video, out_csv=out_csv, host_start_ms=self.host_start_ms)
            t.log.connect(self.log_msg)
            t.frame_ready.connect(partial(self.update_preview, i))
            t.start()
            self.cam_threads.append(t)

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.log_msg("Recording started (IMU + cameras)")

    def update_preview(self, idx: int, qimg: QImage):
        if idx < 0 or idx >= len(self.preview_labels):
            return
        lbl = self.preview_labels[idx]
        pix = QPixmap.fromImage(qimg).scaled(lbl.width(), lbl.height(), Qt.KeepAspectRatio)
        lbl.setPixmap(pix)

    def update_imu_preview(self, parsed: dict):
        if not parsed:
            return
        for key in ["seq","tick_ms","ax","ay","az","gx","gy","gz","temp"]:
            lab = self.imu_labels.get(key)
            if not lab:
                continue
            val = parsed.get(key)
            if key in ("seq","tick_ms"):
                lab.setText(str(val))
            else:
                try:
                    lab.setText(f"{float(val):.3f}")
                except Exception:
                    lab.setText("--")

    def stop_session(self):
        self.log_msg("Stopping session...")
        for t in self.cam_threads:
            try:
                t.stop()
                t.wait(5000)
            except Exception:
                pass
        self.cam_threads = []

        if self.imu_thread:
            self.imu_thread.stop()
            self.imu_thread.wait(2000)
            self.imu_thread = None

        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.log_msg("Session stopped. Files saved in session folder.")

        self.restart_all_idle_previews()

    def open_session_folder(self):
        if not self.session_folder:
            QMessageBox.information(self, "No session", "No session yet")
            return
        folder = os.path.abspath(self.session_folder)
        if sys.platform.startswith("win"):
            subprocess.Popen(["explorer", folder])
        else:
            try:
                subprocess.Popen(["xdg-open", folder])
            except Exception:
                pass

    def start_idle_previews(self):
        self.stop_idle_previews()
        self.idle_threads = []
        for i, cb in enumerate(self.cam_inputs):
            dev = cb.currentData()
            if dev is None or dev < 0:
                continue
            t = IdlePreviewThread(device_index=dev, width=1280, height=720, fps=15)
            t.log.connect(self.log_msg)
            t.frame_ready.connect(partial(self.update_preview, i))
            t.start()
            self.idle_threads.append((i, t))
        self.log_msg("Idle previews started")

    def stop_idle_previews(self):
        if not self.idle_threads:
            return
        for i, t in self.idle_threads:
            try:
                t.stop()
                t.wait(2000)
            except Exception:
                pass
        self.idle_threads = []
        self.log_msg("Idle previews stopped")

    def restart_idle_preview(self, idx: int, *_args):
        keep = []
        for i, t in self.idle_threads:
            if i == idx:
                try:
                    t.stop(); t.wait(1500)
                except Exception:
                    pass
            else:
                keep.append((i, t))
        self.idle_threads = keep

        cb = self.cam_inputs[idx]
        dev = cb.currentData()
        if dev is None or dev < 0:
            self.preview_labels[idx].clear()
            return
        t = IdlePreviewThread(device_index=dev, width=1280, height=720, fps=15)
        t.log.connect(self.log_msg)
        t.frame_ready.connect(partial(self.update_preview, idx))
        t.start()
        self.idle_threads.append((idx, t))
        self.log_msg(f"Idle preview restarted for Camera {idx} (Device {dev})")

    def restart_all_idle_previews(self):
        self.stop_idle_previews()
        self.start_idle_previews()


def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
