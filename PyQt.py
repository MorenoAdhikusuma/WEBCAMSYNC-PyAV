import sys
import os
import time
import csv
import socket
import subprocess
from datetime import datetime
from pathlib import Path
from functools import partial

import requests
import cv2
# numpy is imported by opencv; keep available
import numpy as np
from pygrabber.dshow_graph import FilterGraph

from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QTextEdit, QMessageBox, QComboBox, QLineEdit
)

# ----------------------------- Helpers -----------------------------
def now_epoch_ms() -> int:
    return int(time.time() * 1000)

def make_session_folder(base_dir: str = "./sessions") -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = Path(base_dir) / f"session_{ts}"
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
    Parse ESP IMU per-line CSV into dict matching:
    timestamp,tick_ms,seq,ax,ay,az,gx,gy,gz,temp
    Expected line: tick_ms,seq,ax,ay,az,gx,gy,gz,temp
    """
    parts = line.strip().split(",")
    if len(parts) < 9:
        return None
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

# ----------------------------- IMU UDP Logger -----------------------------
class IMULogger(QThread):
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
                writer = csv.writer(f)
                writer.writerow(["timestamp","tick_ms","seq","ax","ay","az","gx","gy","gz","temp"])
                self._running = True
                self.log.emit(f"IMU (UDP) logging to {self.csv_path}")
                while self._running:
                    try:
                        data, _ = self._sock.recvfrom(4096)
                        raw = data.decode("utf-8", errors="replace")
                        parsed = parse_esp_csv(raw)
                        if not parsed:
                            continue
                        writer.writerow([
                            parsed["timestamp"], parsed["tick_ms"], parsed["seq"],
                            parsed["ax"], parsed["ay"], parsed["az"],
                            parsed["gx"], parsed["gy"], parsed["gz"], parsed["temp"]
                        ])
                        f.flush()
                        self.log.emit(
                            f"IMU seq {parsed['seq']} "
                            f"ax={parsed['ax']:.3f} ay={parsed['ay']:.3f} az={parsed['az']:.3f}"
                        )
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

# ----------------------------- IMU Serial Logger -----------------------------
import serial

class SerialIMULogger(QThread):
    log = pyqtSignal(str)
    imu_packet = pyqtSignal(dict)

    def __init__(self, port: str, baud: int, csv_path: str):
        super().__init__()
        self.port = port
        self.baud = baud
        self.csv_path = csv_path
        self._running = False
        self.ser = None

    def run(self):
        try:
            self.ser = serial.Serial(self.port, self.baud, timeout=1)
        except Exception as e:
            self.log.emit(f"Serial open error: {e}")
            return

        try:
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp","tick_ms","seq","ax","ay","az","gx","gy","gz","temp"])
                self._running = True
                self.log.emit(f"IMU (Serial) logging to {self.csv_path}")
                while self._running:
                    try:
                        line = self.ser.readline().decode("utf-8", errors="replace")
                        if not line:
                            continue
                        parsed = parse_esp_csv(line)
                        if not parsed:
                            continue
                        writer.writerow([
                            parsed["timestamp"], parsed["tick_ms"], parsed["seq"],
                            parsed["ax"], parsed["ay"], parsed["az"],
                            parsed["gx"], parsed["gy"], parsed["gz"], parsed["temp"]
                        ])
                        f.flush()
                        self.log.emit(
                            f"IMU seq {parsed['seq']} "
                            f"ax={parsed['ax']:.3f} ay={parsed['ay']:.3f} az={parsed['az']:.3f}"
                        )
                        self.imu_packet.emit(parsed)
                    except Exception:
                        continue
        finally:
            if self.ser:
                try: self.ser.close()
                except Exception: pass
            self.log.emit("IMU Serial logger exiting")

    def stop(self):
        self._running = False

# ----------------------------- Camera thread -----------------------------
class CameraThread(QThread):
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
        cap = cv2.VideoCapture(self.device_index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            self.log.emit(f"Camera {self.device_index} failed to open")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, self.fps)

        cmd = ffmpeg_cmd_for_windows_output(self.out_path, self.width, self.height, self.fps)
        try:
            self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        except Exception as e:
            self.log.emit(f"Failed to start ffmpeg for cam{self.device_index}: {e}")
            cap.release()
            return

        try:
            csv_f = open(self.out_csv, "w", newline="", encoding="utf-8")
            csv_w = csv.writer(csv_f)
            csv_w.writerow(["frame_index", "epoch_ms"])
        except Exception as e:
            self.log.emit(f"Failed to open frame CSV for cam{self.device_index}: {e}")
            csv_f = None
            csv_w = None

        self._running = True
        self.log.emit(f"Camera {self.device_index} recording -> {self.out_path} ; frames -> {self.out_csv}")

        min_frame_interval = 1.0 / self.fps
        last_frame_time = time.time()

        try:
            while self._running:
                ret, frame = cap.read()
                if not ret:
                    self.log.emit(f"Camera {self.device_index} read failed")
                    time.sleep(0.01)
                    continue

                if frame.shape[1] != self.width or frame.shape[0] != self.height:
                    frame = cv2.resize(frame, (self.width, self.height))

                epoch_ms = now_epoch_ms()
                ms_since_start = epoch_ms - self.host_start_ms
                timecode_str = fmt_hms_ms(ms_since_start)

                font = cv2.FONT_HERSHEY_SIMPLEX
                scale = 0.7
                thickness = 2
                (tw, th), _ = cv2.getTextSize(timecode_str, font, scale, thickness)
                x, y = 10, self.height - 10
                cv2.rectangle(frame, (x - 6, y - th - 6), (x + tw + 6, y + 6), (0, 0, 0), cv2.FILLED)
                cv2.putText(frame, timecode_str, (x, y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)

                if csv_w:
                    csv_w.writerow([self.frame_index, epoch_ms])
                    csv_f.flush()

                try:
                    self.proc.stdin.write(frame.tobytes())
                except Exception as e:
                    self.log.emit(f"FFmpeg stdin write error cam{self.device_index}: {e}")
                    break

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                qimg = QImage(
                    rgb.data, rgb.shape[1], rgb.shape[0], rgb.shape[1] * 3, QImage.Format_RGB888
                ).copy()
                self.frame_ready.emit(qimg)

                self.frame_index += 1
                elapsed = time.time() - last_frame_time
                to_wait = min_frame_interval - elapsed
                if to_wait > 0:
                    time.sleep(to_wait)
                last_frame_time = time.time()
        finally:
            if csv_f:
                csv_f.close()
            if self.proc and self.proc.stdin:
                try:
                    self.proc.stdin.close()
                    self.proc.wait(timeout=5)
                except Exception:
                    self.proc.kill()
            cap.release()
            self.log.emit(f"Camera {self.device_index} stopped (frames: {self.frame_index})")

    def stop(self):
        self._running = False

# ----------------------------- MainWindow -----------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FALL DETECTION RECORDER")
        self.setMinimumSize(1300, 900)

        central = QWidget()
        root = QVBoxLayout()

        # Camera config + previews
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
        root.addLayout(cam_row)

        # Refresh button
        refresh_row = QHBoxLayout()
        self.refresh_btn = QPushButton("Refresh Cameras")
        refresh_row.addWidget(self.refresh_btn)
        root.addLayout(refresh_row)

        # IMU source selection
        imu_mode_row = QHBoxLayout()
        imu_mode_row.addWidget(QLabel("IMU Source:"))
        self.imu_mode = QComboBox()
        self.imu_mode.addItems(["UDP", "Serial"])
        imu_mode_row.addWidget(self.imu_mode)
        root.addLayout(imu_mode_row)

        # UDP config
        udp_row = QHBoxLayout()
        udp_row.addWidget(QLabel("UDP host:"))
        self.udp_host_edit = QLineEdit("0.0.0.0"); udp_row.addWidget(self.udp_host_edit)
        udp_row.addWidget(QLabel("port:"))
        self.udp_port_edit = QLineEdit("5005"); self.udp_port_edit.setMaximumWidth(80); udp_row.addWidget(self.udp_port_edit)
        root.addLayout(udp_row)

        # Serial config
        serial_row = QHBoxLayout()
        serial_row.addWidget(QLabel("Serial port:"))
        self.serial_port_edit = QLineEdit("COM3"); serial_row.addWidget(self.serial_port_edit)
        serial_row.addWidget(QLabel("baud:"))
        self.serial_baud_edit = QLineEdit("115200"); self.serial_baud_edit.setMaximumWidth(100); serial_row.addWidget(self.serial_baud_edit)
        root.addLayout(serial_row)

        # ESP32 HTTP controls
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

        # Real-time IMU panel
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

        # Control buttons
        btn_row = QHBoxLayout()
        self.start_btn = QPushButton("Start Recording")
        self.stop_btn = QPushButton("Stop Recording")
        self.open_btn = QPushButton("Open Session Folder")
        btn_row.addWidget(self.start_btn); btn_row.addWidget(self.stop_btn); btn_row.addWidget(self.open_btn)
        root.addLayout(btn_row)

        # Log
        self.log = QTextEdit(); self.log.setReadOnly(True); root.addWidget(self.log)

        central.setLayout(root)
        self.setCentralWidget(central)

        # State
        self.session_folder = None
        self.host_start_ms = None
        self.imu_thread = None
        self.cam_threads = []

        # Connections
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

    # --------- ESP HTTP controls ---------
    def esp_base(self) -> str:
        return self.esp_ip_edit.text().strip().rstrip("/")

    def esp_start_stream(self):
        try:
            r = requests.get(f"{self.esp_base()}/stream/start", timeout=3)
            self.log_msg(f"ESP start: {r.text}")
        except Exception as e:
            self.log_msg(f"ESP start error: {e}")

    def esp_stop_stream(self):
        try:
            r = requests.get(f"{self.esp_base()}/stream/stop", timeout=3)
            self.log_msg(f"ESP stop: {r.text}")
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

    # --------- Utility ---------
    def log_msg(self, s: str):
        ts = datetime.now().strftime("%H:%M:%S")
        self.log.append(f"[{ts}] {s}")

    def refresh_cameras(self):
        graph = FilterGraph()
        try:
            device_names = graph.get_input_devices()
        except Exception:
            device_names = []

        for i, cb in enumerate(self.cam_inputs):
            cb.blockSignals(True)
            cb.clear()
            if device_names:
                for idx, name in enumerate(device_names):
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

    def start_session(self):
        # Create session folder/get start time
        try:
            folder = make_session_folder()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to create session folder: {e}")
            return
        self.session_folder = folder
        self.host_start_ms = now_epoch_ms()
        with open(os.path.join(folder, "host_start_ms.txt"), "w") as f:
            f.write(str(self.host_start_ms))
        self.log_msg(f"Session folder: {folder} (host_start_ms: {self.host_start_ms})")

        # Start IMU (UDP or Serial)
        imu_csv = os.path.join(folder, "imu.csv")
        mode = self.imu_mode.currentText()
        if mode == "UDP":
            try:
                port = int(self.udp_port_edit.text().strip())
            except ValueError:
                QMessageBox.critical(self, "Error", "Invalid UDP port")
                return
            self.imu_thread = IMULogger(self.udp_host_edit.text().strip(), port, imu_csv)
        else:
            port_name = self.serial_port_edit.text().strip()
            try:
                baud = int(self.serial_baud_edit.text().strip())
            except ValueError:
                QMessageBox.critical(self, "Error", "Invalid baudrate")
                return
            self.imu_thread = SerialIMULogger(port_name, baud, imu_csv)

        self.imu_thread.log.connect(self.log_msg)
        self.imu_thread.imu_packet.connect(self.update_imu_preview)
        self.imu_thread.start()

        # Start cameras
        self.cam_threads = []
        for i, cb in enumerate(self.cam_inputs):
            dev = cb.currentData()
            if dev is None or dev < 0:
                self.log_msg(f"Skipping Camera {i}, no valid device")
                continue
            out_video = os.path.join(folder, f"cam{i}.mp4")
            out_csv = os.path.join(folder, f"cam{i}.csv")
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
        # Update realtime labels
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

# ----------------------------- entrypoint -----------------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())
