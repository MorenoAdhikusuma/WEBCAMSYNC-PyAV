import cv2
import threading
import time
import av
import numpy as np
import os
import queue
from dataclasses import dataclass
#biar ngga terlalu wordy pake dataclass aja
from typing import List, Optional, Tuple
from enum import Enum


class CameraState(Enum):
    IDLE = "idle"
    RECORDING = "recording"
    ERROR = "error"


@dataclass
class CameraConfig:
    index: int
    width: Optional[int] = None
    height: Optional[int] = None
    fps: float = 30.0
    preview_scale: float = 0.5
    output_dir: str = "output"


@dataclass
class RecordingConfig:
    codec: str = 'libx264'
    preset: str = 'ultrafast'
    crf: str = '23'
    pixel_format: str = 'yuv420p'
    queue_size: int = 30


class FrameEncoder:
    """Handles frame encoding in a separate thread"""
    
    def __init__(self, output_path: str, width: int, height: int, fps: float, config: RecordingConfig):
        self.output_path = output_path
        self.width = width
        self.height = height
        self.fps = fps
        self.config = config
        
        self.frame_queue = queue.Queue(maxsize=config.queue_size)
        self.encoding_thread = None
        self.output = None
        self.stream = None
        self.start_time = None
        self.frame_count = 0
        self.is_encoding = False
    
    def start(self):
        """Start the encoding process"""
        self._setup_output()
        self.start_time = time.time()
        self.frame_count = 0
        self.is_encoding = True
        
        self.encoding_thread = threading.Thread(target=self._encoding_worker)
        self.encoding_thread.daemon = True
        self.encoding_thread.start()
    
    def _setup_output(self):
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        
        self.output = av.open(self.output_path, mode='w')
        self.stream = self.output.add_stream(self.config.codec, rate=self.fps)
        
        self.stream.metadata['timecode'] = '00:00:00:00'
        self.stream.width = self.width
        self.stream.height = self.height
        self.stream.pix_fmt = self.config.pixel_format
        self.stream.options = {
            'preset': self.config.preset,
            'crf': self.config.crf
        }
    
    def add_frame(self, frame: np.ndarray, timestamp: float) -> bool:
        """Add a frame to the encoding queue"""
        if not self.is_encoding:
            return False
            
        try:
            self.frame_queue.put((frame.copy(), timestamp), block=False)
            return True
        except queue.Full:
            return False
    
    def _encoding_worker(self):
        """Worker thread for encoding frames"""
        while self.is_encoding:
            try:
                frame_data = self.frame_queue.get(timeout=0.5)
                
                if frame_data is None:  
                    break
                
                frame, timestamp = frame_data
                self._encode_frame(frame, timestamp)
                self.frame_count += 1
                self.frame_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Encoding error: {e}")
        
        self._flush_frames()
    
    def _encode_frame(self, frame: np.ndarray, timestamp: float):
        """Encode a single frame"""
        video_frame = av.VideoFrame.from_ndarray(frame, format='bgr24')
        pts = int((timestamp - self.start_time) * self.fps)
        video_frame.pts = pts
        
        for packet in self.stream.encode(video_frame):
            self.output.mux(packet)
    
    def _flush_frames(self):
        """Flush remaining frames when stopping"""
        try:
            for packet in self.stream.encode(None):
                self.output.mux(packet)
        except Exception as e:
            print(f"Error flushing frames: {e}")
    
    def stop(self):
        """Stop encoding and cleanup"""
        if not self.is_encoding:
            return
            
        self.is_encoding = False
        self.frame_queue.put(None)  # Sentinel to stop worker
        
        if self.encoding_thread:
            self.encoding_thread.join(timeout=5.0)
        
        elapsed = time.time() - self.start_time if self.start_time else 0
        actual_fps = self.frame_count / elapsed if elapsed > 0 else 0
        
        print(f"Encoding stopped. Frames: {self.frame_count}, "
              f"Duration: {elapsed:.2f}s, Effective FPS: {actual_fps:.2f}")
        
        if self.output:
            try:
                self.output.close()
            except Exception as e:
                print(f"Error closing output: {e}")


class Camera:
    
    def __init__(self, config: CameraConfig, recording_config: RecordingConfig):
        self.config = config
        self.recording_config = recording_config
        self.state = CameraState.IDLE
        
        # Camera setup
        self.cap = self._initialize_camera()
        self.width, self.height, self.fps = self._get_camera_properties()
        
        # Threading
        self.capture_thread = None
        self.running = False
        self.lock = threading.Lock()
        
        # Frame data
        self.current_frame = None
        self.preview_frame = None
        
        # Recording
        self.encoder = None
    
    def _initialize_camera(self) -> cv2.VideoCapture:
        """Initialize the camera capture"""
        cap = cv2.VideoCapture(self.config.index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            self.state = CameraState.ERROR
            raise RuntimeError(f"Cannot open camera {self.config.index}")
        return cap
    
    def _get_camera_properties(self) -> Tuple[int, int, float]:
        """Get camera properties and apply config overrides"""
        width = self.config.width or int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = self.config.height or int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.config.fps
        
        cam_fps = self.cap.get(cv2.CAP_PROP_FPS)
        if cam_fps > 0 and cam_fps <= 60:
            fps = min(fps, cam_fps)
        
        return width, height, fps
    
    def start_capture(self):
        """Start the camera capture thread"""
        if self.running:
            return
            
        self.running = True
        self.capture_thread = threading.Thread(target=self._capture_loop)
        self.capture_thread.daemon = True
        self.capture_thread.start()
    
    def stop_capture(self):
        """Stop the camera capture"""
        self.running = False
        if self.capture_thread:
            self.capture_thread.join()
        
        if self.encoder:
            self.encoder.stop()
            self.encoder = None
        
        self.cap.release()
    
    def start_recording(self):
        """Start recording"""
        if self.state == CameraState.RECORDING:
            return
        
        output_path = os.path.join(
            self.config.output_dir, 
            f'cam_{self.config.index}_{int(time.time())}.mp4'
        )
        
        self.encoder = FrameEncoder(
            output_path, self.width, self.height, self.fps, self.recording_config
        )
        self.encoder.start()
        self.state = CameraState.RECORDING
        
        print(f"[Camera {self.config.index}] Recording started: {output_path}")
    
    def stop_recording(self):
        """Stop recording"""
        if self.state != CameraState.RECORDING:
            return
            
        if self.encoder:
            self.encoder.stop()
            self.encoder = None
        
        self.state = CameraState.IDLE
        print(f"[Camera {self.config.index}] Recording stopped")
    
    def _capture_loop(self):
        """Main capture loop running in separate thread"""
        frame_interval = 1.0 / self.fps
        last_capture_time = time.time()
        frame_count = 0
        display_interval = 2  # Process every Nth frame for display
        
        while self.running:
            current_time = time.time()
            
            # Frame rate control
            if current_time - last_capture_time < frame_interval:
                time.sleep(0.001)
                continue
            
            # Capture frame
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.01)
                continue
            
            last_capture_time = current_time
            frame_count += 1
            
            # Add timestamp overlay
            self._add_timestamp_overlay(frame, current_time)
            
            # Update current frame
            with self.lock:
                self.current_frame = frame.copy()
            
            # Create preview frame (reduced frequency)
            if frame_count % display_interval == 0:
                self._update_preview_frame(frame)
            
            # Handle recording
            if self.state == CameraState.RECORDING and self.encoder:
                self.encoder.add_frame(frame, current_time)
    
    def _add_timestamp_overlay(self, frame: np.ndarray, timestamp: float):
        """Add timestamp overlay to frame"""
        if self.encoder and self.encoder.start_time:
            elapsed = timestamp - self.encoder.start_time
        else:
            elapsed = 0
        
        time_text = f'{int(elapsed//60):02d}:{int(elapsed%60):02d}.{int((elapsed%1)*1000):03d}'
        cv2.putText(frame, f'Cam {self.config.index}: {time_text}', 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    def _update_preview_frame(self, frame: np.ndarray):
        """Update the preview frame with scaling"""
        preview = cv2.resize(frame, (0, 0), 
                           fx=self.config.preview_scale, 
                           fy=self.config.preview_scale)
        
        with self.lock:
            self.preview_frame = preview
    
    def get_preview_frame(self) -> Optional[np.ndarray]:
        """Get the current preview frame"""
        with self.lock:
            return self.preview_frame.copy() if self.preview_frame is not None else None


class FrameCombiner:
    """Efficiently combines camera frames for display"""
    
    def __init__(self, max_width: int = 1920, max_height: int = 1080):
        self.max_width = max_width
        self.max_height = max_height
        self.last_grid = None
        self.last_update = 0
        self.update_interval = 1/30  # 30 FPS for UI updates
    
    def combine_frames(self, frames: List[np.ndarray], cols: int = 2) -> Optional[np.ndarray]:
        """Combine multiple frames into a grid layout"""
        current_time = time.time()
        
        # Use cached grid if update interval hasn't passed
        if (self.last_grid is not None and 
            (current_time - self.last_update) < self.update_interval):
            return self.last_grid
        
        if not frames:
            return None
        
        grid = self._create_grid(frames, cols)
        grid = self._scale_grid_if_needed(grid)
        
        self.last_grid = grid
        self.last_update = current_time
        return grid
    
    def _create_grid(self, frames: List[np.ndarray], cols: int) -> np.ndarray:
        """Create a grid layout from frames"""
        if not frames:
            return np.zeros((100, 100, 3), dtype=np.uint8)
        
        h_max = max(f.shape[0] for f in frames)
        w_max = max(f.shape[1] for f in frames)
        rows = (len(frames) + cols - 1) // cols
        
        grid = np.zeros((h_max * rows, w_max * cols, 3), dtype=np.uint8)
        
        for idx, frame in enumerate(frames):
            row = idx // cols
            col = idx % cols
            h, w = frame.shape[:2]
            
            y_offset = row * h_max + (h_max - h) // 2
            x_offset = col * w_max + (w_max - w) // 2
            grid[y_offset:y_offset+h, x_offset:x_offset+w] = frame
        
        return grid
    
    def _scale_grid_if_needed(self, grid: np.ndarray) -> np.ndarray:
        """Scale grid down if it exceeds maximum dimensions"""
        grid_h, grid_w = grid.shape[:2]
        
        if grid_w <= self.max_width and grid_h <= self.max_height:
            return grid
        
        scale = min(self.max_width / grid_w, self.max_height / grid_h)
        new_w = int(grid_w * scale)
        new_h = int(grid_h * scale)
        
        return cv2.resize(grid, (new_w, new_h), interpolation=cv2.INTER_AREA)

#main
class MultiCameraSystem:
    
    def __init__(self, camera_indices: List[int], output_dir: str = "output"):
        self.camera_indices = camera_indices
        self.output_dir = output_dir
        self.cameras: List[Camera] = []
        self.combiner = FrameCombiner()
        self.recording = False
        
        self._initialize_cameras()
    
    def _initialize_cameras(self):
        """Initialize all cameras"""
        os.makedirs(self.output_dir, exist_ok=True)
        
        recording_config = RecordingConfig()
        
        for index in self.camera_indices:
            try:
                camera_config = CameraConfig(
                    index=index,
                    output_dir=self.output_dir
                )
                camera = Camera(camera_config, recording_config)
                self.cameras.append(camera)
                print(f"Camera {index} initialized successfully")
            except RuntimeError as e:
                print(f"Failed to initialize camera {index}: {e}")
        
        if not self.cameras:
            raise RuntimeError("No cameras available")
    
    def start(self):
        """Start all cameras"""
        for camera in self.cameras:
            camera.start_capture()
        print("All cameras started")
    
    def stop(self):
        """Stop all cameras"""
        for camera in self.cameras:
            camera.stop_capture()
        print("All cameras stopped")
    
    def start_recording(self):
        """Start recording on all cameras"""
        if self.recording:
            return
            
        for camera in self.cameras:
            camera.start_recording()
        
        self.recording = True
        print("Recording started on all cameras")
    
    def stop_recording(self):
        """Stop recording on all cameras"""
        if not self.recording:
            return
            
        for camera in self.cameras:
            camera.stop_recording()
        
        self.recording = False
        print("Recording stopped on all cameras")
    
    def get_combined_preview(self) -> Optional[np.ndarray]:
        """Get combined preview from all cameras"""
        frames = []
        
        for camera in self.cameras:
            frame = camera.get_preview_frame()
            if frame is not None:
                frames.append(frame)
            else:
                # Create blank frame if camera not ready
                blank = np.zeros((
                    int(camera.height * camera.config.preview_scale),
                    int(camera.width * camera.config.preview_scale),
                    3
                ), dtype=np.uint8)
                frames.append(blank)
        
        return self.combiner.combine_frames(frames, cols=2)
    
    def run_preview_loop(self):
        """Run the main preview loop with keyboard controls"""
        window_name = 'MultiCam Preview'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        print("Controls:")
        print("  'r' - Start/Stop recording")
        
        last_ui_update = 0
        ui_refresh_rate = 1/30
        
        try:
            while True:
                current_time = time.time()
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    break
                elif key == ord('r'):
                    if self.recording:
                        self.stop_recording()
                    else:
                        self.start_recording()
                
                # Update UI at controlled rate
                if current_time - last_ui_update >= ui_refresh_rate:
                    combined = self.get_combined_preview()
                    if combined is not None:
                        cv2.imshow(window_name, combined)
                        last_ui_update = current_time
                else:
                    time.sleep(0.001)  # Small delay to reduce CPU usage
                    
        except KeyboardInterrupt:
            print("Interrupted by user")
        finally:
            cv2.destroyAllWindows()


def main():
    """Main entry point"""
    # Adjust according to your setup
    CAMERA_INDICES = [0, 1, 2] 
    OUTPUT_DIR = r"C:\Users\moreno\programming\Scientific_Works\senyum\Not_experiment\OUTPUT_VID"
    
    try:
        # Create and start the multi-camera system
        system = MultiCameraSystem(CAMERA_INDICES, OUTPUT_DIR)
        system.start()
        
        # Run the preview loop
        system.run_preview_loop()
        
    except Exception as e:
        print(f"System error: {e}")
    finally:
        # Cleanup
        if 'system' in locals():
            system.stop()
        print("Program exited cleanly")


if __name__ == "__main__":
    main()