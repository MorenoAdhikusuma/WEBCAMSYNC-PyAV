import cv2
import threading
import time
import av
import numpy as np
import os
import queue
from collections import deque
import datetime
from pygrabber.dshow_graph import FilterGraph
import subprocess

def add_start_time_metadata(mp4_path, start_time_str):
    """Adds creation_time metadata to an MP4 file using ffmpeg"""
    base, ext = os.path.splitext(mp4_path)
    output_path = base + "_withmeta" + ext

    cmd = [
        'ffmpeg',
        '-y',  
        '-i', mp4_path,
        '-metadata', f'creation_time={start_time_str}',  # <-- THIS IS THE FIX
        '-codec', 'copy',
        output_path
    ]

    try:
        subprocess.run(cmd, check=True)
        os.replace(output_path, mp4_path)  # Replace original
        print(f"[FFMPEG] Metadata added to {mp4_path}")
    except subprocess.CalledProcessError as e:
        print(f"[FFMPEG] Failed to add metadata: {e}")


#GANTI SESUAI FOLDER OUTPUT DIR
OUTPUT_DIR = r"C:\Users\moreno\programming\Scientific_Works\senyum\Not_experiment\OUTPUT_VID"
os.makedirs(OUTPUT_DIR, exist_ok=True)

cameras_identifier = []

graph = FilterGraph()
devices = graph.get_input_devices()
max_cams_to_check = 10  # Change if you expect more

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

    if name.startswith("Integrated Camera"):
        cameras_identifier.append(real_index)
        print(f"Using camera index {real_index}: {name}")
    else:
        print(f"Skipping camera index {real_index}: {name}")
    
    cap.release()

class CameraWorker(threading.Thread):
    def __init__(self, cam_index):
        super().__init__()
        self.cam_index = cam_index
        self.cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera {cam_index}")

        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        if self.fps <= 0 or self.fps > 60:
            self.fps = 30

        # For display
        self.frame = None
        self.running = True
        self.recording = False
        self.output = None
        self.stream = None
        self.lock = threading.Lock()
        self.start_time = None
        self.frame_count = 0
        
        # Frame queue for encoding
        self.frame_queue = queue.Queue(maxsize=30)  # Limit queue size to prevent memory issues
        self.encoding_thread = None
        
        # Downscale factor for preview (not recording)
        self.preview_scale = 0.5  # Display at half resolution

    def start_recording(self):
        filename = os.path.join(OUTPUT_DIR, f'cam_{self.cam_index}.mp4')
        self.output = av.open(filename, mode='w')
        
        # Use actual camera FPS for recording
        self.stream = self.output.add_stream('libx264', rate=self.fps)
        self.stream.width = self.width
        self.stream.height = self.height
        self.stream.pix_fmt = 'yuv420p'
        self.stream.options = {'preset': 'ultrafast', 'crf': '23'}  # Using ultrafast preset for lower CPU
        
        self.recording = True
        self.start_time = time.time()
        self.frame_count = 0
        
        # Start encoding thread
        self.encoding_thread = threading.Thread(target=self.encoding_worker)
        self.encoding_thread.daemon = True
        self.encoding_thread.start()
        
        print(f"[Camera {self.cam_index}] Recording started: {filename} at {self.fps} FPS")

    def encoding_worker(self):
        """Separate thread for encoding frames"""
        while self.recording:
            try:
                # Get frame from queue with timeout to avoid blocking indefinitely
                frame_data = self.frame_queue.get(timeout=0.5)
                
                # Check if this is a sentinel value signaling end of recording
                if frame_data is None:
                    break
                    
                frame, timestamp = frame_data
                
                # Encode the frame
                video_frame = av.VideoFrame.from_ndarray(frame, format='bgr24')
                pts = int((timestamp - self.start_time) * self.fps)
                video_frame.pts = pts
                
                for packet in self.stream.encode(video_frame):
                    self.output.mux(packet)
                
                self.frame_count += 1
                self.frame_queue.task_done()
                
            except queue.Empty:
                # Queue was empty, just continue
                continue
            except Exception as e:
                print(f"Error in encoding thread: {e}")
        
        # Flush any remaining frames when done
        try:
            for packet in self.stream.encode(None):
                self.output.mux(packet)
        except Exception as e:
            print(f"Error flushing frames: {e}")


    def stop_recording(self):
        if self.recording:
            self.recording = False
            
            # Signal encoding thread to finish
            self.frame_queue.put(None)
            
            if self.encoding_thread:
                self.encoding_thread.join(timeout=5.0)
            
            elapsed = time.time() - self.start_time
            actual_fps = self.frame_count / elapsed if elapsed > 0 else 0
            print(f"[Camera {self.cam_index}] Recording stopped. Frames: {self.frame_count}, "
                  f"Duration: {elapsed:.2f}s, Effective FPS: {actual_fps:.2f}")
            
            try:
                self.output.close()
            except Exception as e:
                print(f"Error closing output: {e}")
            
            # Add metadata after closing the file with UTC+7 timezone
            utc_plus_7 = datetime.timezone(datetime.timedelta(hours=7))
            iso_start_time = datetime.datetime.fromtimestamp(self.start_time, tz=utc_plus_7).isoformat(timespec='milliseconds')
            filename = os.path.join(OUTPUT_DIR, f'cam_{self.cam_index}.mp4')
            add_start_time_metadata(filename, iso_start_time)

    def run(self):
        target_fps = self.fps
        frame_interval = 1.0 / target_fps
        last_capture_time = time.time()
        
        # For frame skipping
        frame_count = 0
        display_interval = 2  # Only process every N frames for display

        while self.running:
            current_time = time.time()
            time_since_last_capture = current_time - last_capture_time
            
            # Sleep if we're ahead of schedule to reduce CPU spinning
            if time_since_last_capture < frame_interval:
                sleep_time = max(0.001, frame_interval - time_since_last_capture)
                time.sleep(sleep_time)
                continue
                
            # Read the frame
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.01)
                continue
                
            last_capture_time = current_time
            frame_count += 1
            
            # Get current timestamp
            timestamp = time.time()
            
            # Calculate frame time (time elapsed since recording started)
            frame_time = timestamp - self.start_time if self.recording and self.start_time else 0
            
            # Create UTC+7 timestamp for display
            utc_plus_7 = datetime.timezone(datetime.timedelta(hours=7))
            frame_time_text = datetime.datetime.fromtimestamp(timestamp, tz=utc_plus_7).isoformat(timespec='milliseconds')
            
            # Only process frames for display at reduced rate
            if frame_count % display_interval == 0:
                # Create a smaller preview frame
                preview_frame = cv2.resize(frame, (0, 0), fx=self.preview_scale, fy=self.preview_scale)
                
                # Add frame time to preview
                cv2.putText(preview_frame, f'Cam {self.cam_index}: {frame_time_text}', 
                            (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                # Update display frame
                with self.lock:
                    self.frame = preview_frame

            # Handle recording - use full quality frame
            if self.recording and not self.frame_queue.full():
                # Add frame time to recording frame
                cv2.putText(frame, f'Cam {self.cam_index}: {frame_time_text}', 
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Add to encoding queue
                try:
                    self.frame_queue.put((frame.copy(), timestamp), block=False)
                except queue.Full:
                    # Skip frame if queue is full
                    pass

        self.stop_recording()
        self.cap.release()

#ini buat combine CAM nya
class SmartCombiner:
    def __init__(self, max_width=1920, max_height=1080):
        self.max_width = max_width
        self.max_height = max_height
        self.last_grid = None
        self.last_update = 0
        self.update_interval = 1/60 # 30 FPS for UI updates

    def combine_frames(self, frames, cols=2):
        current_time = time.time()

        if self.last_grid is not None and (current_time - self.last_update) < self.update_interval:
            return self.last_grid

        if not frames:
            return None

        # ini ganti aja kalo kamera nya lebih
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
            new_w = int(grid_w * scale)
            new_h = int(grid_h * scale)
            grid = cv2.resize(grid, (new_w, new_h), interpolation=cv2.INTER_AREA)

        self.last_grid = grid
        self.last_update = current_time
        return grid



def main():
    NUM_CAMERAS = 3  # Change if you have more cams
    
    # Try to open cameras and handle failure gracefully and show error
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
    print("Press 'r' to start/stop recording, 'q' to quit.")

    window_name = 'MULTICAM'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)  # allow resize by mouse
    
    # Smart frame combiner
    combiner = SmartCombiner()
    
    # For UI refresh rate control
    last_ui_update = 0
    ui_refresh_rate = 1/30  # 30fps UI refresh

    try:
        while True:
            current_time = time.time()
            
            # Only update UI at specified refresh rate
            if current_time - last_ui_update < ui_refresh_rate:
                # Process events but don't redraw
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    recording = not recording
                    if recording:
                        print("Starting recording all cameras...")
                        for cam in cameras:
                            cam.start_recording()
                    else:
                        print("Stopping recording all cameras...")
                        for cam in cameras:
                            cam.stop_recording()
                            
                # Sleep a bit to reduce CPU usage
                time.sleep(0.001)
                continue
                
            # Get frames from cameras
            frames = []
            for cam in cameras:
                with cam.lock:
                    if cam.frame is not None:
                        frames.append(cam.frame)
                    else:
                        blank = np.zeros((int(cam.height * cam.preview_scale), 
                                         int(cam.width * cam.preview_scale), 3), 
                                         dtype=np.uint8)
                        frames.append(blank)

            # Combine frames efficiently
            combined = combiner.combine_frames(frames, cols=2)
            
            if combined is not None:
                cv2.imshow(window_name, combined)
                last_ui_update = current_time

            # Process keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                recording = not recording
                if recording:
                    print("Starting recording all cameras...")
                    for cam in cameras:
                        cam.start_recording()
                else:
                    print("Stopping recording all cameras...")
                    for cam in cameras:
                        cam.stop_recording()
                        
    except KeyboardInterrupt:
        print("Interrupted by user. Cleaning up...")
    finally:
        # Ensure cameras are properly stopped
        for cam in cameras:
            cam.running = False
            cam.join()

        cv2.destroyAllWindows()
        print("Program exited cleanly.")

if __name__ == "__main__":
    main()