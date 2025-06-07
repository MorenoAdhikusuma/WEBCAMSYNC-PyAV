import cv2
import threading
import time
import av
import numpy as np
import os
import queue
from collections import deque
from pygrabber.dshow_graph import FilterGraph

#GANTI SESUAI FOLDER OUTPUT DIR
OUTPUT_DIR = r"C:\Users\moreno\programming\Scientific_Works\senyum\Not_experiment\OUTPUT_VID"
os.makedirs(OUTPUT_DIR, exist_ok=True)

cameras_identifier = []

graph = FilterGraph()
devices = graph.get_input_devices()
for i, name in enumerate(devices):
    if name.startswith("GENERAL"):
        cameras_identifier.append(i)
        print(f"Found camera: {name}")
    else:
        print(f"Skipping camera: {name}")

class CameraWorker(threading.Thread):
    def _init_(self, cam_index):
        super()._init_()
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
            frame_time_text = f'{int(frame_time//60):02d}:{int(frame_time%60):02d}.{int((frame_time%1)*1000):03d}'
            
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
    """Efficiently combines camera frames for display"""
    def _init_(self, max_width=1920, max_height=1080):
        self.max_width = max_width
        self.max_height = max_height
        self.last_grid = None
        self.last_update = 0
        self.update_interval = 1/30  # 30 FPS for UI updates
        
    def combine_frames(self, frames, cols=2):
        current_time = time.time()
        
        # Return cached grid if not enough time has passed
        if self.last_grid is not None and (current_time - self.last_update) < self.update_interval:
            return self.last_grid
            
        if not frames:
            return None
            
        # Calculate grid dimensions
        h_max = max(f.shape[0] for f in frames)
        w_max = max(f.shape[1] for f in frames)
        rows = (len(frames) + cols - 1) // cols
        
        # Create the grid
        grid = np.zeros((h_max * rows, w_max * cols, 3), dtype=np.uint8)
        
        for idx, frame in enumerate(frames):
            r = idx // cols
            c = idx % cols
            h, w = frame.shape[:2]

            y_offset = r * h_max + (h_max - h) // 2
            x_offset = c * w_max + (w_max - w) // 2
            grid[y_offset:y_offset+h, x_offset:x_offset+w] = frame
        
        # Check if we need to scale down the grid
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
    
    # Try to open cameras and handle failures gracefully
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

    window_name = 'MultiCam Preview'
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