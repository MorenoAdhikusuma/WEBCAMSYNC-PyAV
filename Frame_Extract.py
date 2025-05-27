import subprocess
import os

def extract_frames(input_file, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    cmd = ['ffmpeg', '-i', input_file, '-vsync', '0', os.path.join(output_dir, '%08d.png')]
    subprocess.run(cmd, check=True)
#pake ffprobe
def get_frame_timecodes(input_file):
    cmd = [
        'ffprobe', '-select_streams', 'v:0',
        '-show_frames',
        '-show_entries', 'frame=pkt_pts_time:frame_tags=timecode',
        '-of', 'csv',
        input_file
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, text=True, check=True)
    timecodes = []
    for line in result.stdout.splitlines():
        parts = line.split(',')
        if len(parts) >= 3 and parts[0] == 'frame':
            pts_time = parts[1]
            tc = parts[2]
            timecodes.append(tc)
    return timecodes

def rename_frames_with_timecode(frame_dir, timecodes):
    frames = sorted([f for f in os.listdir(frame_dir) if f.endswith('.png')])
    for frame_file, tc in zip(frames, timecodes):
        safe_tc = tc.replace(":", "-")
        old_path = os.path.join(frame_dir, frame_file)
        new_path = os.path.join(frame_dir, f"{safe_tc}.png")
        os.rename(old_path, new_path)
        print(f"Renamed {frame_file} -> {safe_tc}.png")

# input ama output directory nya ganti sesuaiin
input_file = r"C:\Users\moreno\programming\Scientific_Works\senyum\Not_experiment\OUTPUT_VID\cam_0.mp4"
output_dir = r"C:\Users\moreno\programming\Scientific_Works\senyum\Not_experiment\Extracted_frame"

extract_frames(input_file, output_dir)
timecodes = get_frame_timecodes(input_file)
rename_frames_with_timecode(output_dir, timecodes)
