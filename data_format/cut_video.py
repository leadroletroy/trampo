import cv2
import os
import shutil

# --- Input video ---
path = '/mnt/D494C4CF94C4B4F0/Trampoline_avril2025/Videos_trampo_avril2025/20250428AM'
processed_dir = path + "_splittedRAW"
os.makedirs(processed_dir, exist_ok=True)

cameras = ['Camera 1 (M11139)', 'Camera 2 (M11140)', 'Camera 3 (M11141)', 'Camera 4 (M11458)',
           'Camera 5 (M11459)', 'Camera 6 (M11461)', 'Camera 7 (M11462)', 'Camera 8 (M11463)']

seq = '6_5_drill_0428AM_003'

# --- Define output clips (start_frame, end_frame, output_path) ---
clips = [(0, 720, '6_0_drill_0428AM_007'), (720, 1680, '0_5_drill_0428AM_008')]

for cam in cameras:
    
    input_path = path + '/' + seq + '-' + cam + '.avi'
    index = input_path.find('-')

    # --- Open input video ---
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID', 'avc1', etc.

    # --- Iterate over clips ---
    for start_frame, end_frame, new_seq in clips:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        output_path = path + '/' + new_seq + '-' + cam + '.avi'
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        print(f"Writing {output_path}...")

        for i in range(start_frame, end_frame):
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)

        out.release()
        print(f"✅ Saved {output_path}")

    cap.release()

    # --- Move processed input video ---
    dest_path = os.path.join(processed_dir, os.path.basename(input_path))
    shutil.move(input_path, dest_path)
    print(f"📦 Moved {input_path} → {dest_path}")

print("All done!")
