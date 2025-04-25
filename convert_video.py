### Convert .avi to .mp4
import cv2

input_file = r'D:\Images\CRME\intrinsics\focusloin-Camera12(M11139).avi'
output_file = r'D:\Images\CRME\intrinsics\focusloin-Camera12(M11139).mp4'

cap = cv2.VideoCapture(input_file)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    out.write(frame)

cap.release()
out.release()
print("Conversion complete! ✅")