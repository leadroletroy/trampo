import cv2
cam = cv2.VideoCapture(r"C:\Users\LEA\Desktop\Poly\Trampo\video_test\intrinsics\c2\M11140.mp4")
fps = cam.get(cv2.CAP_PROP_FPS)

print(fps)