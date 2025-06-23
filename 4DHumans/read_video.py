import cv2

video_path = '/home/lea/4DHumans/4D-Humans/outputs/PHALP_converted.mp4'

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Erreur : impossible d'ouvrir la vidéo.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Fin de la vidéo ou lecture échouée.")
        break

    cv2.imshow('Vidéo', frame)

    # Quitter avec la touche 'q'
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()