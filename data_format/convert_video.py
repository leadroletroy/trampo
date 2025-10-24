### Convert .avi to .mp4 or a series of .png
import cv2
import os
from tqdm import tqdm
import glob

def convert_video_name(input_file):
    name_no_spaces = input_file.replace(" ", "")
    name_no_par = name_no_spaces.replace("(", "_").replace(")", "")
    output_file = name_no_par.split('.')[0] + '.mp4'

    return output_file

def avi_to_mp4(input_file, output_file=None):
    cap = cv2.VideoCapture(input_file)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if output_file == None:
        output_file = convert_video_name(input_file)

    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)

    cap.release()
    out.release()
    #print("Conversion complete! ✅")


def mp4_to_png(video_path, output_folder=None):
    # Create output folder if it doesn't exist
    if output_folder == None:
        output_folder = video_path.replace('Videos', 'Images').split('.')[0]
    os.makedirs(output_folder, exist_ok=True)

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count = 0

    with tqdm(total=total_frames, desc="Extracting frames") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break  # End of video

            # Save frame as PNG
            frame_filename = os.path.join(output_folder, f"frame_{frame_count:05d}.png")
            cv2.imwrite(frame_filename, frame)
            frame_count += 1
            pbar.update(1)

    cap.release()
    print(f"Done. Extracted {frame_count} frames to '{output_folder}'.")


def png_to_mp4(image_folder, output_video_path=None, fps=120):
    # Créer le chemin de sortie s'il n'est pas précisé
    if output_video_path is None:
        output_video_path = image_folder.replace('Images', 'Videos') + '.mp4'
    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)

    # Récupérer les fichiers PNG triés
    image_files = sorted(glob.glob(os.path.join(image_folder, 'vis_frame_*.png')))
    if not image_files:
        print(f"Aucune image trouvée dans le dossier '{image_folder}'")
        return

    # Lire la première image pour obtenir la taille
    first_frame = cv2.imread(image_files[0])
    height, width, _ = first_frame.shape

    # Initialiser l'objet VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    with tqdm(total=len(image_files), desc="Creating video") as pbar:
        for image_file in image_files:
            frame = cv2.imread(image_file)
            out.write(frame)
            pbar.update(1)

    out.release()
    print(f"Vidéo créée : {output_video_path}")


# Example usage
if __name__ == "__main__":

    path = "/mnt/D494C4CF94C4B4F0/Trampoline_avril2025/Videos_trampo_avril2025/20250428AM"

    for file in tqdm(sorted(os.listdir(path))):
        if file.split('.')[-1] == 'avi':
            video_file = os.path.join(path, file)
            if not os.path.isfile(convert_video_name(video_file)):
                avi_to_mp4(video_file)
            #mp4_to_png(video_file, )