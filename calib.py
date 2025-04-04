"""
Calibration module for CheckerBoard or CharucoBoard

Auteurs : Léa Drolet-Roy

Création : 2025-03-17
Dernière modification : 
"""

### Imports
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pickle
import glob


### Checkerboard or CharucoBoard ?
while not Board:
    board = input('Enter "checker" OR "charuco" to select the board to use')
    if board == 'checker':
        from Checkerboard import Checkerboard
        checkerboard = tuple(input('Checkerboard dimensions? (x,y)'))
        image_size = tuple(input('Image size? (x,y)'))
        Board = Checkerboard(checkerboard, image_size)

    elif board == 'charuco':
        from Charucoboard import Charucoboard
        checkerboard = tuple(input('Checkerboard dimensions? (x,y)'))
        image_size = tuple(input('Image size? (x,y)'))
        square_size = int(input('Square size?'))
        marker_size = int(input('Marker size?'))
        aruco_name = input('Aruco name?')
        Board = Charucoboard(checkerboard, square_size, marker_size, aruco_name)

    else:
        print('Invalid board selected, try again.')


### Global parameters
cams = ['c1', 'c2', 'c3', 'c5', 'c6', 'c7', 'c8']
time_threshold = 15

path = r'C:\Users\LEA\Desktop\Poly\Trampo\video_test'
calib_dir = os.path.join(path, 'corners_found')
os.makedirs(calib_dir, exist_ok=True)  # Ensure output folder exists


### Extract and save frames where CharucoBoard is detected
def extract_save_frames(intrinsics_extension = 'mp4'):
    intrinsics_cam_listdirs_names = next(os.walk(os.path.join(path, 'intrinsics')))[1]

    for _, cam in enumerate(intrinsics_cam_listdirs_names):
        os.makedirs(os.path.join(calib_dir, cam), exist_ok=True)
        video_path = glob.glob(os.path.join(path, 'intrinsics', cam, f'*.{intrinsics_extension}'))
        if len(video_path) == 0:
            raise ValueError(f'The folder {os.path.join(path, 'intrinsics', cam)} does not contain any .{intrinsics_extension} video files.')

        frame_count = 0
        valid_frame_count = 0

        for img_path in video_path:
            cap = cv2.VideoCapture(img_path)
            frame_count = 0
            valid_frame_count = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                ret, _, _, _ = Board.findCorners(img=frame)
                if ret:
                    valid_frame_count += 1
                    frame_filename = os.path.join(calib_dir, cam, f"charuco_frame_{valid_frame_count:03d}.png")
                    cv2.imwrite(frame_filename, frame)

                frame_count += 1

    print(f"Processed {frame_count} frames, saved {valid_frame_count} valid Charuco frames.")
    return


### Save stereo matching frames
def save_stereo_frames(cams):
    video_paths = []

    for cam1 in cams:
        path_cam1 = os.path.join(path, 'intrinsics', cam1)
        for cam2 in cams[cams.index(cam1)+1:]:
            path_cam2 = os.path.join(path, 'intrinsics', cam2)
            video_paths = [os.path.join(path_cam1, os.listdir(path_cam1)[0]), os.path.join(path_cam2, os.listdir(path_cam2)[0])]

            calib_dir = os.path.join(path, 'stereo', f'{cam1}_{cam2}')
            os.makedirs(calib_dir, exist_ok=True)

            valid_frame_count = 0  # This counts frames where both cameras detect the board
            caps = [cv2.VideoCapture(video_path) for video_path in video_paths]

            while True:
                detections = []
                any_failed = False  # Track if any video ended

                for i, cap in enumerate(caps):
                    ret, frame = cap.read()
                    if not ret:
                        any_failed = True  # Stop processing if any video ends
                        break
                    
                    ret, _, _, ids = Board.findCorners(img=frame)
                    if ret:
                        detections.append((i, frame))

                if any_failed:
                    break  # Exit loop if any camera stops

                # Save frames only if both cameras detect the board
                if len(detections) == 2:
                    valid_frame_count += 1
                    for cam_idx, frame in detections:
                        frame_filename = os.path.join(calib_dir, f"c{cam_idx}_frame_{valid_frame_count:03d}.png")
                        cv2.imwrite(frame_filename, frame)

            print(f'{cam1}-{cam2}: Saved {valid_frame_count} valid frames')
    return

        

def StereoCalibration(leftparams, rightparams, Left_corners, Left_ids, Right_corners, Right_ids):
    StereoParams = {}
    ret = False
    
    k1 = leftparams['Intrinsic']
    d1 = leftparams['Distortion']
    k2 = rightparams['Intrinsic']
    d2 = rightparams['Distortion']
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)
    flags = 0
    flags |= cv2.CALIB_FIX_INTRINSIC


    if board == 'charuco':
        obj_points = []  # 3D world coordinates
        img_points_left = []  # 2D image coordinates (left camera)
        img_points_right = []  # 2D image coordinates (right camera)

        for charuco_corners_l, charuco_ids_l, charuco_corners_r, charuco_ids_r in zip(Left_corners, Left_ids, Right_corners, Right_ids):
            # Find common detected corners (based on Charuco IDs)
            obj_pts, img_pts_l, img_pts_r, common_ids = Board.getObjectImagePoints(charuco_corners_l, charuco_ids_l, charuco_corners_r, charuco_ids_r)
            
            obj_points.append(obj_pts)
            img_points_left.append(img_pts_l)
            img_points_right.append(img_pts_r)

    elif board == 'checker':
        obj_points = []
        for i in range(len(Left_corners)):
            obj_points.append(Board.objpoints)


    if len(obj_points) > 1:
        (ret, K1, D1, K2, D2, R, t, E, F) = cv2.stereoCalibrate(obj_points, img_points_left, img_points_right, k1, d1, k2, d2, image_size, criteria=criteria, flags=flags)
    
        T = np.vstack((np.hstack((R,t)),np.array([0,0,0,1])))
        
        StereoParams['Transformation'] = T
        StereoParams['Essential'] = E
        StereoParams['Fundamental'] = F
        StereoParams['MeanError'] = ret
        
    return ret, StereoParams


def SaveParameters(camL, camR, Stereo_Params, Left_Params, Right_Params):
    Parameters = Stereo_Params.copy()  # Ensure we're not modifying the original dictionary

    for Lkey in Left_Params.keys():
        name = 'L_'+str(Lkey)
        Parameters[name] = Left_Params[Lkey]
        
    for Rkey in Right_Params.keys():
        name = 'R_'+str(Rkey)
        Parameters[name] = Right_Params[Rkey]

    # Remove imgpoints if they exist in Left_Params and Right_Params
    Parameters = {k: v for k, v in Parameters.items() if k not in ['L_Imgpoints', 'R_Imgpoints']}

    # Save the Parameters dictionary into an npz file
    file = f'{camL}_{camR}_parameters.npz'
    np.savez(file, **Parameters)
    npz = dict(np.load(file))
    np.savez(file, **npz)

    return



path_stereo = os.path.join(path, 'stereo')

for camL in cams[4:]:
    
    Left_Params = {}
    Left_Params['Intrinsic'] = K[cams.index(camL)]
    Left_Params['Distortion'] = D[cams.index(camL)]

    for camR in cams[cams.index(camL)+1:]:
        print(camL, camR)
        stereopath = os.path.join(path_stereo, f'{camL}_{camR}')

        Left_Paths = [os.path.join(stereopath, fname) for fname in list(os.listdir(stereopath)) if fname.split('_')[0] == 'c0']
        Right_Paths = [os.path.join(stereopath, fname) for fname in list(os.listdir(stereopath)) if fname.split('_')[0] == 'c1']
        Right_Params = {}
        
        if len(Left_Paths) >= 1 and len(Right_Paths) >= 1:
            Left_corners, Left_ids, Left_Paths, Right_corners, Right_ids, Right_Paths = GenerateImagepoints(Left_Paths, Right_Paths)
            if len(Left_corners) >= 1 and len(Right_corners) >= 1:

                Right_Params['Intrinsic'] = K[cams.index(camR)]
                Right_Params['Distortion'] = D[cams.index(camR)]
             
                ret, Stereo_Params = StereoCalibration(Left_Params, Right_Params, Left_corners, Left_ids, Right_corners, Right_ids)
                if ret:
                    print('Transformation Matrix:')
                    print(Stereo_Params['Transformation'])

                    SaveParameters(camL, camR, Stereo_Params, Left_Params, Right_Params)
                else:
                    print('Not enough corners', '\n')
            else:
                print('Not enough corners', '\n')
        else:
            print('Not enough images', '\n')



def var_name(var, scope=globals()):
    return [name for name, value in scope.items() if value is var][0]

def compare_transfo(T12, T23, T13):
    print(f'Original {var_name(T13)}')
    print(T13)
    T13_calc = T12 @ T23
    print(f'Calculated {var_name(T13)}')
    print(T13_calc)
    print('Calculated I')
    print(np.linalg.inv(T13_calc) @ T13, '\n')
    return


## 3D reprojection error
# Function to project 3D points to 2D
def project_points(points_3d, projMat, K, D):
    rvec, _ = cv2.Rodrigues(projMat[0:3,0:3])
    projected_2d, _ = cv2.projectPoints(points_3d, rvec, projMat[0:3,3], K, D)
    return projected_2d.squeeze()

# Compute RMSE
def compute_rmse(original_pts, points_3d, projMat, K, D):
    projected_pts = project_points(points_3d, projMat, K, D)
    error = np.linalg.norm(original_pts - projected_pts, axis=1)  # Euclidean distance per point
    rmse = np.sqrt(np.mean(error**2))  # Compute RMSE
    return rmse


## Stereo initial extrinsic parameters
# Create array of all stereo images with 'im_name, img_pts, cam_ timestamp'
stereo_images = {'Name':[], 'Camera':[], 'Corners':[], 'Charuco_Corners':[], 'Ids':[]}
path_stereo = os.path.join(path, 'stereo')

for cam1 in cams:
    for cam2 in cams[cams.index(cam1)+1:]:
        im_saved = 0

        stereopath = os.path.join(path_stereo, f'{cam1}_{cam2}')
        N = len(os.listdir(stereopath))
        
        number_choice = np.linspace(1, int(N//2), int(N//2))
        number_choice = np.round(number_choice).astype(int)

        for i in number_choice:
                
            name1 = f'c0_frame_{i:03d}.png'
            name2 = f'c1_frame_{i:03d}.png'
            
            im1In = name1 in list(os.listdir(stereopath))
            im2In = name2 in list(os.listdir(stereopath))
                    
            if im1In and im2In:
                ret, Lcorners, Lcharuco_corners, Lcharuco_ids = Board.findCorners(im_name=os.path.join(stereopath, name1))
                ret, Rcorners, Rcharuco_corners, Rcharuco_ids = Board.findCorners(im_name=os.path.join(stereopath, name2))

                if Lcharuco_corners is not None and Rcharuco_corners is not None:
                    _, _, _, common_ids = Board.getObjectImagePoints(Lcharuco_corners, Lcharuco_ids, Rcharuco_corners, Rcharuco_ids)

                    if len(common_ids) > 1:
                        stereo_images['Name'].append(f'{cam1}_{cam2}_{name1}')
                        stereo_images['Camera'].append(int(cam1[-1]))
                        stereo_images['Corners'].append(Lcorners)
                        stereo_images['Charuco_Corners'].append(Lcharuco_corners)
                        stereo_images['Ids'].append(Lcharuco_ids)

                        stereo_images['Name'].append(f'{cam1}_{cam2}_{name2}')
                        stereo_images['Camera'].append(int(cam2[-1]))
                        stereo_images['Corners'].append(Rcorners)
                        stereo_images['Charuco_Corners'].append(Rcharuco_corners)
                        stereo_images['Ids'].append(Rcharuco_ids)

                        im_saved += 1
            
            if im_saved == 20:
                break
        
        print(cam1, cam2, im_saved)
    
with open('stereo_data_trampo.pkl', 'wb') as f:
    pickle.dump(stereo_images, f)
    



def visualize_projection(img_path, pts_2d_original, pts_3d, projMat, K, D):
    """
    Visualize original 2D points and projected 3D points on an image.

    Parameters:
    - img_path: Path to the image.
    - pts_2d_original: Original 2D keypoints (Nx2).
    - pts_3d: Triangulated 3D points (Nx3).
    - projMat: Projection matrix (3x4).
    - K: Camera intrinsic matrix (3x3).
    - D: Distortion coefficients (1x5 or 1x4).
    """

    # Display the image
    plt.figure(figsize=(10, 6))

    for i in range(len(img_path)):

        # Load the image
        img = cv2.imread(img_path[i])
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path[i]}")

        # Project 3D points onto the image plane
        pts_2d_projected, _ = cv2.projectPoints(pts_3d[i], projMat[i][0:3,0:3], projMat[i][0:3,3], K[i], D[i])

        # Reshape projected points
        pts_2d_projected = pts_2d_projected.squeeze()

        # Draw the original and projected points
        for j in range(len(pts_2d_original[i])):
            x_orig, y_orig = map(int, pts_2d_original[i][j])  # Original 2D point
            x_proj, y_proj = map(int, pts_2d_projected[j])  # Reprojected 3D point

            # Draw original points in BLUE
            cv2.circle(img, (x_orig, y_orig), 4, (255, 0, 0), -1)

            # Draw reprojected points in GREEN
            cv2.circle(img, (x_proj, y_proj), 4, (0, 255, 0), -1)

            # Draw line between them (error visualization)
            cv2.line(img, (x_orig, y_orig), (x_proj, y_proj), (0, 255, 255), 1)

            fig = plt.subplot(1,2,i+1)
            fig.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    plt.title("Original 2D Points (Blue) vs. Projected 3D Points (Green)")
    plt.show()


def fun(params):
    errors = np.empty((n_particles,))
    params = np.array(params)

    for n in range(n_particles):
        projMat = np.empty((num_cameras+1, 3, 4))
        projMat[0] = np.hstack((np.eye((3)), np.zeros((3,1))))

        for cam_idx in range(num_cameras):
            cam_params = params[n][cam_idx * num_params_per_cam : (cam_idx + 1) * num_params_per_cam]
            r1, r2, r3, t1, t2, t3 = cam_params
            rvec = np.array([r1, r2, r3])
            R, _ = cv2.Rodrigues(rvec)
            t = np.array([t1, t2, t3]).reshape((3,1))
            projMat[cam_idx+1] = np.hstack((R, t))  #[cam_idx+1]

        RMSE = {c:[] for c in range(6)}
        # Loop on stereo images checkberboard points
        for i in range(0, len(stereo_images['Camera']) - 1, 2):
            j = i+1 # stereo image is the next one

            pts1_im = stereo_images['Charuco_Corners'][i].squeeze()
            pts2_im = stereo_images['Charuco_Corners'][j].squeeze()

            c1 = cams.index(f'c{stereo_images['Camera'][i]}')
            c2 = cams.index(f'c{stereo_images['Camera'][j]}')

            undist_pts1 = cv2.undistortPoints(pts1_im, K[c1], D[c1]).reshape(-1, 2)  # Shape (2, N)
            undist_pts2 = cv2.undistortPoints(pts2_im, K[c2], D[c2]).reshape(-1, 2)  # Shape (2, N)

            if board == 'charuco':
                Lids = stereo_images['Ids'][i].squeeze()
                Rids = stereo_images['Ids'][j].squeeze()

                obj_pts, img_pts_l, img_pts_r, common_ids = Board.getObjectImagePoints(undist_pts1, Lids, undist_pts2, Rids)
                img_pts_l, img_pts_r = img_pts_l.squeeze(), img_pts_r.squeeze()

                ids_to_keepL = [list(Lids).index(t) for t in common_ids]
                ids_to_keepR = [list(Rids).index(t) for t in common_ids]
                pts1_im, pts2_im = pts1_im[ids_to_keepL], pts2_im[ids_to_keepL]

            elif board == 'checker':
                img_pts_l, img_pts_r = undist_pts1, undist_pts2

            # Perform triangulation
            pts_4d = cv2.triangulatePoints(projMat[c1], projMat[c2], img_pts_l.T, img_pts_r.T)
            points_3d = pts_4d[:3, :] / pts_4d[3, :]  # Shape (3, N)
            points_3d = points_3d.T  # Shape (N, 3)

            # Compute RMSE for both cameras
            rmse1 = compute_rmse(pts1_im, points_3d, projMat[c1], K[c1], D[c1])
            rmse2 = compute_rmse(pts2_im, points_3d, projMat[c2], K[c2], D[c2])

            RMSE[c1].append(rmse1)
            RMSE[c2].append(rmse2)

        # Per-camera mean RMSE
        rmse_mean = []
        for _, rmse in RMSE.items():
            rmse_mean.append(np.mean(rmse))

        errors[n] = np.mean(rmse_mean) #+ np.max(RMSE)   # OPTIONAL: add max error to cost function
    
    return errors
