"""
Functions to be used for calibration
See calibration_cobotique.ipynb
Léa Drolet-Roy
2025-02-21
"""

import os
import cv2
import math
import numpy as np
import pandas as pd
import pickle
import re
import glob

class Calibration():
    def __init__(self, Board, image_size):
        self.Board = Board
        self.checkerboard = Board.checkerboard
        self.square_size = Board.square_size
        self.image_size = image_size

        objpoints = np.zeros((self.checkerboard[0] * self.checkerboard[1], 3), np.float32)
        objpoints[:, :2] = np.mgrid[0:self.checkerboard[0], 0:self.checkerboard[1]].T.reshape(-1, 2)
        objpoints[:, :2] = objpoints[:, 0:2] * self.square_size
        self.objpoints = objpoints

        object_coords_3d = np.zeros((self.checkerboard[0] * self.checkerboard[1], 3), np.float32)
        object_coords_3d[:, :2] = np.mgrid[0:self.checkerboard[0], 0:self.checkerboard[1]].T.reshape(-1, 2)
        object_coords_3d[:, :2] = object_coords_3d[:, 0:2] * self.square_size
        self.object_coords_3d = object_coords_3d
    

    # To find images where we see the checkerboard, for individual cameras
    def saveImagesBoard(self, path, intrinsics_extension, manual_confirmation=False, save_corners=False, skip=10):
        output_dir = os.path.join(path, 'corners_found')
        os.makedirs(output_dir, exist_ok=True)  # Ensure output folder exists

        output_dir_corners = os.path.join(path, 'corners_found_shown')
        os.makedirs(output_dir_corners, exist_ok=True)  # Ensure output folder exists

        intrinsics_cam_listdirs_names = next(os.walk(os.path.join(path, 'intrinsics')))[1]
        valid_frame_count = 0
        image_area = self.image_size[0] * self.image_size[1]
        mask_covered = np.zeros((len(intrinsics_cam_listdirs_names), self.image_size[0], self.image_size[1]), dtype=np.uint8)

        if manual_confirmation:
            print("Appuyez sur 'Y' pour garder, 'N' pour ignorer, ou 'Q' pour quitter.")
        else:
            print('Images saving...')

        for i, cam in enumerate(intrinsics_cam_listdirs_names):
            os.makedirs(os.path.join(output_dir, cam), exist_ok=True)
            os.makedirs(os.path.join(output_dir_corners, cam), exist_ok=True)
            # Process frames
            if intrinsics_extension in ['jpg', 'png']:
                for fname in os.listdir(os.path.join(path, 'intrinsics', cam)):

                    if int(re.findall(r'\d+', fname)[-1]) % skip == 0:

                        img = cv2.imread(os.path.join(path, 'intrinsics', cam, fname))

                        frame_filename = os.path.join(output_dir, cam, f"{self.Board.type}_frame_{fname.split('.')[0]}.png")
                        if not frame_filename in os.listdir(os.path.join(path, 'intrinsics', cam)):

                            ret, markerCorners, markerIds, charucoCorners, charucoIds = self.Board.findCorners(img=img)

                            if ret:
                                # Get proportion of the image covered by the checkerboard
                                corners = charucoCorners.squeeze()
                                if corners.shape[0] > 2:
                                    x_min, y_min = np.min(corners, axis=0)
                                    x_max, y_max = np.max(corners, axis=0)
                                    mask_covered[i, int(x_min):int(x_max), int(y_min):int(y_max)] = 1

                                    if manual_confirmation or save_corners:
                                        # Visualisation des coins Charuco détectés
                                        img_with_charuco_corners = img.copy()
                                        img_with_charuco_corners = cv2.aruco.drawDetectedMarkers(img_with_charuco_corners, markerCorners, markerIds)
                                        img_with_charuco_corners = cv2.aruco.drawDetectedCornersCharuco(img_with_charuco_corners, charucoCorners, charucoIds)
                                        
                                        if manual_confirmation:
                                            cv2.imshow(f'{cam}_{fname}', img_with_charuco_corners)
                                            cv2.waitKey(100)

                                            # Demander à l'utilisateur si l'image est bonne
                                            key = cv2.waitKey(0) & 0xFF

                                            if key == ord('Y') or key == ord('y'):
                                                # Save image in /corners_found
                                                frame_filename = os.path.join(output_dir, cam, f"{self.Board.type}_frame_{fname.split('.')[0]}.png")
                                                cv2.imwrite(frame_filename, img)
                                                cv2.destroyAllWindows()
                                                valid_frame_count += 1

                                            elif key == ord('N') or key == ord('n'):
                                                cv2.destroyAllWindows()
                                                continue

                                            elif key == ord('Q') or key == ord('q'):
                                                cv2.destroyAllWindows()
                                                break
                                        
                                        if save_corners:
                                            frame_filename = os.path.join(output_dir_corners, cam, f"{self.Board.type}_frame_{fname.split('.')[0]}.png")
                                            cv2.imwrite(frame_filename, img_with_charuco_corners)

                                            frame_filename = os.path.join(output_dir, cam, f"{self.Board.type}_frame_{fname.split('.')[0]}.png")
                                            cv2.imwrite(frame_filename, img)
                                            valid_frame_count += 1
                                    
                                    else:
                                        # Save image in /corners_found
                                        frame_filename = os.path.join(output_dir, cam, f"{self.Board.type}_frame_{fname.split('.')[0]}.png")
                                        cv2.imwrite(frame_filename, img)
                                        valid_frame_count += 1
            
            # Process video
            if intrinsics_extension in ['mp4', 'avi', 'mjpeg']:
                video_path = glob.glob(os.path.join(path, 'intrinsics', cam, f'*.{intrinsics_extension}'))
                if len(video_path) == 0:
                    print(f'The folder {os.path.join(path, 'intrinsics', cam)} does not contain any .{intrinsics_extension} video files.')
                    continue

                frame_count = 0

                for img_path in video_path:
                    cap = cv2.VideoCapture(img_path)
                    frame_count = 0

                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            break
                        
                        if frame_count % skip == 0:
                            ret, markerCorners, markerIds, charucoCorners, charucoIds = self.Board.findCorners(img=frame)
                            if ret:
                                # Get proportion of the image covered by the checkerboard
                                corners = charucoCorners.squeeze()
                                if corners.shape[0] > 2:
                                    x_min, y_min = np.min(corners, axis=0)
                                    x_max, y_max = np.max(corners, axis=0)
                                    mask_covered[i, int(x_min):int(x_max), int(y_min):int(y_max)] = 1

                                    # Save image in /corners_found
                                    frame_filename = os.path.join(output_dir, cam, f"{self.Board.type}_frame_{frame_count:03d}.png")
                                    cv2.imwrite(frame_filename, frame)
                                    valid_frame_count += 1

                                    if save_corners:
                                        img_with_charuco_corners = img.copy()
                                        img_with_charuco_corners = cv2.aruco.drawDetectedMarkers(img_with_charuco_corners, markerCorners, markerIds)
                                        img_with_charuco_corners = cv2.aruco.drawDetectedCornersCharuco(img_with_charuco_corners, charucoCorners, charucoIds)
                                        frame_filename = os.path.join(output_dir, cam, f"{self.Board.type}_frame_{fname.split('.')[0]}.png")
                                        cv2.imwrite(frame_filename, img_with_charuco_corners)
                        frame_count += 1

            total_area_covered = np.sum(mask_covered[i]) / image_area
            print(f"Coverage for cam {cam}: {total_area_covered:.2%}")

        cv2.destroyAllWindows()
        print(f"Saved {valid_frame_count} valid frames.")

        return valid_frame_count, total_area_covered, mask_covered
    
    def getFrameCount(self, name):
        name_no_ext = name.split('.')[0]
        frame_count = (name_no_ext.split('_')[3].split('-')[0], name_no_ext.split('_')[-1])


        'charuco_frame_extrinsic2_009-camera001_frame00000.png'

        return frame_count

    def saveStereoData(self, path, cams):
        # Create array of all stereo images with 'im_name, img_pts, cam_ timestamp'
        stereo_images = {'Name':[], 'Camera':[], 'Corners':[], 'Charuco_Corners':[], 'Ids':[]}
        path_corners = os.path.join(path, 'corners_found')

        # maximal number of images per camera pair
        Nmax = 100

        for cam1 in cams:
            path_cam1 = os.path.join(path_corners, cam1)

            for cam2 in cams[cams.index(cam1)+1:]:
                path_cam2 = os.path.join(path_corners, cam2)

                im_saved = 0
                for name1 in os.listdir(path_cam1):
                    t1 = self.getFrameCount(name1)

                    for name2 in os.listdir(path_cam2):
                        t2 = self.getFrameCount(name2)
                        
                        if t1 == t2:
                            Lret, LmarkerCorners, LmarkerIds, LcharucoCorners, LcharucoIds = self.Board.findCorners(im_name=os.path.join(path_cam1, name1))
                            Rret, RmarkerCorners, RmarkerIds, RcharucoCorners, RcharucoIds = self.Board.findCorners(im_name=os.path.join(path_cam2, name2))

                            Ncommon_ids = 0
                            if self.Board.type == 'charuco' and Lret and Rret:
                                _, _, _, common_ids = self.Board.getObjectImagePoints(LcharucoCorners, LcharucoIds, RcharucoCorners, RcharucoIds)
                                if common_ids is not None:
                                    Ncommon_ids = len(common_ids)

                            if self.Board.type == 'checker' or Ncommon_ids >= 6:
                                stereo_images['Name'].append(name1)
                                stereo_images['Camera'].append(cam1)
                                stereo_images['Corners'].append(LmarkerCorners)
                                stereo_images['Charuco_Corners'].append(LcharucoCorners)
                                stereo_images['Ids'].append(LcharucoIds)

                                stereo_images['Name'].append(name2)
                                stereo_images['Camera'].append(cam2)
                                stereo_images['Corners'].append(RmarkerCorners)
                                stereo_images['Charuco_Corners'].append(RcharucoCorners)
                                stereo_images['Ids'].append(RcharucoIds)

                                im_saved += 1
                    
                        if im_saved >= Nmax:
                            break
                
                print(cam1, cam2, im_saved)
            
        with open('stereo_data.pkl', 'wb') as f:
            pickle.dump(stereo_images, f)

        return


    def StereoCalibration(self, leftparams, rightparams, Left_corners, Left_ids, Right_corners, Right_ids):
        StereoParams = {}
        calibrated = False
        
        k1 = leftparams['Intrinsic']
        d1 = leftparams['Distortion']
        k2 = rightparams['Intrinsic']
        d2 = rightparams['Distortion']
        
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)
        flags = 0
        flags |= cv2.CALIB_FIX_INTRINSIC

        if self.Board.type == 'charuco':
            obj_points = []  # 3D world coordinates
            img_points_left = []  # 2D image coordinates (left camera)
            img_points_right = []  # 2D image coordinates (right camera)
            Nobj = 0

            for charuco_corners_l, charuco_ids_l, charuco_corners_r, charuco_ids_r in zip(Left_corners, Left_ids, Right_corners, Right_ids):
                # Find common detected corners (based on Charuco IDs)
                obj_pts, img_pts_l, img_pts_r, common_ids = self.Board.getObjectImagePoints(charuco_corners_l, charuco_ids_l, charuco_corners_r, charuco_ids_r)
                if common_ids is not None and len(common_ids) >= 1:
                    obj_points.append(obj_pts)
                    Nobj += len(obj_pts)
                    img_points_left.append(img_pts_l)
                    img_points_right.append(img_pts_r)

        elif self.Board.type == 'checker':
            obj_points = []
            for i in range(len(Left_corners)):
                obj_points.append(self.Board.objpoints)

        if Nobj > 6:
            (ret, K1, D1, K2, D2, R, t, E, F) = cv2.stereoCalibrate(obj_points, img_points_left, img_points_right, k1, d1, k2, d2, self.image_size, criteria=criteria, flags=flags)
            
            T = np.vstack((np.hstack((R, t)), np.array([0, 0, 0, 1])))

            StereoParams['Transformation'] = T
            StereoParams['Essential'] = E
            StereoParams['Fundamental'] = F
            StereoParams['MeanError'] = ret
            StereoParams['Nobj'] = Nobj

            calibrated = True

        return calibrated, StereoParams


    def SaveParameters(self, camL, camR, Stereo_Params, Left_Params, Right_Params):
        Parameters = Stereo_Params.copy()  # Ensure we're not modifying the original dictionary
        Parameters['SquareSize'] = self.square_size
        Parameters['BoardSize'] = self.checkerboard

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


    def var_name(self, var, scope=globals()):
        return [name for name, value in scope.items() if value is var][0]

    def compare_transfo(self, T12, T23, T13):
        print(f'Original {self.var_name(T13)}')
        print(T13)
        T13_calc = T12 @ T23
        print(f'Calculated {self.var_name(T13)}')
        print(T13_calc)
        print('Calculated I')
        print(np.linalg.inv(T13_calc) @ T13, '\n')
        return

    ## 3D reprojection error
    # Function to project 3D points to 2D
    def project_points(self, points_3d, projMat, K, D):
        rvec, _ = cv2.Rodrigues(projMat[0:3,0:3])
        projected_2d, _ = cv2.projectPoints(points_3d, rvec, projMat[0:3,3], K, D)
        return projected_2d.squeeze()

    # Compute RMSE
    def compute_rmse(self, original_pts, points_3d, projMat, K, D):
        projected_pts = self.project_points(points_3d, projMat, K, D)
        error = np.linalg.norm(original_pts - projected_pts, axis=1)  # Euclidean distance per point
        rmse = np.sqrt(np.mean(error**2))  # Compute RMSE
        return rmse