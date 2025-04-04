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
from pose2sim_trampo.common import euclidean_distance

class Calibration():
    def __init__(self, Board):
        self.Board = Board
        self.checkerboard = Board.checkerboard
        self.square_size = Board.square_size

        objpoints = np.zeros((self.checkerboard[0] * self.checkerboard[1], 3), np.float32)
        objpoints[:, :2] = np.mgrid[0:self.checkerboard[0], 0:self.checkerboard[1]].T.reshape(-1, 2)
        objpoints[:, :2] = objpoints[:, 0:2] * self.square_size
        self.objpoints = objpoints

        object_coords_3d = np.zeros((self.checkerboard[0] * self.checkerboard[1], 3), np.float32)
        object_coords_3d[:, :2] = np.mgrid[0:self.checkerboard[0], 0:self.checkerboard[1]].T.reshape(-1, 2)
        object_coords_3d[:, :2] = object_coords_3d[:, 0:2] * self.square_size
        self.object_coords_3d = object_coords_3d

    
    # To find images where we see the checkerboard, for individual cameras
    def saveImagesBoard(self, path, intrinsics_extension):
        output_dir = os.path.join(path, 'corners_found')
        os.makedirs(output_dir, exist_ok=True)  # Ensure output folder exists

        intrinsics_cam_listdirs_names = next(os.walk(os.path.join(path, 'intrinsics')))[1]
        valid_frame_count = 0

        for cam in intrinsics_cam_listdirs_names:
            os.makedirs(os.path.join(output_dir, cam), exist_ok=True)

            # Process frames
            if intrinsics_extension in ['jpg', 'png']:
                for fname in os.listdir(os.path.join(dir, cam)):
                    savepath = os.path.join(path, 'calibration', 'intrinsics', cam)
                    if not os.path.isdir(savepath):
                        os.makedirs(savepath)
                    savename = dir[-4:] + '_' + cam + fname

                    img = cv2.imread(os.path.join(dir, cam, fname))
                    ret, _, _, _ = self.Board.findCorners(img=img)
                    if ret == True:
                        cv2.imwrite(os.path.join(savepath, savename), img)
                        valid_frame_count += 1
            
            # Process video
            if intrinsics_extension in ['mp4', 'avi', 'mjpeg']:
                video_path = glob.glob(os.path.join(path, 'intrinsics', cam, f'*.{intrinsics_extension}'))
                if len(video_path) == 0:
                    raise ValueError(f'The folder {os.path.join(path, 'intrinsics', cam)} does not contain any .{intrinsics_extension} video files.')

                frame_count = 0
                valid_frame_count = 0

                for img_path in video_path:
                    cap = cv2.VideoCapture(img_path)
                    frame_count = 0

                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            break
                        
                        ret, _, _, _ = self.Board.findCorners(img=frame)
                        if ret:
                            frame_filename = os.path.join(output_dir, cam, f"{self.Board.type}_frame_{frame_count:03d}.png")
                            cv2.imwrite(frame_filename, frame)
                            valid_frame_count += 1
                        frame_count += 1

        cv2.destroyAllWindows()
        print(f"Saved {valid_frame_count} valid frames.")
        return
    
    def getFrameCount(name):
        name_no_ext = name.split('.')[0]
        frame_count = name_no_ext.split('_')[-1]
        return frame_count

    def saveStereoData(self, path, cams):
        # Create array of all stereo images with 'im_name, img_pts, cam_ timestamp'
        stereo_images = {'Name':[], 'Camera':[], 'Corners':[], 'Charuco_Corners':[], 'Ids':[]}
        path_corners = os.path.join(path, 'corners_found')

        # maximal number of images per camera pair
        Nmax = 50

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
                            Lret, Lcorners, Lcharuco_corners, Lcharuco_ids = self.Board.findCorners(im_name=os.path.join(path_cam1, name1))
                            Rret, Rcorners, Rcharuco_corners, Rcharuco_ids = self.Board.findCorners(im_name=os.path.join(path_cam2, name2))

                            common_ids = []
                            if self.Board.type == 'charuco' and Lret and Rret:
                                _, _, _, common_ids = self.Board.getObjectImagePoints(Lcharuco_corners, Lcharuco_ids, Rcharuco_corners, Rcharuco_ids)

                            if self.Board.type == 'checker' or len(common_ids) > 1:
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
                    
                        if im_saved == Nmax:
                            break
                
                print(cam1, cam2, im_saved)
            
        with open('stereo_data.pkl', 'wb') as f:
            pickle.dump(stereo_images, f)

        return


    def StereoCalibration(self, leftparams, rightparams, Left_corners, Left_ids, Right_corners, Right_ids):
        StereoParams = {}
        
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

            for charuco_corners_l, charuco_ids_l, charuco_corners_r, charuco_ids_r in zip(Left_corners, Left_ids, Right_corners, Right_ids):
                # Find common detected corners (based on Charuco IDs)
                obj_pts, img_pts_l, img_pts_r, common_ids = self.Board.getObjectImagePoints(charuco_corners_l, charuco_ids_l, charuco_corners_r, charuco_ids_r)
                
                obj_points.append(obj_pts)
                img_points_left.append(img_pts_l)
                img_points_right.append(img_pts_r)

        elif self.Board.type == 'checker':
            obj_points = []
            for i in range(len(Left_corners)):
                obj_points.append(self.Board.objpoints)

        if len(obj_points) > 1:
            (ret, K1, D1, K2, D2, R, t, E, F) = cv2.stereoCalibrate(obj_points, img_points_left, img_points_right, k1, d1, k2, d2, self.image_size, criteria=criteria, flags=flags)
            
            T = np.vstack((np.hstack((R,t)),np.array([0,0,0,1])))
            
            StereoParams['Transformation'] = T
            StereoParams['Essential'] = E
            StereoParams['Fundamental'] = F
            StereoParams['MeanError'] = ret
        return StereoParams


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


    def var_name(var, scope=globals()):
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
    def project_points(points_3d, projMat, K, D):
        rvec, _ = cv2.Rodrigues(projMat[0:3,0:3])
        projected_2d, _ = cv2.projectPoints(points_3d, rvec, projMat[0:3,3], K, D)
        return projected_2d.squeeze()

    # Compute RMSE
    def compute_rmse(self, original_pts, points_3d, projMat, K, D):
        projected_pts = self.project_points(points_3d, projMat, K, D)
        error = np.linalg.norm(original_pts - projected_pts, axis=1)  # Euclidean distance per point
        rmse = np.sqrt(np.mean(error**2))  # Compute RMSE
        return rmse