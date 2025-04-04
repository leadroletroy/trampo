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
import re
import glob
from pose2sim_trampo.common import euclidean_distance

class calibration():
    def __init__(self, checkerboard, square_size):
        self.checkerboard = checkerboard
        self.square_size = square_size

        objpoints = np.zeros((checkerboard[0] * checkerboard[1], 3), np.float32)
        objpoints[:, :2] = np.mgrid[0:checkerboard[0], 0:checkerboard[1]].T.reshape(-1, 2)
        objpoints[:, :2] = objpoints[:, 0:2] * square_size
        self.objpoints = objpoints

        object_coords_3d = np.zeros((checkerboard[0] * checkerboard[1], 3), np.float32)
        object_coords_3d[:, :2] = np.mgrid[0:checkerboard[0], 0:checkerboard[1]].T.reshape(-1, 2)
        object_coords_3d[:, :2] = object_coords_3d[:, 0:2] * square_size
        self.object_coords_3d = object_coords_3d


    # findCorners from Pose2Sim
    def findCorners(self, img_path, subpix = True):
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001) # stop refining after 30 iterations or if error less than 0.001px
    
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # Find corners
        ret, corners = cv2.findChessboardCorners(gray, self.checkerboard, None)
        # If corners are found, refine corners
        if ret == True and subpix == True: 
            imgp = cv2.cornerSubPix(gray, corners, (5, 5), (-1,-1), criteria)
            return ret, img, imgp
        else:
            return ret, img, corners, None
    
    # To find images where we see the checkerboard, for individual cameras
    def save_images_checkerboard(self, path, dir, cam):
        for fname in os.listdir(os.path.join(dir, cam)):
            savepath = os.path.join(path, 'calibration', 'intrinsics', cam)
            if not os.path.isdir(savepath):
                os.makedirs(savepath)
            savename = dir[-4:] + '_' + cam + fname

            ret, img, _ = self.findCorners(os.path.join(dir, cam, fname), subpix = False)
            if ret == True:
                cv2.imwrite(os.path.join(savepath, savename), img)
            
        return
    
    def refine_image_selection(self, path, dir, cam, K, D, image_size:tuple, angle_threshold:float, coverage_threshold:tuple):
        image_area = image_size[0] * image_size[1]
        mask_covered = np.zeros(image_size, dtype=np.uint8)
        
        for fname in os.listdir(os.path.join(path, cam)):
            savepath = os.path.join(path, dir, cam)
            if not os.path.isdir(savepath):
                os.makedirs(savepath)
            
            ret, img, corners = self.findCorners(os.path.join(path, cam, fname), subpix = False)

            if ret == True:
                # Get angle between camera axis and image
                ret, rvec, _ = cv2.solvePnP(self.objpoints, corners, K, D)
                R, _ = cv2.Rodrigues(rvec)
                trace = np.trace(R)
                theta = np.arccos((trace - 1) / 2)  # Compute the angle in radians
                angle = np.degrees(theta)  # Convert to degrees
                if angle > 90:
                    angle = 180 - angle

                # Get proportion of the image covered by the checkerboard
                x_min, y_min = np.min(corners, axis=0)[0]
                x_max, y_max = np.max(corners, axis=0)[0]
                mask_covered[int(x_min):int(x_max), int(y_min):int(y_max)] = 1
                checkerboard_area = (x_max - x_min) * (y_max - y_min)
                coverage = checkerboard_area / image_area

                if coverage_threshold[0] < coverage < coverage_threshold[1] and angle < angle_threshold:
                    cv2.imwrite(os.path.join(savepath, fname), img)
        
        total_area_covered = np.sum(mask_covered) / image_area
            
        return total_area_covered
    
    def calibrate_intrinsics(self, calib_dir, image_size, camera_matrix, dist_coeffs=np.zeros(5, dtype=np.float32)):
        intrinsics_extension = 'png'

        ret, C, S, D, K, R, T = [], [], [], [], [], [], []
        intrinsics_cam_listdirs_names = next(os.walk(os.path.join(calib_dir)))[1]

        for i,cam in enumerate(intrinsics_cam_listdirs_names):
            # Prepare object points
            objpoints = [] # 3d points in world space
            imgpoints = [] # 2d points in image plane

            img_vid_files = glob.glob(os.path.join(calib_dir, cam, f'*.{intrinsics_extension}'))
            if len(img_vid_files) == 0:
                raise ValueError(f'The folder {os.path.join(calib_dir, cam)} does not exist or does not contain any files with extension .{intrinsics_extension}.')
            img_vid_files = sorted(img_vid_files, key=lambda c: [int(n) for n in re.findall(r'\d+', c)]) #sorting paths with numbers
            
            # find corners
            for img_path in img_vid_files:
                _, _, imgp_confirmed = self.findCorners(img_path)
                if isinstance(imgp_confirmed, np.ndarray):
                    imgpoints.append(imgp_confirmed)
                    objpoints.append(self.objpoints)

            # calculate intrinsics
            img = cv2.imread(str(img_path))
            objpoints = np.array(objpoints)
            ret_cam, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, image_size, camera_matrix[i], dist_coeffs[i], flags=(cv2.CALIB_USE_INTRINSIC_GUESS+cv2.CALIB_USE_LU))
            h, w = [np.float32(i) for i in img.shape[:-1]]
            ret.append(ret_cam)
            C.append(cam)
            S.append([w, h])
            D.append(dist)
            K.append(mtx)
            R.append([0.0, 0.0, 0.0])
            T.append([0.0, 0.0, 0.0])

        return ret, C, S, D, K, R, T


    # To keep only the images where the checkerboard is seen simultaneously by 2 cameras
    def retrieve_cali_info(self, path, user, cam):
        cali = pd.read_csv(os.path.join(path, f'cali{cam}.csv'), sep=';')
        list_of_column_names = list(cali.columns)

        if len(list_of_column_names) < 2:
            cali = pd.read_csv(os.path.join(path, f'cali{cam}.csv'), sep=',')
            list_of_column_names = list(cali.columns)

        imnames = [str(imname).split('/')[-1] for imname in cali['pathColor'].tolist()]
        users = cali['user'].tolist()
        timestamps = cali['timestamp'].tolist()

        indices = []
        for i, u in enumerate(users):
            if int(u) == int(user[1:]):
                indices.append(i)

        info = np.array([imnames, timestamps])
        info = info[:,indices]

        return info


    def find_matching_images(self, path, user, cams, time_threshold, dir_stereo, path_intrinsics):
        for cam1 in cams:
            # reading csv file 
            cali1 = self.retrieve_cali_info(path, user, cam1[-1])
            # print(os.listdir(os.path.join(path_intrinsics, cam1)))

            for cam2 in cams[cams.index(cam1)+1:]:
                path_stereo = os.path.join(dir_stereo, f'{cam1}_{cam2}')
                os.makedirs(path_stereo, exist_ok=True)

                cali2 = self.retrieve_cali_info(path, user, cam2[-1])

                for i,t1 in enumerate(cali1[1,:]):
                    # print(cali1[0,i])
                    
                    for j,t2 in enumerate(cali2[1,:]):
                        t1 = float(t1)
                        t2 = float(t2)
                        if abs(t1-t2) < time_threshold:
                            # Lret, Limg, _ = self.findCorners(os.path.join(path, user, cam1, cali1[0,i]), subpix=False)
                            # Rret, Rimg, _ = self.findCorners(os.path.join(path, user, cam2, cali2[0,j]), subpix=False)
                            # if Lret == True and Rret == True:

                            name1 = user + '_' + cam1 + cali1[0,i]
                            name2 = user + '_' + cam2 + cali2[0,j]
                            
                            im1In = name1 in list(os.listdir(os.path.join(path_intrinsics, cam1)))
                            im2In = name2 in list(os.listdir(os.path.join(path_intrinsics, cam2)))

                            if im1In and im2In:
                            
                            #if cali1[0,i] in os.listdir(os.path.join(path, 'calibration', 'intrinsics', cam1)) and cali2[0,j] in os.listdir(os.path.join(path, 'calibration', 'intrinsics', cam2)):
                                #print(cali1[0,i], cali2[0,j])

                                Limg = cv2.imread(os.path.join(path_intrinsics, cam1, name1))
                                cv2.imwrite(os.path.join(path_stereo, f'{i}_{j}_{cam1}_{name1}'), Limg)

                                Rimg = cv2.imread(os.path.join(path_intrinsics, cam2, name2))
                                cv2.imwrite(os.path.join(path_stereo, f'{i}_{j}_{cam2}_{name2}'), Rimg)
        return


    # STEREO CALIBRATION inspired from https://www.kaggle.com/code/danielwe14/stereocamera-calibration-with-opencv#Camera-Calibration 
    def GenerateImagepoints(self, Left_Paths:list, Right_Paths:list, corner_nb):
        Left_imgpoints, Right_imgpoints = [], []
        Left_Paths_copy, Right_Paths_copy = [], []
        for Lname, Rname in zip(Left_Paths, Right_Paths):

            Lret, Limg, Limgp = self.findCorners(Lname)
            Rret, Rimg, Rimgp = self.findCorners(Rname)

            if Lret and Rret:
                Left_imgpoints.append(Limgp)
                Right_imgpoints.append(Rimgp)

                Left_Paths_copy.append(Lname)
                Right_Paths_copy.append(Rname)
        
        # print(len(Left_Paths_copy), len(Left_imgpoints), len(Right_Paths_copy), len(Right_imgpoints))

        return Left_imgpoints, Left_Paths_copy, Right_imgpoints, Right_Paths_copy

    def CalibrateCamera(self, Paths, imgpoints, K = None, D = None):
        CameraParams = {}
        
        ### INTRINSICS ###
        img = cv2.imread(Paths[0])

        # INTRINSICS already computed
        if K is not None and D is not None:
            mtx, dist = K, D
        
        # Compute INTRINSICS    
        else:
            img = cv2.imread(Paths[0])
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            g = gray.shape[1::-1]
            
            flags = (cv2.CALIB_FIX_K3+cv2.CALIB_USE_LU)
            
            objp = []
            for i in range(len(Paths)):
                objp.append(self.objpoints)
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objp, imgpoints, g, None, None, flags=flags)
        
        h, w = [np.uint16(i) for i in img.shape[:-1]]
        newmtx, roi = cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))

        if np.sum(roi) == 0:
            roi = (0,0,w-1,h-1)

        ### EXTRINSICS ###
        ret, rvecs, tvecs, R, T = [], [], [], [], []

        # Compute EXTRINSICS
        for im in Paths:
            imgpoints = self.findCorners(im)[-1]
            imgpoints = np.squeeze(imgpoints)
            obj_coords = np.array(self.object_coords_3d, dtype=np.float32)

            _, r, t = cv2.solvePnP(obj_coords, imgpoints, mtx, dist)
            r = r.flatten()
            #t /= 1000
            rvecs.append(r)
            tvecs.append(t.flatten())
            rmat, _ = cv2.Rodrigues(r)
            R.append(rmat)
            T.append(np.vstack((np.hstack((rmat,t)),np.array([0,0,0,1]))))

            proj_obj = np.squeeze(cv2.projectPoints(obj_coords,r,t,mtx,dist)[0])
            imgp_to_objreproj_dist = [euclidean_distance(proj_obj[n], imgpoints[n]) for n in range(len(proj_obj))]
            rms_px = np.sqrt(np.sum([d**2 for d in imgp_to_objreproj_dist]))
            ret.append(rms_px)
        
        # Save parameters
        CameraParams['Errors'] = ret
        CameraParams['Intrinsic'] = mtx
        CameraParams['Distortion'] = dist
        CameraParams['DistortionROI'] = roi
        CameraParams['DistortionIntrinsic'] = newmtx
        CameraParams['RotVektor'] = rvecs
        CameraParams['RotMatrix'] = R
        CameraParams['Extrinsics'] = T
        CameraParams['TransVektor'] = tvecs
        
        return CameraParams


    def CalculateErrors(self, params, imgpoints):
        imgp = np.array(imgpoints)
        imgp = np.squeeze(imgp)
        # imgp = imgp.reshape((imgp.shape[0], imgp.shape[1], imgp.shape[3]))
        objp = np.array(self.objpoints)
        K = np.array(params['Intrinsic'])
        D = np.array(params['Distortion'])
        R = np.array(params['RotVektor'])
        T = np.array(params['TransVektor'])
        N = imgp.shape[0]
        
        imgpNew = []
        for i in range(N):
            temp = np.squeeze(cv2.projectPoints(objp, R[i], T[i], K, D)[0])
            imgpNew.append(temp)
        imgpNew = np.array(imgpNew)
        
        print(imgp.shape, imgpNew.shape)
        # for i in range(N):
        #     err.append(imgp[i] - imgpNew[i])
        # err = np.array(err)
        
        def RMSE(imgp, imgp_reproj):
            # Compute squared differences
            squared_diff = (imgp - imgp_reproj) ** 2
            mse = np.mean(np.sum(squared_diff, axis=1))
            rmse = np.sqrt(mse)
            
            return rmse
            # return np.sqrt(np.mean(np.sum(err**2, axis=1)))

        rmse = RMSE(imgp, imgpNew)
        
        # print(rmse)
        # errall = np.copy(err[0])
        # rmsePerView = [RMSE(err[0])]
        # for i in range(1,N):
        #     errall = np.vstack((errall, err[i]))
        #     rmsePerView.append(RMSE(err[i]))

        # rmseAll = RMSE(errall)
        return rmse # rmsePerView, rmseAll

    def StereoCalibration(self, leftparams, rightparams, imgpL, imgpR, Left_Paths):
        StereoParams = {}
        
        k1 = leftparams['Intrinsic']
        d1 = leftparams['Distortion']
        k2 = rightparams['Intrinsic']
        d2 = rightparams['Distortion']
        gray = cv2.imread(Left_Paths[0], 0)
        g = gray.shape[::-1]
        
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)
        flags = 0
        flags |= cv2.CALIB_FIX_INTRINSIC
        
        objp = []
        for i in range(len(imgpL)):
            objp.append(self.objpoints)
        
        (ret, K1, D1, K2, D2, R, t, E, F) = cv2.stereoCalibrate(objp, imgpL, imgpR, k1, d1, k2, d2, g, criteria=criteria, flags=flags)
        
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
