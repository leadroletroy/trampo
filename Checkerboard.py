
import os
import cv2
import math
import numpy as np
import pandas as pd
import re
import glob
from pose2sim_trampo.common import euclidean_distance


class Checkerboard():
    def __init__(self, checkerboard, square_size):
        self.type = 'checker' 
        
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
    def findCorners(self, img = None, im_name = None, subpix = True):
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001) # stop refining after 30 iterations or if error less than 0.001px

        if im_name is not None:
            img = cv2.imread(im_name)
    
        if img is None:
            raise ValueError('No image nor image path specified')
        
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # Find corners
        ret, corners = cv2.findChessboardCorners(gray, self.checkerboard, None)
        # If corners are found, refine corners
        if ret == True and subpix == True: 
            imgp = cv2.cornerSubPix(gray, corners, (5, 5), (-1,-1), criteria)
            return ret, imgp, None, None
        else:
            return ret, corners, None, None
    
    
    def calibrate_intrinsics(self, calib_dir, image_size, camera_matrix, dist_coeffs=np.zeros(5, dtype=np.float32)):
        intrinsics_extension = 'png'

        ret, C, S, D, K = [], [], [], [], []
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
            ret_cam, mtx, dist, _, _ = cv2.calibrateCamera(objpoints, imgpoints, image_size, camera_matrix[i], dist_coeffs[i], flags=(cv2.CALIB_USE_INTRINSIC_GUESS+cv2.CALIB_USE_LU))
            h, w = [np.float32(i) for i in img.shape[:-1]]
            ret.append(ret_cam)
            C.append(cam)
            S.append([w, h])
            D.append(dist)
            K.append(mtx)

        return ret, C, S, D, K


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

