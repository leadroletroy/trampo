
import cv2
import numpy as np
import glob
import os

class Charucoboard():
    def __init__(self, checkerboard, square_size, marker_size, aruco_name):
        self.type = 'charuco'
    
        self.checkerboard = checkerboard
        self.square_size = square_size
        self.marker_size = marker_size
        self.aruco_name = aruco_name
        
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_name)
        self.charuco_board = cv2.aruco.CharucoBoard(checkerboard, 112, 86, self.aruco_dict)

    # Test Charuco Board creation
    def create_and_save_new_board(self):
        LENGTH_PX = 640
        MARGIN_PX = 20

        dictionary = cv2.aruco.getPredefinedDictionary(self.aruco_name)
        board = cv2.aruco.CharucoBoard(self.checkerboard, 112, 86, dictionary)
        size_ratio = self.checkerboard[1] / self.checkerboard[0]
        img = cv2.aruco.CharucoBoard.generateImage(board, (LENGTH_PX, int(LENGTH_PX*size_ratio)), marginSize=MARGIN_PX)
        cv2.imshow("img", img)
        cv2.waitKey(1)

    # Charuco-specific functions
    def findCorners(self, img=None, im_name=None):
        if im_name is not None:
            img = cv2.imread(im_name)
        
        if img is None:
            raise ValueError('No image nor image path specified')
        
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        params = cv2.aruco.DetectorParameters()
        corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=params)

        if ids is not None: # and len(ids) >= 4:
            _, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(corners, ids, gray, self.charuco_board)
            if charuco_corners is not None and len(charuco_corners) > 0:
                return True, corners, charuco_corners, charuco_ids
            
        return False, None, None, None

    def calibrate_intrinsics(self, calib_dir, cams, image_size, camera_matrix=None, dist_coeffs=None):
        intrinsics_extension = 'png'
        ret, C, S, D, K = [], [], [], [], []

        for cam in cams:
            image_files = sorted(glob.glob(os.path.join(calib_dir, cam, "*.png")))  # Adjust extension if needed
            if len(image_files) == 0:
                raise ValueError(f'The folder {os.path.join(calib_dir, cam)} does not contain images with .{intrinsics_extension}')
            
            # Data storage
            imgpoints = []  # Detected charuco corners
            charuco_ids = []  # Corresponding IDs
            image_size = None  # To store image resolution

            # Process each image
            for img_path in image_files:
                img = cv2.imread(img_path)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                if image_size is None:
                    image_size = gray.shape[::-1]  # Get image size from first frame

                # Detect Aruco markers
                corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict)

                if ids is not None:
                    # Refine detection by interpolating Charuco corners
                    _, charuco_corners, charuco_ids_found = cv2.aruco.interpolateCornersCharuco(
                        markerCorners=corners, markerIds=ids, image=gray, board=self.charuco_board)

                    if charuco_corners is not None and len(charuco_corners) > 6:  # Ensure enough points
                        imgpoints.append(charuco_corners)
                        charuco_ids.append(charuco_ids_found)
            
            # Ensure we have enough valid frames
            if len(imgpoints) < 10:
                raise ValueError(f"Not enough valid images for calibration of {cam}! Need at least 10.")

            # Convert to NumPy arrays
            imgpoints = [np.array(p, dtype=np.float32) for p in imgpoints]
            charuco_ids = [np.array(p, dtype=np.int32) for p in charuco_ids]

            # Run calibration
            ret_cam, mtx, dist, _, _ = cv2.aruco.calibrateCameraCharuco(
                charucoCorners=imgpoints,
                charucoIds=charuco_ids,
                board=self.charuco_board,
                imageSize=image_size,
                cameraMatrix=camera_matrix,  # Use initialized matrix
                distCoeffs=dist_coeffs,
                flags=cv2.CALIB_USE_INTRINSIC_GUESS+cv2.CALIB_USE_LU
            )

            # Save results
            h, w = map(np.float32, image_size)
            ret.append(ret_cam)
            C.append(cam)
            S.append([w, h])
            D.append(dist)
            K.append(mtx)

            # Print results
            print(f"\nCalibration completed for camera: {cam}")
            print(f'Error: {ret_cam:.4f}')
            print("Camera Matrix:\n", mtx)
            print("Distortion Coefficients:\n", dist)

        return ret, C, S, D, K


    def GenerateImagepoints(self, Left_Paths:list, Right_Paths:list):
        Left_corners, Right_corners = [], []
        Left_ids, Right_ids = [], []
        Left_Paths_copy, Right_Paths_copy = [], []
        for Lname, Rname in zip(Left_Paths, Right_Paths):

            _, Lcharuco_corners, Lcharuco_ids = self.findCorners(im_name=Lname)
            _, Rcharuco_corners, Rcharuco_ids = self.findCorners(im_name=Rname)

            if Lcharuco_corners is not None and Rcharuco_corners is not None:
                Left_corners.append(Lcharuco_corners)
                Left_ids.append(Lcharuco_ids)

                Right_corners.append(Rcharuco_corners)
                Right_ids.append(Rcharuco_ids)

                Left_Paths_copy.append(Lname)
                Right_Paths_copy.append(Rname)

        return Left_corners, Left_ids, Left_Paths_copy, Right_corners, Right_ids, Right_Paths_copy


    def getObjectImagePoints(self, charuco_corners_l, charuco_ids_l, charuco_corners_r, charuco_ids_r):
        common_ids = np.intersect1d(charuco_ids_l, charuco_ids_r, assume_unique=True)
        if len(common_ids) > 0:
            obj_pts = self.charuco_board.getChessboardCorners()[common_ids.flatten()]
            img_pts_l = charuco_corners_l[np.isin(charuco_ids_l, common_ids)].reshape(-1, 1, 2)
            img_pts_r = charuco_corners_r[np.isin(charuco_ids_r, common_ids)].reshape(-1, 1, 2)
            
            return obj_pts, img_pts_l, img_pts_r, common_ids
        
        return None, None, None, None
    
