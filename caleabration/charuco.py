
import cv2
import numpy as np
import glob
import os

class Charucoboard():
    def __init__(self, checkerboard, square_size, marker_size, aruco_name, legacy):
        self.type = 'charuco'
    
        self.checkerboard = checkerboard
        self.square_size = square_size
        self.marker_size = marker_size
        self.aruco_name = aruco_name
        
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_name)
        self.charuco_board = cv2.aruco.CharucoBoard(checkerboard, square_size, marker_size, self.aruco_dict)
        
        self.charuco_board.setLegacyPattern(legacy)

    # Test Charuco Board creation
    def create_and_save_new_board(self):
        LENGTH_PX = 1080
        MARGIN_PX = 40

        size_ratio = self.checkerboard[1] / self.checkerboard[0]
        img = cv2.aruco.CharucoBoard.generateImage(self.charuco_board, (LENGTH_PX, int(LENGTH_PX*size_ratio)), marginSize=MARGIN_PX)
        cv2.imshow("img", img)
        cv2.waitKey(1000)
        cv2.imwrite('board_big.jpg', img)
        cv2.destroyAllWindows()

    # Charuco-specific functions
    def findCorners(self, img=None, im_name=None):
        if im_name is not None:
            img = cv2.imread(im_name)
        
        if img is None:
            raise ValueError('No image nor image path specified')
        
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        charucoParams = cv2.aruco.CharucoParameters()
        detectorParams = cv2.aruco.DetectorParameters()
        charuco_detector = cv2.aruco.CharucoDetector(self.charuco_board)

        charucoCorners, charucoIds, markerCorners, markerIds = charuco_detector.detectBoard(gray)

        """ if markerCorners is not None and markerIds is not None:
            # Interpolate ChArUco corners
            retval, charucoCorners, charucoIds = cv2.aruco.interpolateCornersCharuco(markerCorners, markerIds, gray, self.charuco_board)
        """
        if charucoCorners is not None and len(charucoCorners) >= 6:
            
            charucoIds = np.array(charucoIds, dtype=np.int32)
            charucoCorners = np.array(charucoCorners, dtype=np.float32)

            if len(charucoCorners) >= 6:
                return True, markerCorners, markerIds, charucoCorners, charucoIds
            
        return False, None, None, None, None

    def calibrate_intrinsics(self, calib_dir, cams, image_size=None, camera_matrix=None, dist_coeffs=None):
        intrinsics_extension = 'png'
        ret, C, S, D, K = [], [], [], [], []

        flags = cv2.CALIB_USE_LU
        if camera_matrix is not None:
            flags += cv2.CALIB_USE_INTRINSIC_GUESS

        for cam in cams:
            image_files = sorted(glob.glob(os.path.join(calib_dir, cam, "*.png")))  # Adjust extension if needed
            if len(image_files) == 0:
                print(f"Not enough valid images for calibration of {cam}!")
                continue
                
            """ # Data storage
            marker_corners = []
            marker_ids = []
            marker_counter = []
            charuco_corners = []  # Detected charuco corners
            charuco_ids = []  # Corresponding IDs

            # Process each image
            for img_path in image_files:

                if image_size is None:
                    image_size = cv2.imread(img_path)[:,:,0].shape

                ret_c, markerCorners, markerIds, charucoCorners, charucoIds = self.findCorners(im_name = img_path)

                if ret_c:
                    #marker_corners.append(np.array(markerCorners, dtype=np.float32))
                    #marker_ids.append(np.array(markerIds, dtype=np.int32))
                    #marker_counter.append(len(markerCorners))
                    charuco_corners.append(charucoCorners)
                    charuco_ids.append(charucoIds)
            
            # Ensure we have enough valid frames
            if len(charuco_corners) < 1:
                print(f"Not enough valid images for calibration of {cam}!")
                continue """
            
            # Listes pour stocker les points d'image et d'objet
            all_image_points = []
            all_object_points = []

            # Parcourir les images pour détecter les coins ChArUco
            for image_path in image_files:
                image = cv2.imread(image_path)
                _, _, _, charucoCorners, charucoIds = self.findCorners(img=image)

                if charucoCorners is not None and charucoIds is not None and len(charucoIds) >= 4:
                    obj_points, img_points = self.charuco_board.matchImagePoints(charucoCorners, charucoIds)

                    if obj_points is not None and img_points is not None and len(obj_points) >= 4:
                        obj_points = np.asarray(obj_points, dtype=np.float32).reshape(-1, 1, 3)
                        img_points = np.asarray(img_points, dtype=np.float32).reshape(-1, 1, 2)

                        all_object_points.append(obj_points)
                        all_image_points.append(img_points)

            if len(all_object_points) < 4:
                print(f"Not enough valid images for calibration of {cam}!")
                continue

            print("\nNombre d'images retenues :", len(all_object_points))
            image_size = (image.shape[1], image.shape[0])
            
            ret_cam, mtx, dist, _, _ = cv2.calibrateCamera(all_object_points, all_image_points, image_size, camera_matrix, dist_coeffs,flags=flags)
            
            """ # Run calibration
            ret_cam, mtx, dist, _, _ = cv2.aruco.calibrateCameraCharuco(
                charucoCorners=charuco_corners,
                charucoIds=charuco_ids,
                board=self.charuco_board,
                imageSize=image_size,
                cameraMatrix=camera_matrix,  # Use initialized matrix
                distCoeffs=dist_coeffs,
                flags=flags
            )
            marker_corners = np.concatenate(marker_corners)
            marker_ids_concat = np.concatenate(marker_ids)
            marker_counter = np.array(marker_counter)

            ret_cam, mtx, dist, _, _ = cv2.aruco.calibrateCameraAruco(
                marker_corners, marker_ids_concat, marker_counter,
                self.charuco_board, image_size,
                cameraMatrix=camera_matrix,  # Use initialized matrix
                distCoeffs=dist_coeffs,
                flags=flags
            ) """

            # Save results
            h, w = map(np.int32, image_size)
            ret.append(ret_cam)
            C.append(cam)
            S.append([w, h])
            D.append(dist)
            K.append(mtx)

            # Print results
            print(f"Calibration completed for camera: {cam}")
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
        charuco_ids_l = np.array(charuco_ids_l).squeeze()
        charuco_ids_r = np.array(charuco_ids_r).squeeze()
        common_ids = np.intersect1d(charuco_ids_l, charuco_ids_r, assume_unique=True)
        
        if len(common_ids) > 0:
            obj_pts = self.charuco_board.getChessboardCorners()[common_ids.flatten()]

            charuco_corners_l = np.array(charuco_corners_l).squeeze()
            charuco_corners_r = np.array(charuco_corners_r).squeeze()

            img_pts_l = charuco_corners_l[np.isin(charuco_ids_l, common_ids)].reshape(-1, 1, 2)
            img_pts_r = charuco_corners_r[np.isin(charuco_ids_r, common_ids)].reshape(-1, 1, 2)
            
            return obj_pts, img_pts_l, img_pts_r, common_ids
        
        return None, None, None, None
    
