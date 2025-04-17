
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
        self.charuco_board = cv2.aruco.CharucoBoard(checkerboard, square_size, marker_size, self.aruco_dict)
        
    # Test Charuco Board creation
    def create_and_save_new_board(self):
        LENGTH_PX = 1080
        MARGIN_PX = 40

        dictionary = cv2.aruco.getPredefinedDictionary(self.aruco_name)
        board = cv2.aruco.CharucoBoard(self.checkerboard, self.square_size, self.marker_size, dictionary)
        size_ratio = self.checkerboard[1] / self.checkerboard[0]
        img = cv2.aruco.CharucoBoard.generateImage(board, (LENGTH_PX, int(LENGTH_PX*size_ratio)), marginSize=MARGIN_PX)
        cv2.imshow("img", img)
        cv2.waitKey(1000)
        cv2.imwrite('board_big.jpg', img)
        cv2.destroyAllWindows()

    # Charuco-specific functions
    def findCorners(self, img=None, im_name=None, filter=False):
        if im_name is not None:
            img = cv2.imread(im_name)
        
        if img is None:
            raise ValueError('No image nor image path specified')
        
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        params = cv2.aruco.DetectorParameters()
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        params.minDistanceToBorder = 5

        markerCorners, markerIds, _  = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=params)

        if markerCorners is not None and markerIds is not None:
            # Interpolate ChArUco corners
            retval, charucoCorners, charucoIds = cv2.aruco.interpolateCornersCharuco(markerCorners, markerIds, gray, self.charuco_board)

            if retval and len(charucoCorners) > 6:
                
                # Filtrage
                if filter:
                    filtered_ints, filtered_points = [], []
                    exclude = [0, 8, 16]
                    for i, p in zip(charucoIds, charucoCorners):
                        if i not in exclude:
                            filtered_ints.append(i)
                            filtered_points.append(p)
                    charucoIds = np.array(filtered_ints, dtype=np.int32)
                    charucoCorners = np.array(filtered_points, dtype=np.float32)
                
                charucoIds = np.array(charucoIds, dtype=np.int32)
                charucoCorners = np.array(charucoCorners, dtype=np.float32)

                if len(charucoCorners) > 6:
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
                
            # Data storage
            marker_corners = []
            marker_ids = []
            marker_counter = []
            charuco_corners = []  # Detected charuco corners
            charuco_ids = []  # Corresponding IDs

            # Process each image
            for img_path in image_files:

                if image_size is None:
                    image_size = cv2.imread(img_path)[:,:,-1].shape

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
                continue
                
            #charuco_corners = [np.array(p, dtype=np.float32) for p in charuco_corners]
            #charuco_ids = [np.array(p, dtype=np.int32) for p in charuco_ids]

            # Run calibration
            ret_cam, mtx, dist, _, _ = cv2.aruco.calibrateCameraCharuco(
                charucoCorners=charuco_corners,
                charucoIds=charuco_ids,
                board=self.charuco_board,
                imageSize=image_size,
                cameraMatrix=camera_matrix,  # Use initialized matrix
                distCoeffs=dist_coeffs,
                flags=flags
            )

            """ marker_corners = np.concatenate(marker_corners)
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
    
