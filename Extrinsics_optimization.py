# Extrinsics_optimization
import numpy as np
import cv2
import pickle
import pyswarms as ps
from pyswarms.utils.plotters import plot_cost_history
import matplotlib.pyplot as plt

time_threshold = 15

with open('stereo_data_final.pkl', 'rb') as f:
    stereo_data = pickle.load(f)

K = np.load('Intrinsics_K_final.npz')['arr_0']
D = np.load('Intrinsics_D_final.npz')['arr_0']

projMat = #initial projection matrices


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


# Optimization function : loop on all the stereo images in stereo_data
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

        RMSE = []
        for i in range(0, len(stereo_data['Camera']) - 1, 2):
            j = i+1 # stereo image is the next one

            pts1_im = stereo_data['Img_pts'][i].squeeze()
            pts2_im = stereo_data['Img_pts'][j].squeeze()

            c1 = int(stereo_data['Camera'][i]) - 1
            c2 = int(stereo_data['Camera'][j]) - 1

            t1 = float(stereo_data['Time'][i])
            t2 = float(stereo_data['Time'][j])
            assert abs(t1-t2) < time_threshold, f'Images are not matching: {t1:.4f} and {t2:.4f}'

            undist_pts1 = cv2.undistortPoints(pts1_im, K[c1], D[c1]).reshape(-1, 2).T  # Shape (2, N)
            undist_pts2 = cv2.undistortPoints(pts2_im, K[c2], D[c2]).reshape(-1, 2).T  # Shape (2, N)

            # Perform triangulation
            pts_4d = cv2.triangulatePoints(projMat[c1], projMat[c2], undist_pts1, undist_pts2)

            # Convert from homogeneous coordinates
            points_3d = pts_4d[:3, :] / pts_4d[3, :]  # Shape (3, N)
            points_3d = points_3d.T  # Shape (N, 3)

            # Compute RMSE for both cameras
            rmse1 = compute_rmse(pts1_im, points_3d, projMat[c1], K[c1], D[c1])
            rmse2 = compute_rmse(pts2_im, points_3d, projMat[c2], K[c2], D[c2])

            RMSE.append(rmse1)
            RMSE.append(rmse2)

        errors[n] = np.mean(RMSE) #+ np.max(RMSE) OPTIONAL: add max error to cost function

    return errors


### PSO hyperparameters ###
num_cameras = 5
num_params_per_cam = 6
total_params = num_cameras * num_params_per_cam
n_particles = 60  # Number of particles
options = {'c1': 1.5, 'c2': 1.5, 'w': 0.7}  # PSO Hyperparameters

### Initial parameters ### (OPTIONAL)
init_params = []
for mat in projMat[1:]: # ignore cam1 as it is the global reference
    rvec, _ = cv2.Rodrigues(mat[0:3,0:3])
    rvec = rvec.squeeze()
    params = [rvec[0], rvec[1], rvec[2], mat[0,3], mat[1,3], mat[2,3]]
    init_params.extend(params)

init_params = np.array(init_params, dtype=np.float64)
init_pos = np.tile(init_params, (n_particles,1))
print(init_params.shape)
print(init_pos.shape)

# Define parameter bounds
lower_bounds = [-2*np.pi, -2*np.pi, -2*np.pi, -3000, -3000, -3000]
upper_bounds = [2*np.pi, 2*np.pi, 2*np.pi, 3000, 3000, 3000]

param_bounds = (np.tile(lower_bounds, num_cameras), np.tile(upper_bounds, num_cameras))
print(param_bounds[0].shape)

print(np.all(param_bounds[0] <= init_pos[0]))
print(np.all(init_pos[0] <= param_bounds[1]))


# Run PSO to optimize calibration parameters
optimizer = ps.single.GlobalBestPSO(n_particles=n_particles, dimensions=total_params, options=options, bounds=param_bounds, init_pos=init_pos)
best_error, best_params = optimizer.optimize(fun, iters=500)

# Extract optimized parameters for each camera
optimized_params = np.split(best_params, num_cameras)
extrinsics = []

for cam_idx in range(num_cameras):
    r1, r2, r3, t1, t2, t3 = optimized_params[cam_idx]
    rvec = np.array([r1,r2,r3], dtype=np.float64)
    R, _ = cv2.Rodrigues(rvec)
    t = np.array([t1,t2,t3], dtype=np.float64).reshape((3,1))
    optimized_camera_matrix = np.hstack((R,t))

    extrinsics.append(optimized_camera_matrix)
    print(f"Camera {cam_idx+1} Optimized Parameters:")
    print("Optimized Camera Matrix:\n", optimized_camera_matrix)

print(f"Best Reprojection Error: {best_error} pixels")

np.savez('Extrinsics_optimized.npz', extrinsics)

plot_cost_history(cost_history=optimizer.cost_history)
plt.savefig('Figure_1_pso_cost.png')
plt.show()
