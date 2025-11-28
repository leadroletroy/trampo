#copy Pose2Sim calibration file writing to go from matrices to Calib.toml
import matplotlib.pyplot as plt
import numpy as np
import json
import cv2
import os

calib_file = json.load(open('/mnt/D494C4CF94C4B4F0/DATASETS_HPE/AthletePose3D/cam_param.json'))

C, K, D, R, rot, T = [], [], [], [], [], []

H, W = 1920, 1088
S = [[H,W] for _ in range(len(calib_file.keys()))]

for cam in calib_file.keys():

    if 'fs' in cam:
        C.append(cam)
        K.append(np.array(calib_file[cam]['affine_intrinsics_matrix']))
        D.append(np.array(calib_file[cam]['distortion']))
        Rmat = np.array(calib_file[cam]['extrinsic_matrix'])
        R.append(Rmat)
        rot.append(cv2.Rodrigues(np.array(Rmat))[0].flatten())
        T.append(np.array(calib_file[cam]['xyz']) / 1000) # to meters

#toml_write('Calibration/Calib_AP3D.toml', C, S, D, K, rot, T)

def plot_frame(ax, T, label="Frame", length=500):
    """Plots a 3D reference frame with arrows for X, Y, and Z axes."""
    origin = T[:3, 3]  # Extract translation (position)
    R = T[:3, :3]      # Extract rotation matrix (orientation)

    # Define the x, y, z unit vectors from the rotation matrix
    x_axis = R[:, 0] * length  # X-direction
    y_axis = R[:, 1] * length  # Y-direction
    z_axis = R[:, 2] * length  # Z-direction

    # Plot arrows for the axes
    ax.quiver(*origin, *x_axis, color='r', arrow_length_ratio=0.3)  # X-axis (Red)
    ax.quiver(*origin, *y_axis, color='g', arrow_length_ratio=0.3)  # Y-axis (Green)
    ax.quiver(*origin, *z_axis, color='b', arrow_length_ratio=0.3)  # Z-axis (Blue)

    # Label the frame
    ax.text(*origin, label, fontsize=9, color='black')

    return

WorldTCam_opt = []
for i in range(len(C)):
    RT = np.eye(4)
    RT[0:3, 0:3] = R[i]
    RT[0:3, 3] = T[i]

    WorldTCam_opt.append(np.linalg.inv(RT))

# Plot frames
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111, projection='3d')

for i, T in enumerate(WorldTCam_opt):
    T = np.vstack((T, np.array([0,0,0,1])))
    plot_frame(ax, T, label=C[i], length=.5)

# Labels
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_xlim(-10, 10)
ax.set_ylim(-12, 6)
ax.set_zlim(-10, 10)
ax.view_init(elev=0, azim=0)
plt.title("3D Reference Frames")
plt.show()
plt.close()
