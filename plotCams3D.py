import matplotlib.pyplot as plt
import numpy as np


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


worldTcam_all = np.load('WorldTCam_opt.npz')['arr_0']
cams = ['M11139', 'M11140', 'M11141', 'M11458', 'M11459', 'M11461', 'M11462', 'M11463']

# Plot frames
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for i, T in enumerate(worldTcam_all):
    T = np.vstack((T, np.array([0,0,0,1])))
    plot_frame(ax, T, label=cams[i])

# Labels
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
plt.title("3D Reference Frames")
plt.show()