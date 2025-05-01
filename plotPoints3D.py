import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

points_array_m = np.load(r'C:\Users\Lea\Documents\Trampo\code\trampo\Points3D.npz')['arr_0']

ax.scatter(points_array_m[:, 0], points_array_m[:, 1], points_array_m[:, 2],
           c='red', marker='o', s=30)

ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')
ax.set_title('3D triangulated points covered')
plt.show()
