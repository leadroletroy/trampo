
import numpy as np
import cv2
import random

# Parameters
trampoline_width = 5.0 # meters
trampoline_length = 3.0 # meters
trampoline_height = 1.0 # meters

# Camera intrinsic parameters (assumed, normally obtained from calibration)
focal_length = 800
sensor_width = 0.036 # meters (typical for a DSLR sensor)
sensor_height = 0.024
image_width = 1920
image_height = 1080

# Define a function to calculate the Field of View (FoV) for a given camera
def calculate_fov(focal_length, sensor_width, sensor_height):
    fov_x = 2 * np.arctan(sensor_width / (2 * focal_length))
    fov_y = 2 * np.arctan(sensor_height / (2 * focal_length))
    return np.degrees(fov_x), np.degrees(fov_y)

# Example of camera intrinsic matrix (3x3 matrix)
camera_matrix = np.array([[focal_length, 0, image_width / 2],
                          [0, focal_length, image_height / 2],
                          [0, 0, 1]])

# Initial camera parameters and locations
num_cameras = 4 # Number of cameras
camera_positions = np.array([
    [3, -2, 1], # Camera 1 position (x, y, z)
    [-3, -2, 1], # Camera 2 position
    [3, 2, 1], # Camera 3 position
    [-3, 2, 1] # Camera 4 position
], dtype=np.float32)
camera_orientations = np.zeros((num_cameras, 3)) # Assume facing forward for simplicity

# Define the coverage area
coverage_points = 100 # Number of points to check for coverage on trampoline
trampoline_points = np.array([
    [random.uniform(-trampoline_width/2, trampoline_width/2), 
     random.uniform(-trampoline_length/2, trampoline_length/2), 
     trampoline_height]
    for _ in range(coverage_points)
])

# Function to project trampoline points into camera space
def project_points(points, camera_matrix, rvec, tvec):
    # Convert points from 3D world space to camera space
    projected_points, _ = cv2.projectPoints(points, rvec, tvec, camera_matrix, None)
    return projected_points.reshape(-1, 2)

# Calculate a score for each camera configuration based on coverage
def coverage_score(camera_positions, camera_orientations):
    score = 0
    for cam_pos, orientation in zip(camera_positions, camera_orientations):
        #rvec = np.array([orientation[0], orientation[1], orientation[2]])
        rvec = orientation
        tvec = cam_pos.reshape((3, 1))

        visible_points = 0

        for point in trampoline_points:
            projected_point = project_points(np.array([point]), camera_matrix, rvec, tvec)
            x, y = projected_point[0]
            if 0 <= x < image_width and 0 <= y < image_height: # Check if point is within image bounds
                visible_points += 1

        score += visible_points
    return score

# Simulated Annealing for optimization
def optimize_camera_placement(iterations=1000, temperature=10):
    best_positions = camera_positions.copy()
    best_score = coverage_score(best_positions, camera_orientations)

    for i in range(iterations):
        # Create a small random shift in one camera's position
        new_positions = best_positions.copy()
        cam_index = random.randint(0, num_cameras - 1)
        new_positions[cam_index] += np.random.normal(0, 0.2, 3) # Random movement

        # Calculate the new score
        new_score = coverage_score(new_positions, camera_orientations)
        delta_score = new_score - best_score

        # Decide to accept new configuration based on simulated annealing
        if delta_score > 0 or np.exp(delta_score / temperature) > random.random():
            best_positions = new_positions
            best_score = new_score
            temperature *= 0.99 # Cool down

        if i % 100 == 0: # Print progress every 100 iterations
            print(f"Iteration {i}, Best Score: {best_score}")

    return best_positions, best_score

# Run optimization
best_positions, best_score = optimize_camera_placement()
print("Optimal Camera Positions:")
print(best_positions)
print("Coverage Score:", best_score)