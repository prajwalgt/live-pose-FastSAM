import numpy as np
from scipy.spatial.transform import Rotation as R

np.set_printoptions(
    precision=3, # Display numbers with 3 decimal places
    suppress=True, # Suppress scientific notation
    linewidth=150 # Set a wider line width
)

# camera position in torso frame
translation = np.array([0.06, 0.0, 0.45])

# rpy
euler_xyz = [0, -0.8, -1.57]  

rotation_matrix = R.from_euler('xyz', euler_xyz).as_matrix()

T_torso_from_camera = np.eye(4)
T_torso_from_camera[:3, :3] = rotation_matrix
T_torso_from_camera[:3, 3] = translation

T_torso_camera_new = np.array([[1.0000, 0.000, 0.0000, 0.0],
                           [0.0000, -0.743, 0.669, 0.047],
                           [0.0000, -0.669, -0.743, 0.462],
                           [0.0000, 0.000, 0.0000, 1.0000]])

T_torso_camera_withshiftedy = np.array([[0.0000, -0.743, 0.6691, 0.0],
                                        [-1.000, 0.0000, 0.0000, 0.047],
                                        [0.0000, -0.669, -0.743, 0.462],
                                        [0.0000, 0.000, 0.0000, 1.0000]])

# print("Transformation matrix (Torso <-- Camera):")
print(T_torso_from_camera)

print("Transformation matrix (Camera <-- Torso) NEW:")

# print(T_torso_camera_new)

print(np.linalg.inv(T_torso_camera_withshiftedy))
print(np.linalg.inv(T_torso_camera_new))
breakpoint()
# print(T_torso_camera_new @ np.linalg.inv(T_torso_camera_new))


# print(np.linalg.inv(T_torso_from_camera))
# breakpoint()

point_camera = np.array([0.0, 0.0, 0.0, 1])  # homogeneous coordinates [I have put this just as an example, enter actual points here]

T_torso_camera_new_inv = np.linalg.inv(T_torso_camera_new)

# quaternion_in_camera = R.from_matrix(rotation_matrix_in_camera).as_quat() 

point_torso = T_torso_camera_new @ point_camera

print("Point in Camera Frame:")
print(point_camera[:3])
print("Point in Torso Frame:")
print(point_torso[:3])  
