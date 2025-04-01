import pyrealsense2 as rs
from estimater import *
from FoundationPose.mask import *
import tkinter as tk
from tkinter import filedialog
from scipy.spatial.transform import Rotation as R

parser = argparse.ArgumentParser()
code_dir = os.path.dirname(os.path.realpath(__file__))
parser.add_argument('--est_refine_iter', type=int, default=4)
parser.add_argument('--track_refine_iter', type=int, default=2)
args = parser.parse_args()

set_logging_format()
set_seed(0)

root = tk.Tk()
root.withdraw()

# mesh_path = filedialog.askopenfilename() # PGT
mesh_path = '/home/prajwal/Documents/Xbox.obj'
if not mesh_path:
    print("No mesh file selected")
    exit(0)
# mask_file_path = create_mask() #PGT
mask_file_path = create_mask_with_fastsam()
# mask_file_path = './mask.png' #same mask doesn't work for new run
mesh = trimesh.load(mesh_path)
to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2,3)
# breakpoint()
scorer = ScorePredictor()
refiner = PoseRefinePredictor()
glctx = dr.RasterizeCudaContext()
est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh, scorer=scorer, refiner=refiner,glctx=glctx)
pipeline = rs.pipeline()
config = rs.config()
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
profile = pipeline.start(config)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
clipping_distance_in_meters = 1 #1 meter
clipping_distance = clipping_distance_in_meters / depth_scale
align_to = rs.stream.color
align = rs.align(align_to)
with open('transformed_points.txt', 'w') as f:
    f.write("# Transformed points from object tracking\n")

i = 0

mask = cv2.imread(mask_file_path, cv2.IMREAD_UNCHANGED)
cam_K = np.array([[615.37701416, 0., 313.68743896],
                   [0., 615.37701416, 259.01800537],
                   [0., 0., 1.]])
Estimating = True
time.sleep(3)

# camera position in torso frame
translation = np.array([0.06, 0.0, 0.45])

# rpy
euler_xyz = [0, -0.8, -1.57]  

rotation_matrix = R.from_euler('xyz', euler_xyz).as_matrix()

rotate_in_y_matrix = np.array([[-1, 0, 0],
                            [0, 1, 0],
                            [0, 0, -1]])

rotation_matrix = rotate_in_y_matrix @ rotation_matrix

T_torso_from_camera = np.eye(4)
T_torso_from_camera[:3, :3] = rotation_matrix
T_torso_from_camera[:3, 3] = translation



T_torso_camera_new = np.array([[1.0000, 0.000, 0.0000, 0.0],
                           [0.0000, -0.743, -0.669, 0.344],
                           [0.0000, 0.669, -0.743, 0.312],
                           [0.0000, 0.000, 0.0000, 1.0000]]) ## step one

# T_torso_camera_new = np.array([[0.0000, -1.000, -0.669, 0.344],
#                                [-0.743, 0.0000, 0.0000, 0.000],
#                                [-0.669, -0.669, -0.743, 0.312],
#                                [0.0000, 0.000, 0.0000, 1.0000]]) ## THis one is the last one developed, should be correct but has some z value issues, (10 cm error b/w left and right z points)

T_torso_camera_withshiftedy = np.array([[0.0000, -0.743, 0.6691, 0.0],
                                        [-1.000, 0.0000, 0.0000, 0.047],
                                        [0.0000, -0.669, -0.743, 0.462],
                                        [0.0000, 0.000, 0.0000, 1.0000]]) ## THIS ONE WORKS

# Streaming loop
try:
    while Estimating:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        if not aligned_depth_frame or not color_frame:
            continue
        depth_image = np.asanyarray(aligned_depth_frame.get_data())/1e3 # convert to meters
        color_image = np.asanyarray(color_frame.get_data())
        depth_image_scaled = (depth_image * depth_scale * 1000).astype(np.float32)
        if cv2.waitKey(1) == 13:
            Estimating = False
            break        
        H, W = color_image.shape[:2]
        color = cv2.resize(color_image, (W,H), interpolation=cv2.INTER_NEAREST)
        depth = cv2.resize(depth_image_scaled, (W,H), interpolation=cv2.INTER_NEAREST)
        depth[(depth<0.1) | (depth>=np.inf)] = 0
        if i==0:
            if len(mask.shape)==3:
                for c in range(3):
                    if mask[...,c].sum()>0:
                        mask = mask[...,c]
                        break
            mask = cv2.resize(mask, (W,H), interpolation=cv2.INTER_NEAREST).astype(bool).astype(np.uint8)
        
            pose = est.register(K=cam_K, rgb=color, depth=depth, ob_mask=mask, iteration=args.est_refine_iter)
        else:
            pose = est.track_one(rgb=color, depth=depth, K=cam_K, iteration=args.track_refine_iter)
        center_pose = pose@np.linalg.inv(to_origin)
        object_center_in_camera = center_pose[:3, 3]
        object_center_in_camera = np.append(object_center_in_camera, 1.0).reshape(4, 1)
        object_center_in_torso = T_torso_camera_withshiftedy @ object_center_in_camera

        vis, pt_transformed1, pt_transformed2 = draw_posed_3d_box_wpoints(cam_K, img=color, ob_in_cam=center_pose, bbox=bbox)
        pt_transformed1 = np.append(pt_transformed1, 1.0).reshape(4, 1)
        pt_transformed2 = np.append(pt_transformed2, 1.0).reshape(4, 1)
        
        vis = draw_xyz_axis(color, ob_in_cam=center_pose, scale=0.1, K=cam_K, thickness=3, transparency=0, is_input_rgb=True)
        pt1_robotframe = T_torso_camera_withshiftedy @ pt_transformed1
        pt2_robotframe = T_torso_camera_withshiftedy @ pt_transformed2
        # breakpoint()
        # with open('transformed_points.txt', 'a') as f:
        #     f.write(f"\n# Frame {i}\n")
        #     f.write("# pt_transformed1\n")
        #     np.savetxt(f, pt1_robotframe, fmt='%.6f')
        #     f.write("\n# pt_transformed2\n")
        #     np.savetxt(f, pt2_robotframe, fmt='%.6f')

        with open('transformed_points.txt', 'a') as f:
            f.write(f"\n# Frame {i}\n")
            f.write("# object center in camera frame\n")
            np.savetxt(f, object_center_in_camera, fmt='%.6f')
            f.write("# object center in robot frame\n")
            np.savetxt(f, object_center_in_torso, fmt='%.6f')
            f.write("# pt_transformed1\n")
            np.savetxt(f, pt1_robotframe, fmt='%.6f')
            f.write("\n# pt_transformed2\n")
            np.savetxt(f, pt2_robotframe, fmt='%.6f')
        
        cv2.imshow('1', vis[...,::-1])
        cv2.waitKey(1)        
        i += 1
        
finally:
    pipeline.stop()