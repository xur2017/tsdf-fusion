
import cv2
import numpy as np
import fusion
import ply_io

n_imgs = 1000
fill = "0"
width = 6
cam_intr = np.loadtxt("data/camera-intrinsics.txt", delimiter=' ')
vol_bnds = np.zeros((3,2))
for j in range(n_imgs):
    # Read depth image and camera pose
    depth_im = cv2.imread(f"data/frame-{j:{fill}{width}}.depth.png",-1).astype(float)
    depth_im /= 1000.  # depth is saved in 16-bit PNG in millimeters
    depth_im[depth_im == 65.535] = 0 # handling the invalid pixels
    cam_pose = np.loadtxt(f"data/frame-{j:{fill}{width}}.pose.txt")  # 4x4 rigid transformation matrix

    # Compute camera view frustum and extend convex hull
    view_frust_pts = fusion.get_view_frustum(depth_im, cam_intr, cam_pose)
    vol_bnds[:,0] = np.minimum(vol_bnds[:,0], np.amin(view_frust_pts, axis=1))
    vol_bnds[:,1] = np.maximum(vol_bnds[:,1], np.amax(view_frust_pts, axis=1))


tsdf_vol = fusion.TSDFVolume(vol_bnds, voxel_size=0.02)
for j in range(0, n_imgs, 5):
    print(f"Fusing frame {j+1}/{n_imgs}")

    # Read RGB-D image and camera pose
    color_image = cv2.cvtColor(cv2.imread(f"data/frame-{j:{fill}{width}}.color.jpg"), cv2.COLOR_BGR2RGB)
    depth_im = cv2.imread(f"data/frame-{j:{fill}{width}}.depth.png",-1).astype(float)
    depth_im /= 1000.
    depth_im[depth_im == 65.535] = 0 # handling the invalid pixels
    cam_pose = np.loadtxt(f"data/frame-{j:{fill}{width}}.pose.txt")

    # Integrate observation into voxel volume (assume color aligned with depth)
    tsdf_vol.integrate(color_image, depth_im, cam_intr, cam_pose, obs_weight=1.)


xyzrgb = tsdf_vol.get_point_cloud()
ply_io.write_pcd(f"fused.ply", xyzrgb)