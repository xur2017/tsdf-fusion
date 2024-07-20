
import numpy as np
import open3d as o3d
import fusion

obj_list = []
pcd1 = o3d.io.read_point_cloud("fused.ply")
obj_list.append(pcd1)

cam_intr = np.loadtxt("data/camera-intrinsics.txt", delimiter=' ')
fx, fy = cam_intr[0, 0], cam_intr[1, 1]
cx, cy = cam_intr[0, 2], cam_intr[1, 2]
im_w, im_h = 2*cx, 2*cy 


color = [0.8, 0.2, 0.8]
z = 0.1
n_imgs = 1000
fill = "0"
width = 6
for j in range(0,n_imgs,10):
    cam_pose = np.loadtxt(f"data/frame-{j:{fill}{width}}.pose.txt")  # 4x4 rigid transformation matrix
    R = cam_pose[:3,:3]
    t = cam_pose[:3,3]
    
    cam_pts0 = np.array([
        (np.array([0, 0, 0, im_w, im_w])-cx)*np.array([0, z, z, z, z])/fx,
        (np.array([0, 0, im_h, 0, im_h])-cy)*np.array([0, z, z, z, z])/fy,
         np.array([0, z, z, z, z])
    ]).T
    
    cam_pts1 = fusion.apply_transform(cam_pts0, cam_pose)
    lines = [[0, 1],[0, 2],[0, 3],[0, 4],[1,2],[1,3],[4,2],[4,3]]
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(cam_pts1),
        lines=o3d.utility.Vector2iVector(lines),)
    obj_list.append(line_set)

    w = abs(cam_pts0[1][0]-cam_pts0[3][0])
    h = abs(cam_pts0[1][1]-cam_pts0[2][1])
    plane = o3d.geometry.TriangleMesh.create_box(w, h, depth=1e-6)
    plane.paint_uniform_color(color)
    plane.translate([cam_pts0[1][0], cam_pts0[1][1], cam_pts0[1][2]])
    plane.transform(cam_pose)
    obj_list.append(plane)

o3d.visualization.draw_geometries(obj_list)