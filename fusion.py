
from skimage import measure
import numpy as np
from numba import njit, prange

class TSDFVolume:
    def __init__(self, vol_bnds, voxel_size):
        """
        Args:
        vol_bnds (ndarray): x_min, x_max, y_min, y_max, z_min, z_max in meters.
        voxel_size (float): voxel_size in meters.
        """
        self._vol_bnds = vol_bnds
        self._voxel_size = float(voxel_size)
        self._trunc_margin = 5 * self._voxel_size  # truncation on SDF
        self._color_const = 256 * 256

        self._vol_dim = np.ceil((self._vol_bnds[:,1]-self._vol_bnds[:,0])/self._voxel_size).astype(int)
        self._vol_bnds[:,1] = self._vol_bnds[:,0]+self._vol_dim*self._voxel_size
        self._vol_origin = self._vol_bnds[:,0].astype(np.float32)

        self._tsdf_vol_cpu = np.ones(self._vol_dim).astype(np.float32)
        # for computing the cumulative moving average of observations per voxel
        self._weight_vol_cpu = np.zeros(self._vol_dim).astype(np.float32)
        self._color_vol_cpu = np.zeros(self._vol_dim).astype(np.float32)

        xv, yv, zv = np.meshgrid(range(self._vol_dim[0]), range(self._vol_dim[1]), range(self._vol_dim[2]), indexing='ij')
        self.vox_coords = np.concatenate([xv.reshape(1,-1), yv.reshape(1,-1), zv.reshape(1,-1)], axis=0).astype(int).T

    def integrate(self, color_im, depth_im, cam_intr, cam_pose, obs_weight=1.):
        """Integrate an RGB-D frame into the TSDF volume.
        Args:
        color_im (ndarray): An RGB image of shape (H, W, 3).
        depth_im (ndarray): A depth image of shape (H, W).
        cam_intr (ndarray): The camera intrinsics matrix of shape (3, 3).
        cam_pose (ndarray): The camera pose (i.e. extrinsics) of shape (4, 4).
        obs_weight (float): The weight to assign for the current observation.
        """
        im_h, im_w = depth_im.shape

        # Fold RGB color image into a single channel image
        color_im = color_im.astype(np.float32)
        color_im = np.floor(color_im[...,2]*self._color_const + color_im[...,1]*256 + color_im[...,0])

        # Convert voxel grid coordinates to pixel coordinates
        cam_pts = vox2world(self._vol_origin, self.vox_coords, self._voxel_size)
        cam_pts = apply_transform(cam_pts, np.linalg.inv(cam_pose))
        pix_z = cam_pts[:, 2]
        pix = cam2pix(cam_pts, cam_intr)
        pix_x, pix_y = pix[:, 0], pix[:, 1]

        # Eliminate pixels outside view frustum
        valid_pix = (pix_x >= 0) & (pix_x < im_w) & (pix_y >= 0) & (pix_y < im_h) & (pix_z > 0)
        depth_val = np.zeros(pix_x.shape)
        depth_val[valid_pix] = depth_im[pix_y[valid_pix], pix_x[valid_pix]]

        # Integrate TSDF
        depth_diff = depth_val - pix_z
        valid_pts = (depth_val > 0) & (depth_diff >= -self._trunc_margin)
        dist = np.minimum(1, depth_diff / self._trunc_margin)
        valid_vox_x = self.vox_coords[valid_pts, 0]
        valid_vox_y = self.vox_coords[valid_pts, 1]
        valid_vox_z = self.vox_coords[valid_pts, 2]
        w_old = self._weight_vol_cpu[valid_vox_x, valid_vox_y, valid_vox_z]
        tsdf_vals = self._tsdf_vol_cpu[valid_vox_x, valid_vox_y, valid_vox_z]
        valid_dist = dist[valid_pts]
        tsdf_vol_new, w_new = integrate_tsdf(tsdf_vals, valid_dist, w_old, obs_weight)
        self._weight_vol_cpu[valid_vox_x, valid_vox_y, valid_vox_z] = w_new
        self._tsdf_vol_cpu[valid_vox_x, valid_vox_y, valid_vox_z] = tsdf_vol_new

        # Integrate color
        old_color = self._color_vol_cpu[valid_vox_x, valid_vox_y, valid_vox_z]
        old_b = np.floor(old_color / self._color_const)
        old_g = np.floor((old_color-old_b*self._color_const)/256)
        old_r = old_color - old_b*self._color_const - old_g*256
        new_color = color_im[pix_y[valid_pts],pix_x[valid_pts]]
        new_b = np.floor(new_color / self._color_const)
        new_g = np.floor((new_color - new_b*self._color_const) /256)
        new_r = new_color - new_b*self._color_const - new_g*256
        new_b = np.minimum(255., np.round((w_old*old_b + obs_weight*new_b) / w_new))
        new_g = np.minimum(255., np.round((w_old*old_g + obs_weight*new_g) / w_new))
        new_r = np.minimum(255., np.round((w_old*old_r + obs_weight*new_r) / w_new))
        self._color_vol_cpu[valid_vox_x, valid_vox_y, valid_vox_z] = new_b*self._color_const + new_g*256 + new_r

    def get_point_cloud(self):
        """Extract a point cloud from the voxel volume.
        """
        tsdf_vol, color_vol = self._tsdf_vol_cpu, self._color_vol_cpu

        # Marching cubes
        verts = measure.marching_cubes(tsdf_vol, level=0)[0]
        verts_ind = np.round(verts).astype(int)
        verts = verts*self._voxel_size + self._vol_origin

        # Get vertex colors
        rgb_vals = color_vol[verts_ind[:, 0], verts_ind[:, 1], verts_ind[:, 2]]
        colors_b = np.floor(rgb_vals / self._color_const)
        colors_g = np.floor((rgb_vals - colors_b*self._color_const) / 256)
        colors_r = rgb_vals - colors_b*self._color_const - colors_g*256
        colors = np.floor(np.asarray([colors_r, colors_g, colors_b])).T
        colors = colors.astype(np.uint8)

        pcd = np.hstack([verts, colors])
        return pcd


def vox2world(vol_origin, vox_coords, vox_size):
    """Convert voxel grid coordinates to world coordinates.
    """
    vol_origin = vol_origin.astype(np.float32)
    vox_coords = vox_coords.astype(np.float32)
    cam_pts = np.empty_like(vox_coords, dtype=np.float32)
    cam_pts[:, 0] = vol_origin[0] + (vox_size * vox_coords[:, 0])
    cam_pts[:, 1] = vol_origin[1] + (vox_size * vox_coords[:, 1])
    cam_pts[:, 2] = vol_origin[2] + (vox_size * vox_coords[:, 2])
    return cam_pts

def cam2pix(cam_pts, intr):
    """Convert camera coordinates to pixel coordinates.
    """
    intr = intr.astype(np.float32)
    fx, fy = intr[0, 0], intr[1, 1]
    cx, cy = intr[0, 2], intr[1, 2]
    pix = np.empty((cam_pts.shape[0], 2), dtype=np.int64)
    pix[:, 0] = (np.round((cam_pts[:, 0] * fx / cam_pts[:, 2]) + cx)).astype(np.int64)
    pix[:, 1] = (np.round((cam_pts[:, 1] * fy / cam_pts[:, 2]) + cy)).astype(np.int64)
    return pix

@njit(parallel=True)
def integrate_tsdf(tsdf_vol, dist, w_old, obs_weight):
    """Integrate the TSDF volume.
    """
    tsdf_vol_int = np.empty_like(tsdf_vol, dtype=np.float32)
    w_new = np.empty_like(w_old, dtype=np.float32)
    for i in prange(len(tsdf_vol)):
        w_new[i] = w_old[i] + obs_weight
        tsdf_vol_int[i] = (w_old[i] * tsdf_vol[i] + obs_weight * dist[i]) / w_new[i]
    return tsdf_vol_int, w_new

def apply_transform(xyz, transform):
    """Applies a rigid transform to an (N, 3) pointcloud.
    """
    xyz_h = np.hstack([xyz, np.ones((len(xyz), 1), dtype=np.float32)])
    xyz_t_h = np.dot(transform, xyz_h.T).T
    return xyz_t_h[:, :3]

def get_view_frustum(depth_im, cam_intr, cam_pose):
    """Get corners of view frustum
    """
    im_h, im_w = depth_im.shape
    fx, fy = cam_intr[0, 0], cam_intr[1, 1]
    cx, cy = cam_intr[0, 2], cam_intr[1, 2]
    max_z = np.max(depth_im)
    view_frust_pts = np.array([
        (np.array([0, 0, im_w, im_w, 0, 0, im_w, im_w])-cx)*np.array([0, 0, 0, 0, max_z, max_z, max_z, max_z])/fx,
        (np.array([0, im_h, 0, im_h, 0, im_h, 0, im_h])-cy)*np.array([0, 0, 0, 0, max_z, max_z, max_z, max_z])/fy,
         np.array([0, 0, 0, 0, max_z, max_z, max_z, max_z])
    ])
    view_frust_pts = apply_transform(view_frust_pts.T, cam_pose).T
    return view_frust_pts