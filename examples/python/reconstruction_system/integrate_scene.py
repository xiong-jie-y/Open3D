# Open3D: www.open3d.org
# The MIT License (MIT)
# See license file or visit www.open3d.org for details

# examples/python/reconstruction_system/integrate_scene.py

import numpy as np
import math
import sys
import open3d as o3d
sys.path.append("../utility")
from file import *
sys.path.append(".")
from make_fragments import read_rgbd_image
from scipy.spatial.transform import Rotation

def scalable_integrate_rgb_frames_into_point_cloud(path_dataset, intrinsic, config):
    poses = []
    [color_files, depth_files] = get_rgbd_file_lists(path_dataset)
    n_files = len(color_files)
    n_fragments = int(math.ceil(float(n_files) / \
            config['n_frames_per_fragment']))

    pose_graph_fragment = o3d.io.read_pose_graph(
        join(path_dataset, config["template_refined_posegraph_optimized"]))

    pcds = []
    for fragment_id in range(len(pose_graph_fragment.nodes[:3])):
        print(fragment_id)
        fragment_pose = pose_graph_fragment.nodes[fragment_id].pose
        print(fragment_pose)
        pcd = o3d.io.read_point_cloud(join(path_dataset, config["template_fragment_pointcloud"] % fragment_id))
        print(config["template_fragment_pointcloud"] % fragment_id)
        pcd.transform(fragment_pose)
        pcds.append(pcd)

    integrated_pcd = o3d.geometry.PointCloud()
    all_points =         np.concatenate([
            np.array(pcd.points) for pcd in pcds
        ], axis=0)
    integrated_pcd.points = o3d.utility.Vector3dVector(
        [point for point in all_points] # if point[1]  > -0.6]
    )


    # import IPython; IPython.embed()
    def _translate_to_camera_centric(point_cloud, position, rotation):
        rot_inv = rotation.inv().as_matrix()
        point_cloud.points = o3d.utility.Vector3dVector(np.array([
                    rot_inv.dot(p)
                    for p in (np.asfarray(point_cloud.points) - position)
        ]))

    open3d_to_rtabmap = (Rotation.from_rotvec([0, -np.pi/2.0, 0]))
    bbb = Rotation.from_rotvec([np.pi/2.0, 0, 0])

    _translate_to_camera_centric(integrated_pcd, np.array([0,0,0]), open3d_to_rtabmap)
    _translate_to_camera_centric(integrated_pcd, np.array([0,0,0]), bbb)

    # integrated_pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.6, origin=[0,0,0])

    integrated_pcd.colors = o3d.utility.Vector3dVector(
        np.concatenate([
            np.array(pcd.colors) for pcd in pcds
        ], axis=0)
    )
    o3d.visualization.draw_geometries([integrated_pcd, coord])
    o3d.io.write_point_cloud("/home/yusuke/gitrepos/votenet/demo_files/input_pc_scannet.ply", integrated_pcd)


def scalable_integrate_rgb_frames(path_dataset, intrinsic, config):
    poses = []
    [color_files, depth_files] = get_rgbd_file_lists(path_dataset)
    n_files = len(color_files)
    n_fragments = int(math.ceil(float(n_files) / \
            config['n_frames_per_fragment']))
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=config["tsdf_cubic_size"] / 512.0,
        sdf_trunc=0.04,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)

    pose_graph_fragment = o3d.io.read_pose_graph(
        join(path_dataset, config["template_refined_posegraph_optimized"]))

    for fragment_id in range(len(pose_graph_fragment.nodes)):
        pose_graph_rgbd = o3d.io.read_pose_graph(
            join(path_dataset,
                 config["template_fragment_posegraph_optimized"] % fragment_id))

        for frame_id in range(len(pose_graph_rgbd.nodes)):
            frame_id_abs = fragment_id * \
                    config['n_frames_per_fragment'] + frame_id
            print(
                "Fragment %03d / %03d :: integrate rgbd frame %d (%d of %d)." %
                (fragment_id, n_fragments - 1, frame_id_abs, frame_id + 1,
                 len(pose_graph_rgbd.nodes)))
            rgbd = read_rgbd_image(color_files[frame_id_abs],
                                   depth_files[frame_id_abs], False, config)
            pose = np.dot(pose_graph_fragment.nodes[fragment_id].pose,
                          pose_graph_rgbd.nodes[frame_id].pose)
            volume.integrate(rgbd, intrinsic, np.linalg.inv(pose))
            poses.append(pose)

    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    if config["debug_mode"]:
        o3d.visualization.draw_geometries([mesh])

    mesh_name = join(path_dataset, config["template_global_mesh"])
    o3d.io.write_triangle_mesh(mesh_name, mesh, False, True)

    traj_name = join(path_dataset, config["template_global_traj"])
    write_poses_to_log(traj_name, poses)


def run(config):
    print("integrate the whole RGBD sequence using estimated camera pose.")
    if config["path_intrinsic"]:
        intrinsic = o3d.io.read_pinhole_camera_intrinsic(
            config["path_intrinsic"])
    else:
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)

    if 'make_point_cloud' in config and config['make_point_cloud']:
        scalable_integrate_rgb_frames_into_point_cloud(config["path_dataset"], intrinsic, config)
    else:
        scalable_integrate_rgb_frames(config["path_dataset"], intrinsic, config)
