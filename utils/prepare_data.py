import os

import numpy as np
import open3d as o3d

# from utils.io_utils import read_coords
from io_utils import read_coords


def process_point_clouds(point_path, to_save_path, level, train, tree_point=1024):
    def traverse_get_parent_info(node, node_info):
        early_stop = False

        if isinstance(node, o3d.geometry.OctreeInternalNode):
            if isinstance(node, o3d.geometry.OctreeInternalPointNode):
                tmp = [n for n in node_info_stack if n[1].depth < node_info.depth]
                node_info_stack.clear()
                node_info_stack.extend(tmp)
                node_info_stack.append([node, node_info])

                info_array = np.array([])

                for n, i in node_info_stack[-4:]:
                    child_cnt = 0
                    occupancy = '0b'
                    for x in n.children:
                        if x:
                            occupancy = occupancy + ''.join('1')
                            child_cnt += 1
                        else:
                            occupancy = occupancy + ''.join('0')

                    occupancy = np.array([float(int(occupancy, 2))], dtype=np.float32)
                    child_idx = np.array([i.child_index], dtype=np.float32)
                    depth = np.array([i.depth], dtype=np.float32)
                    coords = (i.origin.reshape(-1) + (i.size / 2.0)).astype(np.float32)

                    """
                    0: occupancy
                    1: depth
                    2: child_idx
                    3-5: coords
                    """
                    info = np.concatenate((occupancy, depth, child_idx, coords))
                    info = np.expand_dims(info, axis=0)

                    if info_array.size == 0:
                        info_array = info
                    else:
                        info_array = np.append(info_array, info, axis=0)

                if info_array.shape[0] < 4:
                    padding = np.tile(info_array[0], 4 - info_array.shape[0]).reshape(4 - info_array.shape[0], 6)
                    info_array = np.concatenate((padding, info_array), axis=0)

                depth = info_array[-1, 1]
                if depth not in pts_by_level.keys():
                    pts_by_level.update({depth: [info_array]})
                else:
                    ls = pts_by_level[depth]
                    ls.append(info_array)
                    pts_by_level[depth] = ls

        elif isinstance(node, o3d.geometry.OctreeLeafNode):
            pass

        else:
            raise NotImplementedError('type not recognized!')

        return early_stop

    """Load data and build the octree with level"""
    point_cloud = read_coords(point_path, dtype="float32")

    pt_num = point_cloud.shape[0]

    offset = point_cloud.min(axis=0)
    point_cloud = point_cloud - offset

    qs = 2 ** (18 - level)

    point_cloud = np.round(point_cloud / qs)
    point_cloud = np.unique(point_cloud, axis=0)

    point_cloud = np.append(point_cloud, [[2**level, 2**level, 2**level]], axis=0)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    octree = o3d.geometry.Octree(max_depth=level)
    octree.convert_from_point_cloud(pcd, size_expand=0.)

    node_info_stack = []
    pts_by_level = {}

    octree.traverse(traverse_get_parent_info)

    pts_by_level = [np.array(pts_by_level[i]).astype(np.float32) for i in range(len(pts_by_level))]
    octree_nodes = np.concatenate(pts_by_level, axis=0).astype(np.float32)

    if train == False:
        filename = os.path.join(to_save_path, os.path.basename(point_path).replace('.ply', f'_{level}.npz'))

        if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))

        max_bound = octree.get_max_bound().reshape(-1)
        min_bound = octree.get_min_bound().reshape(-1)

        np.savez_compressed(filename, octree_nodes=octree_nodes, max_bound=max_bound, min_bound=min_bound, offset=offset, level=level, pt_num=pt_num)
    else:
        length = len(octree_nodes) - len(octree_nodes) % tree_point

        for i in range(0, length, tree_point):
            filename = os.path.join(to_save_path, os.path.basename(point_path).replace('.ply', f'_{i}.npz'))

            if not os.path.exists(os.path.dirname(filename)):
                os.makedirs(os.path.dirname(filename))

            nodes_tmp = octree_nodes[i : i + tree_point + 1]
            if len(nodes_tmp) != tree_point + 1:
                print(f"Error: {len(nodes_tmp)}")
            else:
                np.savez_compressed(filename, octree_nodes=nodes_tmp)

    return filename

if __name__ == "__main__":
    import glob

    import joblib

    point_paths = glob.glob(f"/zhu/data/Ford/Train/Ford_01_q_1mm/**/*", recursive=True)
    point_paths = sorted(point_paths)

    # train
    to_save_path = "./train"

    # test
    # to_save_paht = "test_path"
    # to_save_path = "./test"

    for point_path in point_paths[:2]:
        print(point_path)
        process_point_clouds(point_path, to_save_path, 11, train=True)

    # joblib.Parallel(
    #     n_jobs=24, verbose=10, pre_dispatch="all"
    # )(
    #     [
    #         joblib.delayed(process_point_clouds)(point_path, to_save_path, 11, train=True)
    #         for point_path in point_paths
    #     ]
    # )
