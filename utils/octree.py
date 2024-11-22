import numpy as np
import torch

class TreeNode:
    my_shift = np.array(
        (
            (0, 0, 0),
            (1, 0, 0),
            (0, 1, 0),
            (1, 1, 0),
            (0, 0, 1),
            (1, 0, 1),
            (0, 1, 1),
            (1, 1, 1),
        )
    )

    def __init__(self, min_bound, node_idx, curr_occu, level, par_ctx, voxel_size_by_level):
        self.min_bound = min_bound
        self.node_idx = node_idx
        self.curr_occu = curr_occu
        self.child_node_ls = []
        self.level = level
        self.par_ctx = par_ctx
        self.voxel_size_by_level = voxel_size_by_level
        self.init_origin_coords()

    def init_origin_coords(self):
        voxel_size = self.voxel_size_by_level[self.level]
        self.origin = self.min_bound + self.my_shift[self.node_idx] * voxel_size
        self.coords = self.origin + voxel_size * 0.5

    def get_children_nodes(self, occu_symbols):
        assert self.curr_occu == None
        self.curr_occu = int(occu_symbols)

        min_bound = self.origin
        level = self.level + 1

        occupy = '{0:08b}'.format(self.curr_occu)
        idx_ls = [i for i, e in enumerate(occupy) if e != "0"]

        for _, node_idx in enumerate(idx_ls):
            curr_occu = None
            par_ctx = self.get_decode_context()[1:]
            child_node = TreeNode(min_bound, node_idx, curr_occu, level, par_ctx, self.voxel_size_by_level)
            self.child_node_ls.append(child_node)

        return self.child_node_ls

    def get_decode_context(self):
        # get the context of node
        if self.par_ctx is None:
            occupy = 0 if self.curr_occu is None else self.curr_occu
            curr = torch.tensor([occupy, self.level, self.node_idx, *self.coords])
            context = torch.repeat_interleave(curr.unsqueeze(0), 4, dim=0)
        else:
            prev = self.par_ctx

            occupy = prev[-1, 0] if self.curr_occu is None else self.curr_occu
            curr = torch.tensor([occupy, self.level, self.node_idx, *self.coords])

            context = torch.cat((prev, curr.unsqueeze(0)), dim=0)

        return context


def get_voxel_size_by_level_dict(max_bound, min_bound, level):
    voxel_size_by_level = dict()

    for i in range(level + 1):
        voxel_size_by_level.update({i: (max_bound - min_bound) / (2 ** i)})

    return voxel_size_by_level
