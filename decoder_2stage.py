import os
from collections import deque

import numpy as np
import torch
from tqdm import tqdm

from model.group_mask_2stage import AncestralModel
from utils.io_utils import write_ply_ascii
from utils.octree import TreeNode, get_voxel_size_by_level_dict
from utils.torchac_utils import get_symbol_from_byte_stream

ctx_win = 1024

def process_context(context):
    context = torch.stack(context, dim=0).long()
    ctx_len = context.shape[0]

    padding_len = ctx_win - ctx_len % ctx_win

    padding = torch.repeat_interleave(context[-1].unsqueeze(0), padding_len, dim=0)
    context = torch.vstack((context, padding)).unsqueeze(1)

    padding_idx = torch.ones((padding_len)) * -1
    node_idx = torch.arange(ctx_len)
    node_idx = torch.hstack((node_idx, padding_idx))

    return context.long().cuda(), node_idx.long().cuda()


if __name__ == "__main__":
    data_frame = np.load("test/Ford_02_vox1mm-0100_11.npz")

    occupys = data_frame["octree_nodes"][:, -1, 0].astype(np.int32)
    levels = data_frame["octree_nodes"][:, -1, 1].astype(np.int32)
    octants = data_frame["octree_nodes"][:, -1, 2].astype(np.int32)
    coords = data_frame["octree_nodes"][:, -1, 3:].astype(np.int32)

    offset = data_frame["offset"]
    max_level = data_frame["level"]
    max_bound = data_frame["max_bound"]
    min_bound = data_frame["min_bound"]

    range_bound = max_bound - min_bound

    # Dictionary to obtain voxel size by level
    voxel_size_by_level = get_voxel_size_by_level_dict(max_bound, min_bound, max_level)

    model = AncestralModel.load_from_checkpoint("./last.ckpt").cuda()
    model.eval()

    total_symbol = []
    # start decode
    softmax = torch.nn.Softmax(dim=1)

    nodes = deque()
    nodes.append(TreeNode(min_bound, node_idx=0, curr_occu=None, level=0, par_ctx=None, voxel_size_by_level=voxel_size_by_level))

    sum_length = 0
    for level in tqdm(range(0, max_level)):
        ref = torch.tensor(occupys[levels == level])

        with open(f"output/{level}_1.bin", "rb") as fin:
            byte_stream_1 = fin.read()

        if level != 0:
            with open(f"output/{level}_2.bin", "rb") as fin:
                byte_stream_2 = fin.read()

        length = len(nodes)
        padding_ctx, padding_idx = process_context([node.get_decode_context() for node in nodes])
        padding_ctx_len = padding_ctx.shape[0]
        padding_ctx[length:, :, -1, 0] = 0

        prob1 = torch.zeros((length + 1, 256))

        # stage 1
        for i in range(0, padding_ctx_len, ctx_win):
            idx1 = padding_idx[i : i + ctx_win].clone()
            ctx1 = padding_ctx[i : i + ctx_win].clone()
            output1 = model(ctx1).reshape(-1, 256)
            prob1[idx1] = softmax(model(ctx1).reshape(-1, 256)).detach().cpu()

        prob1 = prob1[:-1].cuda()
        syms1 = get_symbol_from_byte_stream(byte_stream_1, prob1[0::2])

        if level != 0:
            prob2 = torch.zeros((length + 1, 256))

            for i in range(0, padding_ctx_len, ctx_win):
                idx2 = padding_idx[i : i + ctx_win].clone()
                ctx2 = padding_ctx[i : i + ctx_win].clone()

                odd = torch.arange(0, min(length, ctx_win), 2)

                ctx2[odd, :, -1, 0] = syms1.reshape(-1, 1).long().cuda()

                # if i == padding_ctx_len // ctx_win - 1:
                #     ctx2[length % ctx_win:] = ctx2[length % ctx_win - 1]

                if level == 6:
                    ref = torch.load(f"ctx2_6_0.pt")

                output2 = model(ctx2).reshape(-1, 256)

                if level == 6:
                    print(output2.eq(ref.cuda()).all())

                prob2[idx2] = softmax(output2.cuda()).detach().cpu()

            syms2 = get_symbol_from_byte_stream(byte_stream_2, prob2[:-1][1::2])

            syms_len = len(syms1) + len(syms2)
            syms = torch.zeros((syms_len), dtype=torch.int64)
            syms[0::2] = syms1
            syms[1::2] = syms2
        else:
            syms = syms1

        total_symbol.extend(syms)
        assert total_symbol == occupys[: len(total_symbol)].tolist()

        new_nodes = deque()
        for i in range(length):
            nodes[i].get_children_nodes(syms[i])
            new_nodes.extend(nodes[i].child_node_ls)

        nodes = new_nodes

        torch.cuda.empty_cache()

    coords = [np.expand_dims(node.origin, axis=0) for node in nodes]
    coords = np.concatenate(coords, axis=0)

    qs = 2 ** (18 - max_level)
    dec_pt = coords * qs + offset

    write_ply_ascii("output/dec.ply", dec_pt)
    print("Done!")
