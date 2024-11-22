import glob
import yaml
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

from model.group_mask_2stage import AncestralModel

from utils.torchac_utils import save_byte_stream

ctx_win = 1024
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def process_context(context):
    ctx_len = context.shape[0]

    padding_len = ctx_win - ctx_len % ctx_win

    padding = torch.repeat_interleave(context[-1].unsqueeze(0), padding_len, dim=0)
    context = torch.vstack((context, padding)).unsqueeze(1)

    padding_idx = torch.ones((padding_len)) * -1
    node_idx = torch.arange(ctx_len)
    node_idx = torch.hstack((node_idx, padding_idx))

    return context.long(), node_idx.long()


def encode(model, octree_nodes):
    octree_nodes = torch.tensor(octree_nodes, dtype=torch.long)

    levels = octree_nodes[:, -1, 1]

    softmax = nn.Softmax(dim=1)

    with torch.no_grad():
        bits = 0
        for level in range(levels.max() + 1):
            if level == 0:
                context = torch.zeros((1, 4, 6), dtype=torch.long)
                context[:, :, 3:] = 2 ** levels.max()
            else:
                context = octree_nodes[levels == level]

            length = context.shape[0]
            prob = torch.zeros((length + 1, 256))

            padding_context, padding_idx = process_context(context)
            padding_ctx_len = padding_context.shape[0]

            for i in range(0, padding_ctx_len, ctx_win):
                idx = padding_idx[i : i + ctx_win]
                ctx = padding_context[i : i + ctx_win]

                ctx1 = ctx.clone()
                ctx1[:, :, -1, 0] = ctx1[:, :, -2, 0]
                ctx1[length:, :, -1, 0] = 0

                ctx2 = ctx.clone()
                ctx2[1::2, :, -1, 0] = ctx2[1::2, :, -2, 0]
                ctx2[length:, :, -1, 0] = 0

                output1 = model(ctx1.to(device)).reshape(-1, 256).detach().cpu()
                output2 = model(ctx2.to(device)).reshape(-1, 256).detach().cpu()

                if (level == 6):
                    torch.save(output2, f"ctx2_{level}_{i}.pt")

                output = torch.zeros((ctx_win, 256))
                output[0::2] = output1[0::2]
                output[1::2] = output2[1::2]

                prob[idx] = softmax(output)

            occupys = octree_nodes[levels == level][:, -1, 0]
            if level == 0:
                bits += save_byte_stream(prob[:-1], occupys, f"output/{level}_1.bin")
            else:
                bits += save_byte_stream(prob[:-1][0::2], occupys[0::2], f"output/{level}_1.bin")
                bits += save_byte_stream(prob[:-1][1::2], occupys[1::2], f"output/{level}_2.bin")


    return bits


def main(model, file_name):
    frame_data = np.load(file_name)

    pt_num = frame_data["pt_num"]
    octree_nodes = frame_data["octree_nodes"]

    bits = encode(model, octree_nodes)

    return bits / pt_num


if __name__ == "__main__":
    # ckpt path
    model = AncestralModel.load_from_checkpoint("./last.ckpt").to(device)

    model.eval()

    # origin ply file path
    list_orifile = glob.glob("/zhu/data/Ford/Test/Ford_02_q_1mm/**/*", recursive=True)
    list_orifile = [f for f in list_orifile if f.endswith('h5') or f.endswith('ply') or f.endswith('bin')]
    list_orifile.sort()
    assert list_orifile, "No file found"
    print(f"Total {len(list_orifile)} files")

    level_list = [11, 12, 13, 14, 15, 16, 17, 18]

    for idx, path in enumerate(tqdm(list_orifile)):
        new_row = {"filedir": path}

        for level in level_list:
            # test_file prepared advance
            name = os.path.basename(path)
            file_name = f"./test/{name[:-4]}_{level}.npz"

            bpp = main(model, file_name)

            new_row[f"r{level}_bpp"] = bpp

        results = pd.DataFrame([new_row])

        logdir = f"result_2stage.csv"

        if idx  == 0:
            results.to_csv(logdir, index=False)
        else:
            results.to_csv(logdir, mode='a', header=False, index=False)

