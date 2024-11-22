import glob

import numpy as np
from torch.utils.data import Dataset


class NodeDataset(Dataset):
    def __init__(self, point_path):
        self.file_list = sorted(glob.glob(point_path + "/**/*", recursive=True))

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        nodes = np.load(self.file_list[idx])['octree_nodes']

        return nodes

if __name__ == "__main__":
    import torch.utils.data.dataloader as DataLoader

    train_data_root = "/media/qy11506/Data/zhu/DataSets/Ford/Ford_MPEG/Train/Ford_01_q_1mm_octree_ancestor4_tile_BFS_perfect_2024-1-3"
    # train_data_root = "./data/"
    train_data = NodeDataset(point_path=train_data_root)
    train_loader = DataLoader.DataLoader(dataset=train_data, batch_size=4, shuffle=False, num_workers=8, drop_last=True)

    for idx, data in enumerate(train_loader):
        print(idx, data.shape)
