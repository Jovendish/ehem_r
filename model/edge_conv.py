import torch
import torch.nn as nn
import torch.nn.functional as F


def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)

    xx = torch.sum(x**2, dim=1, keepdim=True)

    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k, dim=-1)[1]

    return idx


def get_graph_feature(x, k=20):
    batch_size = x.size(0)
    num_points = x.size(2)

    x = x.view(batch_size, -1, num_points)

    idx = knn(x, k)

    idx_base = torch.arange(0, batch_size, device=x.device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()

    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)

    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()

    return feature


class BaseConv(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(BaseConv, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.LeakyReLU(negative_slope=0.2)
        )

    def forward(self, x):
        return self.conv(x)


class ChannelAttention(nn.Module):
    def __init__(self, input_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(input_channels, input_channels // reduction_ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(input_channels // reduction_ratio, input_channels, 1, bias=False),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class EdgeConv(nn.Module):
    def __init__(self, k):
        super(EdgeConv, self).__init__()

        self.k = k

        self.encoder0_32 = nn.Embedding(256 + 1, 6, padding_idx=0)
        self.encoder1_32 = nn.Embedding(18 + 1, 1)
        self.encoder2_32 = nn.Embedding(8 + 1, 1)

        self.encoder0_128 = nn.Embedding(256 + 1, 30, padding_idx=0)
        self.encoder1_128 = nn.Embedding(18 + 1, 1)
        self.encoder2_128 = nn.Embedding(8 + 1, 1)

        self.encoder0_512 = nn.Embedding(256 + 1, 126, padding_idx=0)
        self.encoder1_512 = nn.Embedding(18 + 1, 1)
        self.encoder2_512 = nn.Embedding(8 + 1, 1)

        self.conv1 = BaseConv(6, 32)
        self.conv3 = BaseConv(128, 128)
        self.conv5 = BaseConv(512, 512)

        self.conv  = nn.Sequential(
            nn.Conv1d(1344, 512, kernel_size=1, bias=False),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(negative_slope=0.2)
        )

        self.mlp = nn.Linear(2048, 512)

    def forward(self, occupy, level, octant, pos):
        # 32 channels
        occupy_emb_32 = self.encoder0_32(occupy)
        level_emb_32  = self.encoder1_32(level)
        octant_emb_32 = self.encoder2_32(octant)

        emb_32 = torch.cat((occupy_emb_32, level_emb_32, octant_emb_32), -1)
        emb_32 = emb_32.reshape((emb_32.shape[0], emb_32.shape[1], -1))

        # 128 channels
        occupy_emb_128 = self.encoder0_128(occupy)
        level_emb_128  = self.encoder1_128(level)
        octant_emb_128 = self.encoder2_128(octant)

        emb_128 = torch.cat((occupy_emb_128, level_emb_128, octant_emb_128), -1)
        emb_128 = emb_128.reshape((emb_128.shape[0], emb_128.shape[1], -1))

        # 512 channels
        occupy_emb_512 = self.encoder0_512(occupy)
        level_emb_512  = self.encoder1_512(level)
        octant_emb_512 = self.encoder2_512(octant)

        emb_512 = torch.cat((occupy_emb_512, level_emb_512, octant_emb_512), -1)
        emb_512 = emb_512.reshape((emb_512.shape[0], emb_512.shape[1], -1))

        x = pos.permute(1, 2, 0)

        k = min(self.k, x.size(2))

        x1 = get_graph_feature(x, k)
        x1 = self.conv1(x1)
        x1 = torch.max(x1, dim=-1)[0].permute(2, 0, 1)
        x1 = torch.cat((x1, emb_32), dim=-1)
        x1 = x1.permute(1, 2, 0)

        x3 = get_graph_feature(x1, k)
        x3 = self.conv3(x3)
        x3 = torch.max(x3, dim=-1)[0].permute(2, 0, 1)
        x3 = torch.cat((x3, emb_128), dim=-1)
        x3 = x3.permute(1, 2, 0)

        x5 = get_graph_feature(x3, k)
        x5 = self.conv5(x5)
        x5 = torch.max(x5, dim=-1)[0].permute(2, 0, 1)
        x5 = torch.cat((x5, emb_512), dim=-1)
        x5 = x5.permute(1, 2, 0)

        x = torch.cat((x1, x3, x5), dim=1)
        x = self.conv(x)

        l, _, _ = pos.size()
        x_avg = F.avg_pool1d(x, l).squeeze(-1).unsqueeze(0).repeat(l, 1, 1)
        x_max = F.max_pool1d(x, l).squeeze(-1).unsqueeze(0).repeat(l, 1, 1)

        out = torch.cat((x5.permute(2, 0, 1), x_avg, x_max), dim=-1)
        out = self.mlp(out)

        return out
