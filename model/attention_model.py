import torch
import torch.nn as nn

class MultiheadAttention(nn.Module):
    def __init__(self, embed_dim, heads) -> None:
        super().__init__()

        self.embed_dim = embed_dim
        self.heads = heads
        self.head_dim = embed_dim // heads

        assert (self.head_dim * self.heads == self.embed_dim), "Embedding size needs to be divisible by heads"

        self.linear = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(self, Q, K, V, mask=None, idx=0):
        # b, l, _, _ = Q.size()
        b, l, _ = Q.size()

        Q = Q.reshape(b, l, self.heads, self.head_dim) # (b, l, h, d)
        K = K.reshape(b, l, self.heads, self.head_dim) # (b, l, h, d)
        V = V.reshape(b, l, self.heads, self.head_dim) # (b, l, h, d)

        energy = torch.einsum("nqhd, nkhd -> nhqk", [Q, K])

        if mask is not None:
            energy = energy.masked_fill(mask, float("-inf"))

        attention_map = torch.softmax(energy / (self.embed_dim ** (1 / 2)), dim=-1)

        out = torch.einsum("nhql, nlhd -> nqhd", [attention_map, V])

        out = out.reshape(b, l, self.heads * self.head_dim)

        out = self.linear(out)

        return out


class SelfAttention(nn.Module):
    def __init__(self, embed_dim, heads) -> None:
        super().__init__()

        # self.heads = heads
        # self.head_dim = embed_dim // heads

        # self.Q_linear = nn.Linear(self.head_dim, self.head_dim, bias=False)
        # self.K_linear = nn.Linear(self.head_dim, self.head_dim, bias=False)
        # self.V_linear = nn.Linear(self.head_dim, self.head_dim, bias=False)

        self.Q_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.K_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.V_linear = nn.Linear(embed_dim, embed_dim, bias=False)

        self.attention = MultiheadAttention(embed_dim, heads)

    def forward(self, src, mask, idx):
        l, b, c = src.size()

        src = src.transpose(0, 1).contiguous()  # (b, l, c)

        # src = src.reshape(b, l, self.heads, self.head_dim) # (b, l, h, d)

        Q = self.Q_linear(src)
        K = self.K_linear(src)
        V = self.V_linear(src)

        out = self.attention(Q, K, V, mask, idx)

        out = out.transpose(0, 1).contiguous()  # (l, b, c)

        return out

class AttentionLayer(nn.Module):
    def __init__(self, ninp, nhead, nhid, dropout):
        super().__init__()

        self.attention = SelfAttention(ninp, nhead)

        self.sequnetial0 = nn.Sequential(
            nn.Linear(ninp, nhid),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(nhid, ninp),
            nn.Dropout(dropout),
        )

        self.ln0 = nn.LayerNorm(ninp, eps=1e-5)
        self.ln1 = nn.LayerNorm(ninp, eps=1e-5)

        self.dropout = nn.Dropout(dropout)

    def forward(self, src, mask, idx):
        out = self.dropout(self.attention(src, mask, idx))
        src = self.ln0(src + out)

        out = self.sequnetial0(src)
        src = self.ln1(src + out)

        return src

class SelfAttentionModule(nn.Module):
    def __init__(self, ninp, nhead, nhid, dropout, nlayers):
        super().__init__()

        self.layers = nn.ModuleList([
            AttentionLayer(ninp, nhead, nhid, dropout)
            for _ in range(nlayers)
        ])

    def forward(self, src, mask):
        for idx, layer in enumerate(self.layers):
            src = layer(src, mask, idx)

        return src
