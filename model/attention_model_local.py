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

    def forward(self, Q, K, V, mask=None):
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

        self.Q_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.K_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.V_linear = nn.Linear(embed_dim, embed_dim, bias=False)

        self.attention = MultiheadAttention(embed_dim, heads)

    def forward(self, src, mask):
        src = src.transpose(0, 1).contiguous()  # (b, l, c)

        Q = self.Q_linear(src)
        K = self.K_linear(src)
        V = self.V_linear(src)

        out = self.attention(Q, K, V, mask)

        out = out.transpose(0, 1).contiguous()  # (l, b, c)

        return out


class CrossAttention(nn.Module):
    def __init__(self, embed_dim, heads) -> None:
        super().__init__()

        self.Q_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.K_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.V_linear = nn.Linear(embed_dim, embed_dim, bias=False)

        self.attention = MultiheadAttention(embed_dim, heads)

    def forward(self, src, tgt, mask):
        src = src.transpose(0, 1).contiguous()  # (b, l, c)
        tgt = tgt.transpose(0, 1).contiguous()  # (b, l, c)

        Q = self.Q_linear(src)
        K = self.K_linear(tgt)
        V = self.V_linear(tgt)

        out = self.attention(Q, K, V, mask)

        out = out.transpose(0, 1).contiguous()  # (l, b, c)

        return out


class AttentionLayer(nn.Module):
    def __init__(self, ninp, nhid, dropout):
        super().__init__()

        self.dropout = nn.Dropout(dropout)

        self.sequnetial0 = nn.Sequential(
            nn.Linear(ninp, nhid),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(nhid, ninp),
            nn.Dropout(dropout),
        )

        self.ln0 = nn.LayerNorm(ninp, eps=1e-5)
        self.ln1 = nn.LayerNorm(ninp, eps=1e-5)


class SelfAttentionModule_local(nn.Module):
    def __init__(self, ninp, nhead, nhid, dropout, nlayers):
        super().__init__()

        self.layers = nn.ModuleList([
            self.SelfAttentionLayer(ninp, nhead, nhid, dropout)
            for _ in range(nlayers)
        ])

    class SelfAttentionLayer(AttentionLayer):
        def __init__(self, ninp, nhead, nhid, dropout):
            super().__init__(ninp, nhid, dropout)

            self.attention = SelfAttention(ninp, nhead)

        def forward(self, src, mask):
            out = self.dropout(self.attention(src, mask))
            src = self.ln0(src + out)

            out = self.sequnetial0(src)
            src = self.ln1(src + out)

            return src

    def forward(self, src, mask):
        len_local_window = 256
        window = src.shape[0]//len_local_window
        src_window_list = []
        src_window_list_out = []
        for i in range(window):
            src_window_list.append(src[i*len_local_window:(i+1)*len_local_window])
        for j in range(len(src_window_list)):
            src = src_window_list[j]
            for layer in self.layers:
                src = layer(src, mask)
            src_window_list_out.append(src)

        src_out = torch.cat(src_window_list_out,dim=0)


        return src_out


class CrossAttentionModule(nn.Module):
    def __init__(self, ninp, nhead, nhid, dropout, nlayers):
        super().__init__()

        self.layers = nn.ModuleList([
            self.CrossAttentionLayer(ninp, nhead, nhid, dropout)
            for _ in range(nlayers)
        ])

    class CrossAttentionLayer(AttentionLayer):
        def __init__(self, ninp, nhead, nhid, dropout):
            super().__init__(ninp, nhid, dropout)

            self.attention = CrossAttention(ninp, nhead)

        def forward(self, src, tgt, mask):
            out = self.dropout(self.attention(src, tgt, mask))
            src = self.ln0(src + out)

            out = self.sequnetial0(src)
            src = self.ln1(src + out)

            return src

    def forward(self, src, tgt, mask):
        for layer in self.layers:
            src = layer(src, tgt, mask)

        return src
