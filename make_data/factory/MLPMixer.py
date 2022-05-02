import torch.nn as nn
from einops.layers.torch import Rearrange
from functools import partial


def FeedForward(dim, expansion_factor=4, dropout=0.0, dense=nn.Linear):
    return nn.Sequential(
        dense(dim, int(dim * expansion_factor)),
        nn.GELU(),
        nn.Dropout(dropout),
        dense(int(dim * expansion_factor), dim),
        nn.Dropout(dropout),
    )


class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x)) + x


def MLPMixer(
    in_chans,
    out_chans,
    seq_len,
    depth=1,
    kernel_size=5,
    padding=2,
    expansion_factor=1,
    dropout=0.0,
):
    chan_first, chan_last = (
        partial(nn.Conv1d, kernel_size=1),
        partial(nn.Conv1d, kernel_size=kernel_size, padding=padding),
    )
    return nn.Sequential(
        nn.Linear(seq_len, seq_len),
        *[
            nn.Sequential(
                PreNormResidual(
                    seq_len,
                    FeedForward(in_chans, expansion_factor, dropout, chan_first),
                ),
                PreNormResidual(
                    seq_len, FeedForward(in_chans, expansion_factor, dropout, chan_last)
                ),
            )
            for _ in range(depth)
        ],
        nn.Conv1d(in_chans, out_chans, kernel_size=kernel_size, padding=padding),
    )


class MLPBlock(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, act_layer=nn.GELU):
        super().__init__()
        self.fc1 = nn.Conv1d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv1d(hidden_features, out_features, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x
