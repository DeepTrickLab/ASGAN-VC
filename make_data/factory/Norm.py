import torch
import torch.nn as nn


class ConvNorm(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=5,
        stride=1,
        padding=None,
        dilation=1,
        bias=True,
        w_init_gain="linear",
    ):
        super(ConvNorm, self).__init__()
        if padding is None:
            assert kernel_size % 2 == 1
            padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )

        nn.init.xavier_uniform_(
            self.conv.weight, gain=nn.init.calculate_gain(w_init_gain)
        )

    def forward(self, signal):
        conv_signal = self.conv(signal)
        return conv_signal


class LinearNorm(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, w_init_gain="linear"):
        super(LinearNorm, self).__init__()
        self.linear_layer = nn.Linear(in_dim, out_dim, bias=bias)

        nn.init.xavier_uniform_(
            self.linear_layer.weight, gain=nn.init.calculate_gain(w_init_gain)
        )

    def forward(self, x):
        return self.linear_layer(x)


class GroupNorm(nn.GroupNorm):
    """
    Group Normalization with 1 group.
    Input: tensor in shape [B, C, H, W]
    """

    def __init__(self, num_channels, **kwargs):
        super().__init__(1, num_channels, **kwargs)


class PatchEmbed(nn.Module):
    """
    Patch Embedding that is implemented by a layer of conv.
    Input: tensor in shape [B, C, H, W]
    Output: tensor in shape [B, C, H/stride, W/stride]
    """

    def __init__(
        self,
        patch_size=5,
        stride=1,
        padding=2,
        in_chans=336,
        embed_dim=512,
    ):
        super().__init__()
        self.proj = nn.Conv1d(
            in_chans, embed_dim, kernel_size=patch_size, stride=stride, padding=padding
        )
        self.norm = nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        return x


class AdaIN(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()

    def forward(self, content, mus, stds):
        size = content.size()
        in_cs = (content - content.mean().expand(size)) / content.std()
        tmp = []
        for in_c, mu, std in zip(in_cs, mus, stds):
            tmp.append(((in_c * std) + mu).unsqueeze(0))
        return torch.cat(tmp, dim=0)


class IN(nn.Module):
    def __init__(
        self,
    ):

        super().__init__()

    def forward(self, content):
        size = content.size()
        mu = content.mean().expand(size)
        std = content.std()
        return ((content - mu) / std), mu, std


class Modulated_Conv1D(nn.Module):
    def __init__(self, in_chans=80, out_chans=80, kernel_size=5, padding=2):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.randn([in_chans, out_chans, kernel_size])
        )
        self.convnorm = ConvNorm(
            in_chans, out_chans, kernel_size=kernel_size, padding=padding
        )

    def forward(self, x, mus, std):
        batch_size = x.shape[0]
        w = self.weight.unsqueeze(0)  # NOIK
        w = w * std.reshape(batch_size, -1, 1, 1)
        dcoefs = ((w ** 2).sum(dim=[2, 3]) + 1e-8).rsqrt()  # NO
        w = w * dcoefs.reshape(batch_size, -1, 1, 1)  # NOIK
        self.convnorm.conv.weight = self.weight = torch.nn.Parameter(w.mean(dim=0))
        outputs = self.convnorm(x)
        tmp = []
        for output, mu in zip(outputs, mus):
            tmp.append((output + mu).unsqueeze(0))
        return torch.cat(tmp, dim=0)
