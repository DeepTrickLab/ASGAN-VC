import torch
import torch.nn as nn
from .Norm import ConvNorm, GroupNorm, PatchEmbed, LinearNorm, AdaIN
from .MLPMixer import MLPMixer, MLPBlock


class Metablock(nn.Module):
    def __init__(
        self, dim, token_mixer, norm_layer=GroupNorm,
    ):

        super().__init__()
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.token_mixer = token_mixer
        self.mlp = MLPBlock(dim, dim, dim)

    def forward(self, x):
        x = x + self.token_mixer(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class Encoder(nn.Module):
    def __init__(self, dim_emb, dim_neck, freq, num_layers=3):
        super().__init__()
        self.freq = freq
        self.dim_neck = dim_neck
        """
        Pre Extract feature first here
        """
        feature_pre_extract = []
        for i in range(3):
            conv_layer = nn.Sequential(
                ConvNorm(
                    80,
                    80,
                    kernel_size=5,
                    stride=1,
                    padding=2,
                    dilation=1,
                    w_init_gain="linear",
                ),
                nn.BatchNorm1d(80),
            )
            feature_pre_extract.append(conv_layer)
        self.feature_pre_extract = nn.ModuleList(feature_pre_extract)
        self.embedding = PatchEmbed(in_chans=dim_emb + 80, embed_dim=2 * dim_emb)
        metablock = []
        for _ in range(num_layers):
            metablock.append(
                Metablock(
                    dim=2 * dim_emb,
                    token_mixer=nn.Sequential(
                        ConvNorm(
                            2 * dim_emb,
                            2 * dim_emb,
                            kernel_size=5,
                            padding=2,
                            w_init_gain="relu",
                        ),
                        nn.BatchNorm1d(2 * dim_emb),
                        nn.ReLU(),
                    ),
                )
            )
        self.metablock = nn.Sequential(*metablock)
        self.down_mlp_1 = MLPMixer(in_chans=512, out_chans=256,seq_len=128, depth=1)
        self.down_mlp_2 = MLPMixer(in_chans=256, out_chans=128, seq_len=128,depth=1)
        self.down_mlp_3 = MLPMixer(in_chans=128, out_chans=2 * dim_neck, seq_len=128,depth=1)

    def forward(self, x, c_org):

        x = x.squeeze(1).transpose(2, 1)
        # Here we pre-extract feature here!
        features = []
        for feature_layer in self.feature_pre_extract:
            x = feature_layer(x)
            features.append(
                [
                    x.view(x.size(0), -1).mean(1, keepdim=True).unsqueeze(-1),
                    x.view(x.size(0), -1).std(1, keepdim=True).unsqueeze(-1),
                ]
            )
        c_org = c_org.unsqueeze(-1).expand(-1, -1, x.size(-1))
        x = torch.cat((x, c_org), dim=1)
        x = self.embedding(x)
        for metablock in self.metablock:
            x = metablock(x)
        x = self.down_mlp_1(x)
        x = self.down_mlp_2(x)
        x = self.down_mlp_3(x)
        outputs = x.transpose(1, 2)
        out_forward = outputs[:, :, : self.dim_neck]
        out_backward = outputs[:, :, self.dim_neck :]
        codes = []
        for i in range(0, outputs.size(1), self.freq):
            codes.append(
                torch.cat(
                    (out_forward[:, i + self.freq - 1, :], out_backward[:, i, :]),
                    dim=-1,
                )
            )
        return codes, features


class Postnet(nn.Module):
    """
    Postnet
    - Five 1-d convolution with 512 channels and kernel size 5
    """

    def __init__(self):
        super(Postnet, self).__init__()
        self.convolutions = nn.ModuleList()

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(
                    80,
                    512,
                    kernel_size=5,
                    stride=1,
                    padding=2,
                    dilation=1,
                    w_init_gain="tanh",
                ),
                nn.BatchNorm1d(512),
            )
        )

        for _ in range(1, 5 - 1):
            self.convolutions.append(
                nn.Sequential(
                    ConvNorm(
                        512,
                        512,
                        kernel_size=5,
                        stride=1,
                        padding=2,
                        dilation=1,
                        w_init_gain="tanh",
                    ),
                    nn.BatchNorm1d(512),
                )
            )

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(
                    512,
                    80,
                    kernel_size=5,
                    stride=1,
                    padding=2,
                    dilation=1,
                    w_init_gain="linear",
                ),
                nn.BatchNorm1d(80),
            )
        )
        self.adain = AdaIN()
        feature_last_combine = []
        for _ in range(3):
            conv_layer = nn.Sequential(
                ConvNorm(
                    80,
                    80,
                    kernel_size=5,
                    stride=1,
                    padding=2,
                    dilation=1,
                    w_init_gain="linear",
                ),
            )
            feature_last_combine.append(conv_layer)
        self.feature_last_combine = nn.ModuleList(feature_last_combine)

    def forward(self, x, features):
        for i in range(len(self.convolutions) - 1):
            x = torch.tanh(self.convolutions[i](x))

        x = self.convolutions[-1](x)
        for combine_layer, features in zip(self.feature_last_combine, features):
            x = self.adain(x, features[0], features[1])
            x = combine_layer(x)

        return x


class Decoder(nn.Module):
    """Decoder module:"""

    def __init__(self, dim_neck, dim_emb, dim_pre, num_layers=3):
        super(Decoder, self).__init__()
        self.embedding = PatchEmbed(in_chans=dim_emb + 2 * dim_neck, embed_dim=dim_pre)
        metablock = []
        for _ in range(num_layers):
            metablock.append(
                Metablock(
                    dim=dim_pre,
                    token_mixer=nn.Sequential(
                        ConvNorm(
                            2 * dim_emb,
                            2 * dim_emb,
                            kernel_size=5,
                            padding=2,
                            w_init_gain="relu",
                        ),
                        nn.BatchNorm1d(2 * dim_emb),
                        nn.ReLU(),
                    ),
                )
            )
        self.metablock = nn.Sequential(*metablock)
        self.up_mlp_1 = MLPMixer(in_chans=512, out_chans=1024, seq_len=128,depth=3)
        self.up_mlp_2 = MLPMixer(in_chans=1024, out_chans=1024,seq_len=128, depth=3)
        self.linear = LinearNorm(2 * dim_pre, 80)
        self.posnet = Postnet()

    def forward(self, x, features):
        x = x.transpose(1, 2)
        x = self.embedding(x)
        for metablock in self.metablock:
            x = metablock(x)
        x = self.up_mlp_1(x)
        x = self.up_mlp_2(x)
        x = x.transpose(1, 2)
        x = self.linear(x)
        mel_outputs_postnet = x + self.posnet(x.transpose(1, 2), features).transpose(
            1, 2
        )
        return x, mel_outputs_postnet


class MetaVC2(nn.Module):
    def __init__(self, dim_neck, dim_emb, dim_pre, freq):
        super(MetaVC2, self).__init__()
        self.encoder = Encoder(dim_emb, dim_neck, freq)
        self.decoder = Decoder(dim_neck, dim_emb, dim_pre)

    def forward(self, x, c_org, c_trg, target_feature=None):

        codes, features = self.encoder(x, c_org)
        if c_trg is None and target_feature is None:
            return torch.cat(codes, dim=-1), features

        tmp = []
        for code in codes:
            tmp.append(code.unsqueeze(1).expand(-1, int(x.size(1) / len(codes)), -1))
        code_exp = torch.cat(tmp, dim=1)
        encoder_outputs = torch.cat(
            (code_exp, c_trg.unsqueeze(1).expand(-1, x.size(1), -1)), dim=-1
        )
        if target_feature is not None:
            mel_outputs, mel_outputs_postnet = self.decoder(
                encoder_outputs, target_feature
            )
        else:
            mel_outputs, mel_outputs_postnet = self.decoder(encoder_outputs, features)

        mel_outputs = mel_outputs.unsqueeze(1)
        mel_outputs_postnet = mel_outputs_postnet.unsqueeze(1)
        return mel_outputs, mel_outputs_postnet, torch.cat(codes, dim=-1)
