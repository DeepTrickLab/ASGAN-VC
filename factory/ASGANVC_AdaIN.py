import torch
import torch.nn as nn
import torch.nn.functional as F
from .Norm import ConvNorm, LinearNorm, AdaIN


class Encoder(nn.Module):
    """Encoder module:"""

    def __init__(self, dim_neck, dim_emb, freq):
        super(Encoder, self).__init__()
        self.dim_neck = dim_neck
        self.freq = freq
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

        convolutions = []
        for i in range(3):
            conv_layer = nn.Sequential(
                ConvNorm(
                    80 + dim_emb if i == 0 else 512,
                    512,
                    kernel_size=5,
                    stride=1,
                    padding=2,
                    dilation=1,
                    w_init_gain="relu",
                ),
                nn.BatchNorm1d(512),
            )
            convolutions.append(conv_layer)
        self.convolutions = nn.ModuleList(convolutions)

        self.lstm = nn.LSTM(512, dim_neck, 2, batch_first=True, bidirectional=True)

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
        # Now put the embeding back
        c_org = c_org.unsqueeze(-1).expand(-1, -1, x.size(-1))
        x = torch.cat((x, c_org), dim=1)
        for conv in self.convolutions:
            x = F.relu(conv(x))
        x = x.transpose(1, 2)
        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)
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


class Decoder(nn.Module):
    def __init__(self, dim_neck, dim_emb, dim_pre):
        super(Decoder, self).__init__()
        self.lstm1 = nn.LSTM(dim_neck * 2 + dim_emb, dim_pre, 1, batch_first=True)
        convolutions = []
        for _ in range(3):
            conv_layer = nn.Sequential(
                ConvNorm(
                    dim_pre,
                    dim_pre,
                    kernel_size=5,
                    stride=1,
                    padding=2,
                    dilation=1,
                    w_init_gain="relu",
                ),
                nn.BatchNorm1d(dim_pre),
            )
            convolutions.append(conv_layer)
        self.convolutions = nn.ModuleList(convolutions)
        self.lstm2 = nn.LSTM(dim_pre, 1024, 2, batch_first=True)
        self.linear_projection = LinearNorm(1024, 80)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x = x.transpose(1, 2)
        for conv in self.convolutions:
            x = F.relu(conv(x))
        x = x.transpose(1, 2)
        outputs, _ = self.lstm2(x)
        decoder_output = self.linear_projection(outputs)
        return decoder_output


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


class ASGANVC_AdaIN(nn.Module):
    def __init__(self, dim_neck, dim_emb, dim_pre, freq):
        super(ASGANVC_AdaIN, self).__init__()
        self.encoder = Encoder(dim_neck, dim_emb, freq)
        self.decoder = Decoder(dim_neck, dim_emb, dim_pre)
        self.postnet = Postnet()

    def forward(
        self, x, c_org, c_trg, target_feature=None,
    ):

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
        mel_outputs = self.decoder(encoder_outputs)
        if target_feature is not None:
            mel_outputs_postnet = self.postnet(
                mel_outputs.transpose(2, 1), target_feature
            )
        else:
            mel_outputs_postnet = self.postnet(mel_outputs.transpose(2, 1), features)

        mel_outputs_postnet = mel_outputs + mel_outputs_postnet.transpose(2, 1)
        mel_outputs = mel_outputs.unsqueeze(1)
        mel_outputs_postnet = mel_outputs_postnet.unsqueeze(1)

        return mel_outputs, mel_outputs_postnet, torch.cat(codes, dim=-1)
