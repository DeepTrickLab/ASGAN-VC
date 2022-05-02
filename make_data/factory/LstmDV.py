import torch.nn as nn


class LstmDV(nn.Module):
    def __init__(
        self, num_classes, num_layers=3, dim_input=80, dim_cell=768, dim_emb=256
    ):
        super(LstmDV, self).__init__()
        self.lstm = nn.LSTM(
            input_size=dim_input,
            hidden_size=dim_cell,
            num_layers=num_layers,
            batch_first=True,
        )
        self.embedding = nn.Linear(dim_cell, dim_emb)
        self.output = nn.Linear(dim_emb, num_classes)

    def forward(self, x):
        self.lstm.flatten_parameters()
        lstm_out, _ = self.lstm(x)
        embeds = self.embedding(lstm_out[:, -1, :])
        norm = embeds.norm(p=2, dim=-1, keepdim=True)
        embeds_normalized = embeds.div(norm)
        predictions = self.output(embeds)
        return predictions, embeds_normalized
