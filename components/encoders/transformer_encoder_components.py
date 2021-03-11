import math

import torch
from torch import nn


class TransformerEncoderBlock(nn.Module):

    def __init__(self, num_items, embedding_size, ):
        super(TransformerEncoderBlock, self).__init__()
        self.embedding_size = embedding_size
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_size,
                                                                    nhead=1,
                                                                    dim_feedforward=embedding_size * 2, )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer=self.transformer_encoder_layer,
                                                         num_layers=1,
                                                         )
        self.positional_encoding = PositionalEncoding(d_model=embedding_size)

    def forward(self, embedded_sequences, mask):
        out = embedded_sequences
        # mask = mask.permute(1, 0)
        out = embedded_sequences * math.sqrt(self.embedding_size)
        out = self.positional_encoding(out)
        out = out.permute(1, 0, 2)
        out = self.transformer_encoder(out)

        return out


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=6500):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)