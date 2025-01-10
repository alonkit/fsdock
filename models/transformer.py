import torch
import torch.nn as nn
import math
from datetime import datetime




class PositionalEncoding(nn.Module):

    def __init__(self, embedding_dim: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2) * (-math.log(10000.0) / embedding_dim))
        pe = torch.zeros(max_len, 1, embedding_dim)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class TransformerEncoder(nn.Module):
    def __init__(self, vocab_dim, embedding_dim, hidden_size, nhead, n_layers, max_length, pad_token, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.seq_len = max_length
        self.positional_encoding = PositionalEncoding(embedding_dim, max_len=self.seq_len)
        self.token_embedding = nn.Embedding(vocab_dim, embedding_dim)
        enc_layer = nn.TransformerEncoderLayer(embedding_dim, nhead, hidden_size, dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.pad_token = pad_token

    def create_src_key_padding_mask(self, src):
        return (src == self.pad_token)

    def forward(self, src, src_key_padding_mask):
        
        embedded_src = self.token_embedding(src.permute(1, 0))
        final_src = self.positional_encoding(embedded_src)
        output = self.encoder(final_src.permute(1, 0, 2), src_key_padding_mask=src_key_padding_mask)
        return output



class TransformerDecoder(nn.Module):
    def __init__(self, vocab_dim, embedding_dim, hidden_size, nhead, n_layers, max_length,pad_token, dropout=0.1, ):
        super(TransformerDecoder, self).__init__()
        self.seq_len = max_length
        self.positional_encoding = PositionalEncoding(embedding_dim, max_len=self.seq_len)
        self.token_embedding = nn.Embedding(vocab_dim, embedding_dim)
        dec_layer = nn.TransformerDecoderLayer(embedding_dim, nhead, hidden_size, dropout, batch_first=True)
        self.decoder : nn.TransformerDecoder = nn.TransformerDecoder(dec_layer, num_layers=n_layers)
        self.dense = nn.Linear(embedding_dim, vocab_dim)
        self.pad_id = pad_token

    def _generate_square_subsequent_mask(self, sz, device):
        mask = (torch.triu(torch.ones((sz, sz), device=device), diagonal=0) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


    def create_target_masks(self, tgt):
        tgt_seq_len = tgt.shape[1]
        tgt_mask = self._generate_square_subsequent_mask(tgt_seq_len, tgt.device)
        tgt_padding_mask = (tgt == self.pad_id)
        return tgt_mask, tgt_padding_mask

    def forward(self, target, memory, target_mask, target_padding_mask,memory_key_padding_mask=None, memory_mask=None):
        embedded_target = self.token_embedding(target.permute(1, 0))
        final_target = self.positional_encoding(embedded_target)
        output = self.decoder(final_target.permute(1, 0, 2), memory,
                              tgt_mask=target_mask, tgt_key_padding_mask=target_padding_mask,
                              memory_key_padding_mask=memory_key_padding_mask, memory_mask=memory_mask)
        logits = self.dense(output)
        return logits

