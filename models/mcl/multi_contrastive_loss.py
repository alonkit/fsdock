from torch import nn
import torch

class AttnProjection(nn.Module):
    def __init__(self, d_model, out_dim):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, 1)
        )
        self.proj = nn.Sequential(
            nn.Linear(d_model, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim)
        )
        self.softmax = nn.Softmax(dim=0)
    def forward(self, x, mask=None, lengths=None): 
        # x: (T, B, D)
        # mask: optional padding mask, shape (T, B) or (B, T). Non-zero/True = valid tokens.
        attn_logits = self.attn(x).squeeze(-1)   # (T, B)
        if lengths is not None:
            mask = self.make_mask(lengths).to(x.device)
        if mask is not None:
            mask_bool = mask.to(torch.bool)
            # set padded positions to a large negative so softmax ~ 0 there
            attn_logits = attn_logits.masked_fill(~mask_bool, -1e9)
        attn_weights = self.softmax(attn_logits)  # (T, B)
        attn_weights = attn_weights.unsqueeze(-1) # (T, B, 1)
        pooled = (attn_weights * x).sum(dim=0)    # (B, D)
        return self.proj(pooled)
    
    def make_mask(self, lengths:list):
        # lengths: list of sequence lengths, size B
        max_len = max(lengths)
        batch_size = len(lengths)
        mask = torch.zeros((max_len, batch_size), dtype=torch.bool)
        for i, l in enumerate(lengths):
            mask[:l, i] = 1
        return mask

class MultiContrastiveLoss(nn.Module):
    def __init__(self):
        super(MultiContrastiveLoss, self).__init__()
        self.cross_entropy = nn.CrossEntropyLoss()
    
    def forward(self, v1, v2, good_labels, temperature=0.07):
        # v1, v2: (B, D), normalized
        # what vectors to ignore (the bad ones i mean)
        logits = v1 @ v2.T / temperature           # (B, B)
        labels = torch.arange(v1.size(0)).to(v1.device)
        loss_i = self.cross_entropy(logits[good_labels], labels[good_labels])
        loss_j = self.cross_entropy(logits.T[good_labels], labels[good_labels])
        return (loss_i + loss_j) / 2
    
    