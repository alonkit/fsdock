import torch
import torch.nn as nn

class ConditionalPriorCVAE(nn.Module):
    def __init__(self, d_model, d_latent):
        super().__init__()
        
        hidden_dim = d_model * 2
        dropout = 0.1
        
        # --- 1. POSTERIOR ENCODER (Training Only) ---
        # Q(Z | Anchor, Fragment)
        self.fragment_encoder = nn.Linear(d_model, d_model) 
        
        self.post_mu_net = nn.Sequential(
            nn.Linear(d_model * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_latent)
        )
        self.post_logvar_net = nn.Sequential(
            nn.Linear(d_model * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_latent)
        )
        
        # --- 2. PRIOR ENCODER (Training & Inference) ---
        # P(Z | Anchor, Input Candidate)
        # We assume candidate_embedding is also of size d_model.
        self.prior_mu_net = nn.Sequential(
            nn.Linear(d_model * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_latent)
        )
        
        self.prior_logvar_net = nn.Sequential(
            nn.Linear(d_model * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_latent)
        )
        
        # --- 3. DECODER COMPONENTS ---
        self.fusion = nn.Linear(d_model + d_latent, d_model)
        self.d_latent = d_latent
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, anchor, candidate_embedding, fragment_embedding=None, training=True):
        """
        anchor: [Batch, d_model] (e.g., pocket + scaffold)
        candidate_embedding: [Batch, d_model] (The current input molecule candidate)
        fragment_embedding: [Batch, d_model] (Summary of the ground truth fragment)
        """
        
        # 1. Always compute the Prior distribution: P(Z | Anchor, Candidate)
        # We concatenate the anchor (context) and the current candidate state
        prior_input = torch.cat([anchor, candidate_embedding], dim=-1)
        prior_mu = self.prior_mu_net(prior_input)
        prior_logvar = self.prior_logvar_net(prior_input)
        
        if training:
            # --- TRAINING PATH ---
            assert fragment_embedding is not None, "fragment_embedding is required during training."
            
            # 2. Compute the Posterior distribution: Q(Z | Anchor, Fragment)
            frag_feat = self.fragment_encoder(fragment_embedding)
            post_input = torch.cat([anchor, frag_feat], dim=-1)
            
            post_mu = self.post_mu_net(post_input)
            post_logvar = self.post_logvar_net(post_input)
            
            # 3. Sample Z from the Posterior
            z = self.reparameterize(post_mu, post_logvar)
            
            # 4. Calculate KLD Loss between Posterior and Prior
            # D_KL( Q(z|x,y) || P(z|x,c) )
            kld_loss = -0.5 * torch.sum(
                1 + post_logvar - prior_logvar 
                - ((post_mu - prior_mu).pow(2) + post_logvar.exp()) / prior_logvar.exp()
            )
            kld_loss = kld_loss / anchor.shape[0]  # Average over batch
            
        else:
            # --- INFERENCE PATH ---
            # We DON'T have the ground truth fragment.
            # We sample Z directly from the learned Prior: P(Z | Anchor, Candidate)
            
            z = self.reparameterize(prior_mu, prior_logvar)
            kld_loss = 0  # No KLD needed during inference

        # --- DECODING (Shared) ---
        # Combine Anchor + sampled Z
        decoder_input = torch.cat([anchor, z], dim=-1)
        decoder_input = self.fusion(decoder_input)
        
        return decoder_input, kld_loss