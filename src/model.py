import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerAnchoredVAE(nn.Module):
    def __init__(self, metrics_dim, vocab_size, embed_dim=128, latent_dim=64, num_heads=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # Positional Embeddings for the Transformer
        self.pos_embedding = nn.Parameter(torch.zeros(1, 128, embed_dim)) 
        
        # Structural Semantic Anchoring Projection
        self.metrics_to_embed = nn.Sequential(
            nn.Linear(metrics_dim, embed_dim),
            nn.Tanh() 
        )
        
        # CPU-Optimized Shallow Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=128, 
            dropout=0.2, batch_first=True, activation='gelu'
        )
        self.transformer_enc = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        self.metrics_enc = nn.Sequential(
            nn.Linear(metrics_dim, 64),
            nn.ReLU()
        )
        
        # --- NOVELTY UPGRADE: Multi-Head Cross-Attention Bridge ---
        # Projects metrics to match embed_dim for multi-head attention querying
        self.metrics_query_proj = nn.Linear(64, embed_dim)
        self.cross_attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        
        # --- NOVELTY UPGRADE: Gated Multimodal Fusion (GMF) ---
        # Projects sequence context to match the 64-dim metrics space
        self.context_proj = nn.Linear(embed_dim, 64)
        # The gate layer decides the dynamic routing ratio between modalities
        self.gate = nn.Sequential(
            nn.Linear(64 + 64, 64),
            nn.Sigmoid()
        )
        self.fc_fuse = nn.Linear(64, 64) 
        
        # Latent space Space
        self.mu = nn.Linear(64, latent_dim)
        self.logvar = nn.Linear(64, latent_dim)
        
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 32), 
            nn.ReLU(), 
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def forward(self, x_metrics, x_seq):
        batch_size, seq_len = x_seq.size()
        
        # 1. Base Token Embeddings + Positional Context
        token_embeddings = self.embedding(x_seq) + self.pos_embedding[:, :seq_len, :]
        
        # Dynamic Anchoring
        metrics_anchor = self.metrics_to_embed(x_metrics).unsqueeze(1) 
        anchored_embeddings = token_embeddings + metrics_anchor 
        
        # 2. Process via Self-Attention Transformer pass
        trans_out = self.transformer_enc(anchored_embeddings) 
        
        # 3. Process Tabular Metrics 
        h_metrics = self.metrics_enc(x_metrics) 
        
        # 4. Upgrade: Multi-Head Cross-Attention Bridge
        # Query: Projected Metrics [B, 1, embed_dim]
        # Key/Value: Transformer Sequence Output [B, seq_len, embed_dim]
        query = self.metrics_query_proj(h_metrics).unsqueeze(1)
        attn_output, _ = self.cross_attention(query=query, key=trans_out, value=trans_out)
        context = attn_output.squeeze(1) # Shape: [Batch, embed_dim]
        
        # 5. Upgrade: Gated Multimodal Fusion (GMF)
        context_mapped = F.relu(self.context_proj(context)) # Match dimensions [Batch, 64]
        
        # Compute dynamic gating vector based on both modalities
        gate_input = torch.cat((context_mapped, h_metrics), dim=-1)
        g = self.gate(gate_input) # Values strictly between 0 and 1
        
        # Adaptively blend features based on the gate's confidence
        fused = g * context_mapped + (1 - g) * h_metrics
        h_shared = F.relu(self.fc_fuse(fused))
        
        # 6. Bottleneck & Inference
        mu = self.mu(h_shared)
        logvar = self.logvar(h_shared)
        z = self.reparameterize(mu, logvar)
        logits = self.classifier(z)
        
        return logits, mu, logvar