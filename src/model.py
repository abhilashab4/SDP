
import torch
import torch.nn as nn

class MultiModalVAE(nn.Module):
    def __init__(self, metrics_dim, vocab_size, embed_dim=64, latent_dim=32):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # LSTM Configuration
        self.lstm_hidden = 64
        self.seq_enc = nn.LSTM(embed_dim, self.lstm_hidden, batch_first=True, bidirectional=True)
        
        # Metrics Encoder
        self.metrics_enc = nn.Sequential(
            nn.Linear(metrics_dim, 64),
            nn.ReLU()
        )

        # Fusion: 2 * 64 (BiLSTM) + 64 (Metrics) = 192
        self.fc_fuse = nn.Linear((2 * self.lstm_hidden) + 64, 64) 
        
        # Latent space mapping
        self.mu = nn.Linear(64, latent_dim)
        self.logvar = nn.Linear(64, latent_dim)

        # Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 16), 
            nn.ReLU(), 
            nn.Linear(16, 1)
        )

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            # During evaluation, drop the stochastic noise for stable predictions
            return mu

    def forward(self, x_metrics, x_seq):
        # 1. Process Text Sequence via BiLSTM
        # embed shape: [batch_size, seq_len, embed_dim]
        embedded = self.embedding(x_seq) 
        _, (h_n, _) = self.seq_enc(embedded)
        
        # h_n shape: [2, batch_size, lstm_hidden]
        # Robustly extract and concatenate last forward and backward states
        h_forward = h_n[-2, :, :]
        h_backward = h_n[-1, :, :]
        h_seq = torch.cat((h_forward, h_backward), dim=-1) # Shape: [batch_size, 128]
        
        # 2. Process Tabular Metrics
        h_metrics = self.metrics_enc(x_metrics) # Shape: [batch_size, 64]
        
        # 3. Multimodal Fusion
        fused = torch.cat((h_seq, h_metrics), dim=-1) # Shape: [batch_size, 192]
        h_shared = torch.relu(self.fc_fuse(fused))
        
        # 4. Latent Space Bottleneck
        mu = self.mu(h_shared)
        logvar = self.logvar(h_shared)
        z = self.reparameterize(mu, logvar)
        
        # 5. Prediction output
        logits = self.classifier(z)
        
        # Return mu and logvar so your training loop can calculate the KL-Divergence Loss
        return logits, mu, logvar