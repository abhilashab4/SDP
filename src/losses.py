import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss
    Reference:
    https://arxiv.org/abs/2004.11362

    Usage:
        loss = SupConLoss()(features, labels)

    features : [batch_size, latent_dim]
    labels   : [batch_size]
    """

    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):

        device = features.device

        batch_size = features.shape[0]

        # Normalize latent vectors
        features = F.normalize(features, dim=1)

        # Similarity matrix
        similarity = torch.matmul(features, features.T)
        similarity = similarity / self.temperature

        # Numerical stability
        logits_max, _ = torch.max(similarity, dim=1, keepdim=True)
        logits = similarity - logits_max.detach()

        labels = labels.contiguous().view(-1, 1)

        mask = torch.eq(labels, labels.T).float().to(device)

        # Remove self-comparisons
        logits_mask = torch.ones_like(mask)
        logits_mask.fill_diagonal_(0)

        mask = mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask

        log_prob = logits - torch.log(
            exp_logits.sum(dim=1, keepdim=True) + 1e-8
        )

        positive_count = mask.sum(dim=1)

        positive_count = torch.where(
            positive_count == 0,
            torch.ones_like(positive_count),
            positive_count,
        )

        mean_log_prob_pos = (
            mask * log_prob
        ).sum(dim=1) / positive_count

        loss = -mean_log_prob_pos.mean()

        return loss