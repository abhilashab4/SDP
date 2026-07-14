import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer

class MultiModalDefectDataset(Dataset):
    def __init__(self, df, feature_cols, label_col, text_cols, tokenizer=None, vocab_size=8000, max_len=128, mean=None, std=None):
        # 1. Process Tabular Metrics safely
        metrics = df[feature_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
        
        if mean is None or std is None:
            self.mean = metrics.mean()
            self.std = metrics.std() + 1e-8
        else:
            self.mean = mean
            self.std = std
            
        normalized = (metrics - self.mean) / self.std
        
        # Cross-Version Drift Protection: Clip outliers between [-5, 5] 
        # Prevents extreme feature shifts in newer releases from blowing up VAE latent spaces
        clipped_values = np.clip(normalized.values, -5.0, 5.0)
        self.X_metrics = torch.as_tensor(clipped_values, dtype=torch.float32)
        
        # 2. Sequence processing (AST + Code Tokens mixed)
        combined_text = df[text_cols].astype(str).agg(" ".join, axis=1).tolist()
        
        # Train or configure the BPE Subword Tokenizer
        if tokenizer is None:
            self.tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
            # FIX: Dynamically pass vocab_size from config to prevent dead embedding parameters
            trainer = BpeTrainer(special_tokens=["[UNK]", "[PAD]"], vocab_size=vocab_size)
            self.tokenizer.train_from_iterator(combined_text, trainer)
        else:
            self.tokenizer = tokenizer 
            
        # Explicitly re-enable padding & truncation rules 
        # Tokenizer references lose these state details when passed across validation/test splits
        self.tokenizer.enable_padding(pad_id=1, pad_token="[PAD]", length=max_len)
        self.tokenizer.enable_truncation(max_length=max_len)
            
        # Encode sequences cleanly to uniform subwords matrix
        encodings = self.tokenizer.encode_batch(combined_text)
        self.X_seq = torch.as_tensor([e.ids for e in encodings], dtype=torch.long)
        
        # 3. Binary Label Matrix Setup
        # FIX: Force labels to long integers (int64) to ensure exact matrix matching 
        # inside your Supervised Contrastive Loss calculations.
        self.y = torch.as_tensor((df[label_col] > 0).astype(int).values, dtype=torch.long)

    def get_pos_weight(self):
        num_clean = (self.y == 0).sum().item()
        num_bug = (self.y == 1).sum().item()
        return torch.tensor([num_clean / max(num_bug, 1)], dtype=torch.float32)

    def __len__(self): 
        return len(self.y)
        
    def __getitem__(self, idx): 
        return self.X_metrics[idx], self.X_seq[idx], self.y[idx]
    