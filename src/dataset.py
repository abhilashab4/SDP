
import torch
import pandas as pd
from torch.utils.data import Dataset
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

class MultiModalDefectDataset(Dataset):
    def __init__(self, df, feature_cols, label_col, text_cols, tokenizer=None, max_len=128, mean=None, std=None):
        """
        Args:
            df (pd.DataFrame): The source dataframe.
            feature_cols (list): List of column names for tabular metrics.
            label_col (str): The name of the target/bug column.
            text_cols (list): List of column names containing text/tokens to combine.
            tokenizer (Tokenizer, optional): Pre-fitted Keras tokenizer.
            max_len (int): Maximum sequence length for padding.
            mean (pd.Series, optional): Training mean for normalization.
            std (pd.Series, optional): Training std for normalization.
        """
        # 1. Process Tabular Metrics safely
        metrics = df[feature_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
        
        if mean is None or std is None:
            self.mean = metrics.mean()
            self.std = metrics.std() + 1e-8
        else:
            self.mean = mean
            self.std = std
            
        normalized = (metrics - self.mean) / self.std
        self.X_metrics = torch.as_tensor(normalized.values, dtype=torch.float32)
        
        # 2. Sequence processing (AST + Code Tokens)
        combined_text = df[text_cols].astype(str).agg(" ".join, axis=1)
        
        if tokenizer is None:
            self.tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
            self.tokenizer.fit_on_texts(combined_text)
        else:
            self.tokenizer = tokenizer 
            
        seqs = self.tokenizer.texts_to_sequences(combined_text)
        self.X_seq = torch.as_tensor(pad_sequences(seqs, maxlen=max_len), dtype=torch.long)
        
        # 3. Binary Label
        self.y = torch.as_tensor((df[label_col] > 0).astype(int).values, dtype=torch.float32)

    def get_pos_weight(self):
        """Helper to get pos_weight specifically for the training split"""
        num_clean = (self.y == 0).sum().item()
        num_bug = (self.y == 1).sum().item()
        return torch.tensor([num_clean / max(num_bug, 1)], dtype=torch.float32)

    def __len__(self): 
        return len(self.y)
        
    def __getitem__(self, idx): 
        return self.X_metrics[idx], self.X_seq[idx], self.y[idx]