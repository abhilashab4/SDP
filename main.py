import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import f1_score, precision_recall_curve
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os, copy, random
import pandas as pd
import numpy as np

# Internal Imports
import config as cfg
from src.dataset import MultiModalDefectDataset
from src.model import TransformerAnchoredVAE
from src.utils import save_performance_report

# CPU Multithreading Optimization
torch.set_num_threads(4) 
torch.set_num_interop_threads(4)

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def supervised_contrastive_loss(z, targets, temperature=0.07):
    # Guarantee targets are absolute column matrices [Batch, 1] to prevent broadcasting corruption
    if targets.dim() == 1:
        targets = targets.unsqueeze(1)
        
    z_norm = torch.nn.functional.normalize(z, dim=1)
    similarity_matrix = torch.matmul(z_norm, z_norm.T) / temperature
    labels_equality = torch.eq(targets, targets.T).float()
    identity_mask = torch.eye(targets.shape[0], device=targets.device)
    mask = labels_equality - identity_mask
    logits_max, _ = torch.max(similarity_matrix, dim=1, keepdim=True)
    logits = similarity_matrix - logits_max.detach()
    exp_logits = torch.exp(logits) * (1 - identity_mask)
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)
    mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-8)
    return -mean_log_prob_pos.mean()

seed_everything(42)
os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

print(f"--- Starting Academic-Rigorous Experiment: {cfg.PROJECT_NAME.upper()} ---")
train_paths = [f"data/{cfg.PROJECT_NAME}/{cfg.PROJECT_NAME}_{v}_enriched.csv" for v in cfg.TRAIN_VERSIONS]
train_dfs = [pd.read_csv(p) for p in train_paths if os.path.exists(p)]
combined_train_df = pd.concat(train_dfs).reset_index(drop=True)

metrics_list = [c for c in combined_train_df.columns if c not in ['name', 'ast_seq', 'code_tokens', 'bug']]
metrics_list = metrics_list[:cfg.METRICS_DIM]

# 1. FIXED LEAKAGE STEP: Isolate dataframes using stratified splits BEFORE building any dataset object
train_labels = (combined_train_df['bug'] > 0).astype(int).values
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_indices, val_indices = next(sss.split(np.zeros(len(train_labels)), train_labels))

train_df_clean = combined_train_df.iloc[train_indices].reset_index(drop=True)
val_df_clean = combined_train_df.iloc[val_indices].reset_index(drop=True)

# Build the pure training dataset (this will internally compute clean mean/std stats)
train_ds = MultiModalDefectDataset(
    train_df_clean, feature_cols=metrics_list, label_col='bug', text_cols=['ast_seq', 'code_tokens']
)

# Build the validation dataset using strictly training normalization parameters
val_ds = MultiModalDefectDataset(
    val_df_clean, feature_cols=metrics_list, label_col='bug', text_cols=['ast_seq', 'code_tokens'],
    tokenizer=train_ds.tokenizer, mean=train_ds.mean, std=train_ds.std
)

# Load the completely unseen historical test version dataset
test_path = f"data/{cfg.PROJECT_NAME}/{cfg.PROJECT_NAME}_{cfg.TEST_VERSION}_enriched.csv"
test_ds = MultiModalDefectDataset(
    pd.read_csv(test_path), feature_cols=metrics_list, label_col='bug', text_cols=['ast_seq', 'code_tokens'],
    tokenizer=train_ds.tokenizer, mean=train_ds.mean, std=train_ds.std
)

train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=cfg.BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=cfg.BATCH_SIZE, shuffle=False)

# Instantiates your upgraded model featuring Multi-Head Cross-Attention and GMF
model = TransformerAnchoredVAE(cfg.METRICS_DIM, cfg.VOCAB_SIZE, cfg.EMBED_DIM, cfg.LATENT_DIM).to(cfg.DEVICE)

# Added weight decay (1e-4) to safely regularize cross-attention query matrices during Mixup
optimizer = optim.Adam(model.parameters(), lr=cfg.LR, weight_decay=1e-4)

# 2. FIXED CRITERION STEP: Unweighted loss allows Latent Mixup balancing to work cleanly
criterion = nn.BCEWithLogitsLoss()

best_val_f1 = 0.0
final_optimized_threshold = 0.5
best_model_wts = copy.deepcopy(model.state_dict())

# Early stopping parameters
EARLY_STOP_PATIENCE = 50   
early_stop_counter = 0

print(f"Training for {cfg.EPOCHS} Epochs (Stratified Validation Tuning Active)...")
for epoch in range(cfg.EPOCHS):
    model.train()
    epoch_loss = 0.0
    for m, s, y in train_loader:
        m, s, y = m.to(cfg.DEVICE), s.to(cfg.DEVICE), y.to(cfg.DEVICE).unsqueeze(1)
        optimizer.zero_grad()
        
        logits, mu, logvar = model(m, s)
        logvar = torch.clamp(logvar, min=-10.0, max=10.0)
        
        # --- DYNAMIC MINORITY DETECTION & AUGMENTATION LAYER ---
        num_class_1 = (y == 1).sum().item()
        num_class_0 = (y == 0).sum().item()
        minority_label = 1 if num_class_1 <= num_class_0 else 0
        
        minority_idx = (y == minority_label).nonzero(as_tuple=True)[0]
        
        if len(minority_idx) > 1:
            perm_idx = minority_idx[torch.randperm(len(minority_idx))]
            
            std_minority = torch.exp(0.5 * logvar[minority_idx])
            z_min1 = mu[minority_idx] + torch.randn_like(std_minority) * std_minority
            z_min2 = mu[perm_idx] + torch.randn_like(std_minority) * std_minority
            
            lam = np.random.beta(0.3, 0.7)
            z_syn = lam * z_min1 + (1 - lam) * z_min2
            
            l_syn = model.classifier(z_syn)
            syn_targets = torch.full((len(minority_idx), 1), minority_label, dtype=torch.float, device=cfg.DEVICE)
            
            all_logits = torch.cat([logits, l_syn], dim=0)
            all_targets = torch.cat([y, syn_targets], dim=0)
            all_mu = torch.cat([mu, z_syn], dim=0)
        else:
            all_logits, all_targets = logits, y
            all_mu = mu

        classification_loss = criterion(all_logits, all_targets.float())
        kl_loss = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
        contrastive_loss = supervised_contrastive_loss(all_mu, all_targets)
        
        loss = classification_loss + (cfg.KL_COEFF * kl_loss) + (cfg.CONTRASTIVE_COEFF * contrastive_loss)
        loss.backward()
        
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        epoch_loss += loss.item()

    # 3. VALIDATION TUNING STEP: Scan validation set ONLY to select threshold parameters
    model.eval()
    y_true_val, y_scores_val = [], []
    with torch.no_grad():
        for m_v, s_v, y_v in val_loader:
            l_v, _, _ = model(m_v.to(cfg.DEVICE), s_v.to(cfg.DEVICE))
            scores = torch.sigmoid(l_v).cpu().squeeze(-1).numpy()
            y_scores_val.extend(scores)
            y_true_val.extend(y_v.numpy())
    
    y_true_val = np.array(y_true_val)
    y_scores_val = np.array(y_scores_val)
    
    precisions, recalls, thresholds = precision_recall_curve(y_true_val, y_scores_val)
    f1_scores = (2 * precisions * recalls) / (precisions + recalls + 1e-8)
    best_idx = np.argmax(f1_scores)
    
    epoch_best_val_f1 = f1_scores[best_idx]
    safe_thresh_idx = min(best_idx, len(thresholds) - 1)
    epoch_threshold = thresholds[safe_thresh_idx] if len(thresholds) > 0 else 0.5
    
    if epoch_best_val_f1 > best_val_f1:
        best_val_f1 = epoch_best_val_f1
        final_optimized_threshold = epoch_threshold
        best_model_wts = copy.deepcopy(model.state_dict())
        torch.save(best_model_wts, cfg.MODEL_SAVE_PATH)
        early_stop_counter = 0   # reset patience
    else:
        early_stop_counter += 1
        if early_stop_counter >= EARLY_STOP_PATIENCE:
            print(f"\nEarly stopping triggered at epoch {epoch+1}")
            print(f"Best Validation F1: {best_val_f1:.4f}")
            break
        
    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"Epoch {epoch+1:03d} | Total Loss: {epoch_loss/len(train_loader):.4f} | Val Thresh: {epoch_threshold:.4f} | Val F1: {epoch_best_val_f1:.4f} | Max Val F1: {best_val_f1:.4f}")

# 4. FINAL TEST REPORTING STEP: Evaluate once on the completely untouched test set
print(f"\n📊 Evaluating Locked-In Model on Untouched Test Set...")
model.load_state_dict(best_model_wts)
train_ds.tokenizer.save(cfg.VOCAB_SAVE_PATH)

model.eval()
final_true, final_pred = [], []
all_mu_eval = []

with torch.no_grad():
    for m_t, s_t, y_t in test_loader:
        l_t, mu_t, _ = model(m_t.to(cfg.DEVICE), s_t.to(cfg.DEVICE))
        scores = torch.sigmoid(l_t).cpu().squeeze(-1).numpy()
        
        preds = (scores > final_optimized_threshold).astype(int)
        
        final_pred.extend(preds)
        final_true.extend(y_t.numpy())
        all_mu_eval.append(mu_t.cpu().numpy()) 

test_f1 = f1_score(final_true, final_pred, zero_division=0)
mu_combined = np.concatenate(all_mu_eval, axis=0)
save_performance_report(final_true, final_pred, cfg.PROJECT_NAME)

print("Running t-SNE dimension reduction over Contrastive Space...")
z_2d = TSNE(n_components=2, random_state=42).fit_transform(mu_combined)

plt.figure(figsize=(10, 7))
plt.scatter(z_2d[:, 0], z_2d[:, 1], c=test_ds.y.numpy(), cmap='coolwarm', alpha=0.7)
plt.colorbar(label="Bug Present (1) vs Clean (0)")
plt.title(f"{cfg.PROJECT_NAME.upper()} Latent Space Topology (F1: {test_f1:.4f})")
plt.savefig(f"results_charts/{cfg.PROJECT_NAME}_best_tsne_final.png") 
plt.close()

print(f"\n🏁 Finished.")
print(f"🔒 Final Locked Decision Threshold (from Val Set): {final_optimized_threshold:.4f}")
print(f"🏆 Final Objective Test F1-Score: {test_f1:.4f}")

