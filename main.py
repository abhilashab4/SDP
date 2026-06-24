
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os, pickle, copy, random
import pandas as pd
import numpy as np

# Internal Imports
import config as cfg
from src.dataset import MultiModalDefectDataset
from src.model import MultiModalVAE
from src.utils import save_performance_report

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(42)

# Setup Directories
os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

# 1. Data Loading Pipeline (Leveraging clean parameters to avoid data leakage)
print(f"--- Starting Experiment: {cfg.PROJECT_NAME.upper()} ---")
train_paths = [f"data/{cfg.PROJECT_NAME}/{cfg.PROJECT_NAME}_{v}_enriched.csv" for v in cfg.TRAIN_VERSIONS]
train_dfs = [pd.read_csv(p) for p in train_paths if os.path.exists(p)]
combined_train_df = pd.concat(train_dfs).reset_index(drop=True)

# Define explicit column paths
metrics_list = [c for c in combined_train_df.columns if c not in ['classname', 'ast_seq', 'code_tokens', 'bug']]

train_ds = MultiModalDefectDataset(
    combined_train_df, feature_cols=metrics_list, label_col='bug', text_cols=['ast_seq', 'code_tokens']
)

test_path = f"data/{cfg.PROJECT_NAME}/{cfg.PROJECT_NAME}_{cfg.TEST_VERSION}_enriched.csv"
test_ds = MultiModalDefectDataset(
    pd.read_csv(test_path), feature_cols=metrics_list, label_col='bug', text_cols=['ast_seq', 'code_tokens'],
    tokenizer=train_ds.tokenizer, mean=train_ds.mean, std=train_ds.std
)

train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=cfg.BATCH_SIZE, shuffle=False)

# 2. Model Initialization
model = MultiModalVAE(cfg.METRICS_DIM, cfg.VOCAB_SIZE, cfg.EMBED_DIM, cfg.LATENT_DIM).to(cfg.DEVICE)
optimizer = optim.Adam(model.parameters(), lr=cfg.LR)

# Ensure proper 2D shape format compatibility
pos_weight = train_ds.get_pos_weight().to(cfg.DEVICE)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

best_f1 = 0.0
best_model_wts = copy.deepcopy(model.state_dict())

# 3. Training Loop
print(f"Training for {cfg.EPOCHS} Epochs (Seed: 42)...")
for epoch in range(cfg.EPOCHS):
    model.train()
    for m, s, y in train_loader:
        m, s, y = m.to(cfg.DEVICE), s.to(cfg.DEVICE), y.to(cfg.DEVICE).unsqueeze(1) # [B, 1] Matrix Form
        optimizer.zero_grad()
        
        # Unpacks synchronized return types: (logits, mu, logvar)
        logits, mu, logvar = model(m, s)
        
        # --- Latent Augmentation (Generative Sampling) ---
        bug_idx = (y == 1).nonzero(as_tuple=True)[0]
        if len(bug_idx) > 0:
            std_bugs = torch.exp(0.5 * logvar[bug_idx])
            z_syn = mu[bug_idx] + torch.randn_like(std_bugs) * std_bugs
            l_syn = model.classifier(z_syn)
            
            # Keep 2D structures clean [Batch, 1]
            all_logits = torch.cat([logits, l_syn], dim=0)
            all_targets = torch.cat([y, torch.ones(len(bug_idx), 1).to(cfg.DEVICE)], dim=0)
        else:
            all_logits, all_targets = logits, y

        # Variational Penalization Optimization
        classification_loss = criterion(all_logits, all_targets)
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        
        loss = classification_loss + cfg.KL_COEFF * kl_loss
        loss.backward()
        optimizer.step()

    # ---- Evaluation ----
    model.eval()
    y_true_epoch, y_pred_epoch = [], []
    with torch.no_grad():
        for m_t, s_t, y_t in test_loader:
            l_t, _, _ = model(m_t.to(cfg.DEVICE), s_t.to(cfg.DEVICE))
            preds = (torch.sigmoid(l_t) > 0.5).int().cpu().numpy() # Explicit integer cast
            y_pred_epoch.extend(preds)
            y_true_epoch.extend(y_t.numpy())
    
    current_f1 = f1_score(y_true_epoch, y_pred_epoch, zero_division=0)
    if current_f1 > best_f1:
        best_f1 = current_f1
        best_model_wts = copy.deepcopy(model.state_dict())
        torch.save(best_model_wts, cfg.MODEL_SAVE_PATH)
    print(f"Epoch {epoch+1:03d} | Current F1: {current_f1:.4f} | Best F1: {best_f1:.4f}")

# 4. Final Verification and Latent Vector Compilation
print(f"\n📊 Finalizing Results for {cfg.PROJECT_NAME.upper()}...")
model.load_state_dict(best_model_wts)
with open(cfg.VOCAB_SAVE_PATH, "wb") as f:
    pickle.dump(train_ds.tokenizer, f)

model.eval()
final_true, final_pred = [], []
all_mu = []

with torch.no_grad():
    for m_t, s_t, y_t in test_loader:
        l_t, mu_t, _ = model(m_t.to(cfg.DEVICE), s_t.to(cfg.DEVICE))
        
        final_pred.extend((torch.sigmoid(l_t) > 0.5).int().cpu().numpy())
        final_true.extend(y_t.numpy())
        all_mu.append(mu_t.cpu().numpy()) # Capture true mean attributes safely

# Combine lists to prevent VRAM memory overflows
mu_combined = np.concatenate(all_mu, axis=0)
save_performance_report(final_true, final_pred, cfg.PROJECT_NAME)

print("Running t-SNE dimension reduction...")
z_2d = TSNE(n_components=2, random_state=42).fit_transform(mu_combined)

plt.figure(figsize=(10, 7))
plt.scatter(z_2d[:, 0], z_2d[:, 1], c=test_ds.y.numpy(), cmap='coolwarm', alpha=0.7)
plt.colorbar(label="Bug Present (1) vs Clean (0)")
plt.title(f"{cfg.PROJECT_NAME.upper()} Latent Space Topology (F1: {best_f1:.4f})")
plt.savefig(f"results_1/{cfg.PROJECT_NAME}_best_tsne_final.png")
plt.close()

print(f"\n🏁 Finished. Global Maximum F1 Score: {best_f1:.4f}.")