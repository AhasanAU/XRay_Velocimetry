"""
XV Deep Learning Pipeline — Improved (CPU-Optimised)
Laptop: Intel i7-8650U | 16 GB RAM | CPU-only | PyTorch 2.10

Key improvements over v1:
  1. BCEWithLogitsLoss + pos_weight=3.0 — penalises Control misclassification 3x more.
  2. Asymmetric augmentation: 10x copies for Control volumes, 3x for Tumor (corrects the 2:6 imbalance).
  3. Training-fold threshold search: optimal decision threshold is found on training data
     (using Youden's J) and applied at test time — no more fixed 0.5 cutoff.
  4. WeightedRandomSampler for MLP — ensures 50:50 class draw per batch.
"""
import pandas as pd
import numpy as np
import glob, os, warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             roc_auc_score, f1_score, roc_curve)
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context('paper', font_scale=1.2)
torch.manual_seed(42); np.random.seed(42)

DEVICE  = torch.device('cpu')
GRID    = 6
EPOCHS  = 60
LR      = 8e-4
BATCH   = 4
POS_W   = 3.0   # Control class weight in loss (upweights Control detection)

print("="*60)
print("  XV DEEP LEARNING v2 — IMPROVED SPECIFICITY")
print(f"  Laptop: Intel i7-8650U | 16 GB RAM | CPU-only")
print("="*60)

# ── 1. BUILD 3D VOLUMETRIC MAPS ────────────────────────────────
control_ids = {'2061', '2064'}
xv_files = sorted(glob.glob('raw_XV_data/raw_XV_data/*.specificVentilation.csv'))

volumes = []
region_rows = []

print("\nBuilding 3D SV volumes ...")
for f in xv_files:
    sid   = os.path.basename(f).split('.')[0]
    label = 0 if sid in control_ids else 1
    try:
        df = pd.read_csv(f, header=0, names=['SV','x','y','z'])
    except Exception as e:
        print(f"  Skip {sid}: {e}"); continue

    for c in ['x','y','z']:
        mn, mx = df[c].min(), df[c].max()
        df[c] = (df[c]-mn)/(mx-mn+1e-8)
    df['xb'] = (df['x']*GRID).clip(0,GRID-1).astype(int)
    df['yb'] = (df['y']*GRID).clip(0,GRID-1).astype(int)
    df['zb'] = (df['z']*GRID).clip(0,GRID-1).astype(int)

    vol = np.zeros((GRID,GRID,GRID), dtype=np.float32)
    for (xb,yb,zb), g in df.groupby(['xb','yb','zb']):
        vol[xb,yb,zb] = g['SV'].mean()

    volumes.append({'subject':sid, 'label':label, 'volume':vol})

    sv_all = df['SV'].values
    region_rows.append({
        'Subject':sid, 'Label':label,
        'SV_Mean':np.mean(sv_all), 'SV_Median':np.median(sv_all),
        'SV_Std':np.std(sv_all),
        'SV_IQR':np.percentile(sv_all,75)-np.percentile(sv_all,25),
        'SV_Skew':stats.skew(sv_all), 'SV_Kurt':stats.kurtosis(sv_all),
        'SV_p10':np.percentile(sv_all,10), 'SV_p90':np.percentile(sv_all,90),
        'Defect':np.mean(sv_all<0.05), 'Hyper':np.mean(sv_all>0.80),
        'TotalVox':len(sv_all)/10000.0,
    })
    print(f"  {sid} ({'Control' if label==0 else 'Tumor'}): non-zero cells={np.count_nonzero(vol)}")

mlp_df = pd.DataFrame(region_rows)
print(f"\nTotal mice loaded: {len(volumes)}\n")

# ── 2. ASYMMETRIC AUGMENTATION ─────────────────────────────────
def augment_volume(vol):
    v = vol.copy()
    v += np.random.normal(0, 0.02, v.shape).astype(np.float32)
    v = np.clip(v, 0, 1)
    if np.random.rand() > 0.5: v = np.flip(v, axis=0).copy()
    if np.random.rand() > 0.5: v = np.flip(v, axis=2).copy()
    v = np.clip(v * np.random.uniform(0.88, 1.12), 0, 1)
    return v

def build_aug_set(train_volumes, train_labels):
    """
    Asymmetric augmentation: 10x copies for Control (label=0),
    3x copies for Tumor (label=1) — corrects the 2:6 structural imbalance.
    """
    aug_v, aug_l = [], []
    for vol, lbl in zip(train_volumes, train_labels):
        n_copies = 10 if lbl == 0 else 3
        for _ in range(n_copies):
            aug_v.append(augment_volume(vol))
            aug_l.append(lbl)
    return aug_v, aug_l

# ── 3. PYTORCH DATASETS ────────────────────────────────────────
class VolumeDataset(Dataset):
    def __init__(self, vols, labels):
        self.vols   = vols
        self.labels = labels
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx):
        x = torch.tensor(self.vols[idx][np.newaxis], dtype=torch.float32)
        y = torch.tensor([self.labels[idx]], dtype=torch.float32)
        return x, y

class TabularDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.X[idx], self.y[idx].unsqueeze(0)

# ── 4. MODELS ──────────────────────────────────────────────────
class Lightweight3DCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(1, 8, 3, padding=1), nn.BatchNorm3d(8), nn.ReLU(),
            nn.Conv3d(8, 16, 3, padding=1), nn.BatchNorm3d(16), nn.ReLU(),
            nn.MaxPool3d(2), nn.Dropout3d(0.3)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(432, 64), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(64, 1)   # raw logits — no Sigmoid (BCEWithLogitsLoss)
        )
    def forward(self, x): return self.classifier(self.encoder(x))

class MLP(nn.Module):
    def __init__(self, inp=12):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(inp, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(32, 1)    # raw logits
        )
    def forward(self, x): return self.net(x)

# ── 5. THRESHOLD SEARCH (Youden's J) ──────────────────────────
def best_threshold(y_true, probs):
    """Find threshold maximising Sensitivity + Specificity - 1 (Youden's J)."""
    fpr, tpr, thresholds = roc_curve(y_true, probs)
    j_scores = tpr - fpr
    return thresholds[np.argmax(j_scores)]

# ── 6. TRAINING HELPER ────────────────────────────────────────
def train_model(model, train_loader, val_loader, pos_weight_val):
    # pos_weight: Controls the weight for positive (Tumor=1) class.
    # Setting pos_weight < 1 makes the model more sensitive to Control.
    # Note: in BCEWithLogitsLoss, pos_weight applies to the '1' class (Tumor).
    # To favour Control detection, we use a low pos_weight (< 1.0).
    pw   = torch.tensor([1.0 / pos_weight_val])  # < 1.0 penalises Tumor FNs less
    crit = nn.BCEWithLogitsLoss(pos_weight=pw)
    opt  = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-3)
    sch  = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    tr_losses, tr_accs = [], []
    for ep in range(EPOCHS):
        model.train()
        ep_loss = correct = total = 0
        for xb, yb in train_loader:
            opt.zero_grad()
            logits = model(xb)
            loss   = crit(logits, yb)
            loss.backward(); opt.step()
            preds      = (torch.sigmoid(logits) > 0.5).float()
            correct   += (preds == yb).sum().item()
            total     += len(yb)
            ep_loss   += loss.item() * len(yb)
        sch.step()
        tr_losses.append(ep_loss / total)
        tr_accs.append(correct / total)

    # Collect probabilities on train set for threshold search
    model.eval()
    tr_probs, tr_true = [], []
    with torch.no_grad():
        for xb, yb in train_loader:
            tr_probs.extend(torch.sigmoid(model(xb)).squeeze().tolist())
            tr_true.extend(yb.squeeze().tolist())

    # Threshold search on training fold
    try:
        thr = best_threshold(np.array(tr_true), np.array(tr_probs))
        thr = float(np.clip(thr, 0.2, 0.9))
    except Exception:
        thr = 0.5

    # Predict on test
    te_probs = []
    with torch.no_grad():
        for xb, _ in val_loader:
            p = torch.sigmoid(model(xb)).squeeze()
            val = p.item() if p.dim()==0 else p.tolist()
            if isinstance(val, list): te_probs.extend(val)
            else: te_probs.append(val)

    return np.array(te_probs, dtype=np.float32), np.array(tr_losses), np.array(tr_accs), thr

# ── 7. LOGOCV — 3D-CNN ────────────────────────────────────────
print("--- 3D-CNN LOGOCV ---")
subjects = [v['subject'] for v in volumes]
labels   = [v['label']   for v in volumes]
vols_arr = [v['volume']  for v in volumes]

cnn_probs  = np.zeros(len(subjects))
cnn_preds  = np.zeros(len(subjects))
cnn_losses, cnn_accs = [], []

for fold_i, test_i in enumerate(range(len(subjects))):
    train_idx = [i for i in range(len(subjects)) if i != test_i]
    tr_vols_raw = [vols_arr[i] for i in train_idx]
    tr_labs_raw = [labels[i]   for i in train_idx]

    aug_v, aug_l = build_aug_set(tr_vols_raw, tr_labs_raw)

    tr_ds = VolumeDataset(aug_v, aug_l)
    te_ds = VolumeDataset([vols_arr[test_i]], [labels[test_i]])
    tr_ld = DataLoader(tr_ds, batch_size=BATCH, shuffle=True)
    te_ld = DataLoader(te_ds, batch_size=1, shuffle=False)

    model = Lightweight3DCNN().to(DEVICE)
    probs, losses, accs, thr = train_model(model, tr_ld, te_ld, POS_W)

    cnn_probs[test_i] = probs[0]
    cnn_preds[test_i] = int(probs[0] > thr)
    cnn_losses.append(losses); cnn_accs.append(accs)
    print(f"  Fold {fold_i+1}/8 | {subjects[test_i]} "
          f"{'Control' if labels[test_i]==0 else 'Tumor ':6s} | "
          f"prob={probs[0]:.3f} thr={thr:.3f} pred={'Tumor' if probs[0]>thr else 'Control'}")

# ── 8. LOGOCV — MLP ───────────────────────────────────────────
print("\n--- MLP LOGOCV ---")
FEAT_COLS = ['SV_Mean','SV_Median','SV_Std','SV_IQR','SV_Skew','SV_Kurt',
             'SV_p10','SV_p90','Defect','Hyper','TotalVox','SV_Std']
X_tab = mlp_df[FEAT_COLS].values
y_tab = mlp_df['Label'].values
g_tab = mlp_df['Subject'].values

logo = LeaveOneGroupOut()
mlp_probs = np.zeros(len(y_tab))
mlp_preds = np.zeros(len(y_tab))
mlp_losses, mlp_accs = [], []

for fold_i, (tr_idx, te_idx) in enumerate(logo.split(X_tab, y_tab, g_tab)):
    sc = StandardScaler()
    X_tr = sc.fit_transform(X_tab[tr_idx]).astype(np.float32)
    X_te = sc.transform(X_tab[te_idx]).astype(np.float32)
    y_tr = y_tab[tr_idx].astype(np.float32)

    # Asymmetric augmentation for tabular: 10x Control, 3x Tumor
    aug_X, aug_y = [], []
    for xi, yi in zip(X_tr, y_tr):
        n = 10 if yi == 0 else 3
        for _ in range(n):
            noise = np.random.normal(0, 0.05, xi.shape).astype(np.float32)
            aug_X.append(xi + noise); aug_y.append(yi)
    X_aug = np.vstack(aug_X); y_aug = np.array(aug_y)

    # WeightedRandomSampler for balanced batch draws
    class_counts = np.bincount(y_aug.astype(int))
    sample_w = 1.0 / class_counts[y_aug.astype(int)]
    sampler  = WeightedRandomSampler(sample_w, len(sample_w), replacement=True)

    tr_ds = TabularDataset(X_aug, y_aug)
    te_ds = TabularDataset(X_te, y_tab[te_idx].astype(np.float32))
    tr_ld = DataLoader(tr_ds, batch_size=BATCH, sampler=sampler)
    te_ld = DataLoader(te_ds, batch_size=BATCH, shuffle=False)

    model = MLP(inp=len(FEAT_COLS)).to(DEVICE)
    probs, losses, accs, thr = train_model(model, tr_ld, te_ld, POS_W)

    mlp_probs[te_idx] = probs
    mlp_preds[te_idx] = (probs > thr).astype(int)
    mlp_losses.append(losses); mlp_accs.append(accs)
    print(f"  Fold {fold_i+1}/8 | {np.unique(g_tab[te_idx])[0]} | "
          f"prob={probs[0]:.3f} thr={thr:.3f}")

# ── 9. EVALUATION ─────────────────────────────────────────────
y_cnn = np.array(labels)
y_mlp = y_tab

def compute_metrics(y_true, y_pred, y_prob, name):
    cm = confusion_matrix(y_true, y_pred)
    tn,fp,fn,tp = cm.ravel() if cm.size==4 else (0,0,0,0)
    acc  = accuracy_score(y_true, y_pred)
    sens = tp/(tp+fn) if (tp+fn)>0 else 0
    spec = tn/(tn+fp) if (tn+fp)>0 else 0
    try: auc = roc_auc_score(y_true, y_prob)
    except: auc = 0.5
    f1 = f1_score(y_true, y_pred, zero_division=0)
    print(f"\n[{name}]  Acc={acc*100:.1f}%  Sens={sens*100:.1f}%  Spec={spec*100:.1f}%  AUC={auc:.3f}  F1={f1:.3f}")
    return dict(Model=name,Accuracy=acc,Sensitivity=sens,Specificity=spec,AUC=auc,F1=f1), cm

print("\n" + "="*60)
r_cnn, cm_cnn = compute_metrics(y_cnn, cnn_preds, cnn_probs, "3D-CNN")
r_mlp, cm_mlp = compute_metrics(y_mlp, mlp_preds, mlp_probs, "MLP")
perf_df = pd.DataFrame([r_cnn, r_mlp])

# ── 10. FIGURES ────────────────────────────────────────────────
# Fig 1 — Training Curves
fig, axes = plt.subplots(2,2, figsize=(14,9))
for ai,(losses,accs,mn,col) in enumerate([
    (cnn_losses,cnn_accs,'3D-CNN','#1f77b4'),
    (mlp_losses,mlp_accs,'MLP','#d62728')]):
    L=np.array(losses); A=np.array(accs); ep=np.arange(1,EPOCHS+1)
    axes[0,ai].plot(ep,L.mean(0),color=col,lw=2)
    axes[0,ai].fill_between(ep,L.mean(0)-L.std(0),L.mean(0)+L.std(0),alpha=0.2,color=col)
    axes[0,ai].set_title(f'{mn} — BCE Loss',fontweight='bold'); axes[0,ai].set_xlabel('Epoch')
    axes[1,ai].plot(ep,A.mean(0),color=col,lw=2)
    axes[1,ai].fill_between(ep,A.mean(0)-A.std(0),A.mean(0)+A.std(0),alpha=0.2,color=col)
    axes[1,ai].set_title(f'{mn} — Accuracy',fontweight='bold')
    axes[1,ai].set_xlabel('Epoch'); axes[1,ai].set_ylim(0,1.05)
plt.suptitle('Training Curves (Mean +/- Std, 8 LOGOCV Folds)',fontsize=13,fontweight='bold')
plt.tight_layout(); plt.savefig('DL_Fig_TrainingCurves.png',dpi=300,bbox_inches='tight')
print("Saved DL_Fig_TrainingCurves.png")

# Fig 2 — Confusion Matrices + ROC
fig, axes = plt.subplots(2,2,figsize=(14,11))
cls_labels = ['Control','Tumor']
for ai,(cm,probs_p,y_true,mn,cmap,col) in enumerate([
    (cm_cnn,cnn_probs,y_cnn,'3D-CNN','Blues','#1f77b4'),
    (cm_mlp,mlp_probs,y_mlp,'MLP','Reds','#d62728')]):
    r = perf_df[perf_df.Model==mn].iloc[0]
    sns.heatmap(cm,annot=True,fmt='d',cmap=cmap,ax=axes[0,ai],annot_kws={'size':20},
                xticklabels=cls_labels,yticklabels=cls_labels,cbar=False)
    axes[0,ai].set_title(f'{mn}\nAcc={r.Accuracy*100:.1f}%  Sens={r.Sensitivity*100:.1f}%  Spec={r.Specificity*100:.1f}%',fontweight='bold')
    axes[0,ai].set_ylabel('True'); axes[0,ai].set_xlabel('Predicted')
    try:
        fpr,tpr,_ = roc_curve(y_true,probs_p)
        axes[1,ai].plot(fpr,tpr,color=col,lw=2.5,label=f'AUC={r.AUC:.3f}')
        axes[1,ai].plot([0,1],[0,1],'k--',lw=1)
    except: pass
    axes[1,ai].set_title(f'{mn} — ROC Curve',fontweight='bold')
    axes[1,ai].set_xlabel('False Positive Rate'); axes[1,ai].set_ylabel('True Positive Rate')
    axes[1,ai].legend(loc='lower right'); axes[1,ai].set_xlim(0,1); axes[1,ai].set_ylim(0,1.02)
plt.suptitle('Confusion Matrices & ROC (LOGOCV, n=8 mice, Tuned Threshold)',
             fontsize=13,fontweight='bold',y=1.01)
plt.tight_layout(); plt.savefig('DL_Fig_CM_ROC.png',dpi=300,bbox_inches='tight')
print("Saved DL_Fig_CM_ROC.png")

# Fig 3 — Performance bar
melt = perf_df.melt(id_vars='Model',value_vars=['Accuracy','Sensitivity','Specificity','AUC','F1'],
                    var_name='Metric',value_name='Score')
plt.figure(figsize=(9,5))
sns.barplot(data=melt,x='Model',y='Score',hue='Metric',
            palette=['#1f77b4','#ff7f0e','#2ca02c','#9467bd','#d62728'],edgecolor='black')
plt.title('DL Model Performance (LOGOCV)',fontsize=15,fontweight='bold')
plt.ylim(0,1.05); plt.ylabel('Score')
plt.legend(bbox_to_anchor=(1.01,1),loc='upper left')
plt.tight_layout(); plt.savefig('DL_Fig_PerfBar.png',dpi=300,bbox_inches='tight')
print("Saved DL_Fig_PerfBar.png\n")
print(perf_df.to_string(index=False))
