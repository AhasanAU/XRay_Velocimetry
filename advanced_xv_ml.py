import pandas as pd
import numpy as np
import glob
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, f1_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import warnings

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.3)

print("=========================================================")
print("  IMPROVED ADVANCED XV ML: RICH FEATURES + SMOTE + TUNING ")
print("=========================================================\n")

control_ids = ['2061', '2064']

print("Loading and Binning Raw XV Point Clouds...")
xv_files = glob.glob('raw_XV_data/raw_XV_data/*.specificVentilation.csv')

GRID_SIZE = 6   # Coarser grid => larger regions => more robust regional stats

augmented_data = []

for f in xv_files:
    sid = os.path.basename(f).split('.')[0]
    label = 0 if sid in control_ids else 1

    try:
        df = pd.read_csv(f, header=0, names=['SV', 'x', 'y', 'z'])
    except Exception as e:
        print(f"Skipping {sid}: {e}")
        continue

    print(f"Subject {sid} ({'Control' if label==0 else 'Tumor'}): {len(df)} voxels")

    # Normalize to [0,1]
    for col in ['x', 'y', 'z']:
        cmin, cmax = df[col].min(), df[col].max()
        df[col] = (df[col] - cmin) / (cmax - cmin + 1e-8)

    # Assign 3D grid bins
    df['x_bin'] = (df['x'] * GRID_SIZE).clip(0, GRID_SIZE-1).astype(int)
    df['y_bin'] = (df['y'] * GRID_SIZE).clip(0, GRID_SIZE-1).astype(int)
    df['z_bin'] = (df['z'] * GRID_SIZE).clip(0, GRID_SIZE-1).astype(int)

    for (xb, yb, zb), group in df.groupby(['x_bin', 'y_bin', 'z_bin']):
        if len(group) < 30:  # minimum voxels for reliable stats
            continue

        sv = group['SV'].values

        # --- Rich Feature Engineering ---
        augmented_data.append({
            'Subject': sid,
            'Label': label,
            # Central tendency
            'SV_Mean': np.mean(sv),
            'SV_Median': np.median(sv),
            # Spread / heterogeneity
            'SV_Std': np.std(sv),
            'SV_IQR': np.percentile(sv, 75) - np.percentile(sv, 25),
            # Shape
            'SV_Skewness': stats.skew(sv),
            'SV_Kurtosis': stats.kurtosis(sv),
            # Percentile-based functional thresholds (XV disease markers)
            'SV_p10': np.percentile(sv, 10),
            'SV_p90': np.percentile(sv, 90),
            # Defect / Hyper fractions
            'Defect_Frac': np.mean(sv < 0.05),
            'Hyper_Frac': np.mean(sv > 0.80),
            # Spatial position
            'X_Bin': xb,
            'Y_Bin': yb,
            'Z_Bin': zb,
        })

master_df = pd.DataFrame(augmented_data)
print(f"\n[Extraction] n=8 mice -> n={len(master_df)} regional samples (GRID_SIZE={GRID_SIZE})\n")

features = ['SV_Mean', 'SV_Median', 'SV_Std', 'SV_IQR', 'SV_Skewness', 'SV_Kurtosis',
            'SV_p10', 'SV_p90', 'Defect_Frac', 'Hyper_Frac', 'X_Bin', 'Y_Bin', 'Z_Bin']

X = master_df[features]
y = master_df['Label']
groups = master_df['Subject']

logo = LeaveOneGroupOut()

# Define tuned models
rf = RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=5,
                            class_weight='balanced', random_state=42)
xgb = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8,
                    scale_pos_weight=3.0,
                    eval_metric='logloss', random_state=42)
gbt = GradientBoostingClassifier(n_estimators=150, max_depth=4, learning_rate=0.05,
                                  subsample=0.8, random_state=42)
svm = SVC(kernel='rbf', C=2.0, class_weight='balanced', probability=True, random_state=42)

# Hard-voting ensemble
ensemble = VotingClassifier(
    estimators=[('rf', rf), ('xgb', xgb), ('gbt', gbt), ('svm', svm)],
    voting='soft'
)

model_names = ['Random Forest', 'XGBoost', 'GradientBoosting', 'SVM (RBF)',  'Ensemble (Soft Vote)']
all_models = [rf, xgb, gbt, svm, ensemble]
all_preds = {n: np.zeros(len(y)) for n in model_names}
all_probs = {n: np.zeros(len(y)) for n in model_names}
rf_fi = np.zeros(len(features))
xgb_fi = np.zeros(len(features))
fold = 0

print("--- LOGOCV TRAINING (All Models + SMOTE Balancing) ---")
for train_idx, test_idx in logo.split(X, y, groups):
    X_tr, X_te = X.iloc[train_idx].values, X.iloc[test_idx].values
    y_tr, y_te = y.iloc[train_idx].values, y.iloc[test_idx].values

    # Scale
    sc = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr)
    X_te_s = sc.transform(X_te)

    # SMOTE on training regions only (oversample minority = Control regions)
    try:
        sm = SMOTE(k_neighbors=min(5, np.sum(y_tr==0)-1), random_state=42)
        X_tr_s, y_tr = sm.fit_resample(X_tr_s, y_tr)
    except Exception:
        pass  # if too few samples to SMOTE, proceed without

    for name, mdl in zip(model_names, all_models):
        mdl.fit(X_tr_s, y_tr)
        all_preds[name][test_idx] = mdl.predict(X_te_s)
        all_probs[name][test_idx] = mdl.predict_proba(X_te_s)[:, 1]

    rf_fi += rf.feature_importances_
    xgb_fi += xgb.feature_importances_
    fold += 1

rf_fi /= fold
xgb_fi /= fold

# --- Results ---
print("\n{'='*55}")
print(f"{'Model':<28} {'Acc':>6} {'Sens':>6} {'Spec':>6} {'AUC':>6} {'F1':>6}")
print(f"{'-'*60}")

result_rows = []
cms = {}
for name in model_names:
    yp = all_preds[name]
    prob = all_probs[name]
    y_arr = y.values

    cm = confusion_matrix(y_arr, yp)
    tn, fp, fn, tp = cm.ravel()
    acc  = accuracy_score(y_arr, yp)
    sens = tp / (tp+fn) if (tp+fn) > 0 else 0
    spec = tn / (tn+fp) if (tn+fp) > 0 else 0
    auc  = roc_auc_score(y_arr, prob)
    f1   = f1_score(y_arr, yp)

    cms[name] = cm
    result_rows.append({'Model': name, 'Accuracy': acc, 'Sensitivity': sens,
                        'Specificity': spec, 'AUC': auc, 'F1': f1})
    print(f"{name:<28} {acc*100:>5.1f}% {sens*100:>5.1f}% {spec*100:>5.1f}%  {auc:.3f}  {f1:.3f}")

result_df = pd.DataFrame(result_rows)

# === VISUALIZATIONS ===
# 1. Confusion Matrices (2x3 grid)
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()
cmaps  = ['Blues', 'Reds', 'Greens', 'Purples', 'Oranges']
labels = ['Control', 'Tumor']
for i, (name, cmap) in enumerate(zip(model_names, cmaps)):
    cm = cms[name]
    tn, fp, fn, tp = cm.ravel()
    acc  = result_df[result_df['Model']==name]['Accuracy'].values[0]
    sens = result_df[result_df['Model']==name]['Sensitivity'].values[0]
    spec = result_df[result_df['Model']==name]['Specificity'].values[0]
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, ax=axes[i], annot_kws={"size": 16},
                xticklabels=labels, yticklabels=labels, cbar=False)
    axes[i].set_title(f'{name}\nAcc={acc*100:.1f}%  Sens={sens*100:.1f}%  Spec={spec*100:.1f}%',
                      fontweight='bold', fontsize=11)
    axes[i].set_ylabel('True Label')
    axes[i].set_xlabel('Predicted Label')
axes[5].axis('off')
plt.suptitle('LOGOCV Confusion Matrices – 3D Spatial XV Regions (n=8 mice, SMOTE balanced)',
             fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('Postdoc_Fig_Advanced_CM.png', dpi=300, bbox_inches='tight')
print("\nSaved Postdoc_Fig_Advanced_CM.png")

# 2. Grouped Performance Bar Chart
perf_melt = result_df.melt(id_vars='Model', value_vars=['Accuracy', 'Sensitivity', 'Specificity', 'AUC', 'F1'],
                            var_name='Metric', value_name='Score')
plt.figure(figsize=(14, 6))
sns.barplot(data=perf_melt, x='Model', y='Score', hue='Metric',
            palette=['#1f77b4','#ff7f0e','#2ca02c','#9467bd','#d62728'], edgecolor='black')
plt.title('Model Performance Comparison (LOGOCV)', fontsize=16, fontweight='bold')
plt.ylabel('Score')
plt.ylim(0, 1.05)
plt.xticks(rotation=20, ha='right')
plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
plt.tight_layout()
plt.savefig('Postdoc_Fig_Advanced_Perf.png', dpi=300, bbox_inches='tight')
print("Saved Postdoc_Fig_Advanced_Perf.png")

# 3. Feature Importance (dual RF & XGBoost)
fi_df = pd.DataFrame({'Feature': features, 'Random Forest': rf_fi, 'XGBoost': xgb_fi})\
          .melt(id_vars='Feature', var_name='Model', value_name='Importance')
plt.figure(figsize=(12, 6))
sns.barplot(data=fi_df, y='Feature', x='Importance', hue='Model',
            palette=['#1f77b4','#d62728'], edgecolor='black')
plt.title('Mean Feature Importance (Averaged over 8 LOGOCV Folds)', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig('Postdoc_Fig_Advanced_FI.png', dpi=300, bbox_inches='tight')
print("Saved Postdoc_Fig_Advanced_FI.png")
