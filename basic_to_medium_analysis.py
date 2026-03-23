import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix
import scipy.stats as stats
import pingouin as pg
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Set professional plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.2)

print("=========================================================")
print("  X-RAY VELOCIMETRY (XV) & CT: EXTENSIVE ML ANALYSIS      ")
print("=========================================================\n")

# 1. Load Data
df = pd.read_csv('Parameters.csv')
df['Class'] = df['Genotype'].apply(lambda x: 0 if x == 'Control' else 1)
df['Class_Label'] = df['Class'].map({0: 'Control', 1: 'Tumor'})

features = ['VDP', 'nVDP', 'VH', 'MSV', 'TV', 'CTgray']

print("--- 1. DESCRIPTIVE STATISTICS & 2. NORMALITY TESTING ---")
desc_stats = []
for feat in features:
    for cls_name, cls_num in [('Control', 0), ('Tumor', 1)]:
        vals = df[df['Class'] == cls_num][feat].dropna().values
        if len(vals) < 2: continue
        
        # Shapiro-Wilk Normality Test
        stat, p_shapiro = stats.shapiro(vals)
        
        desc_stats.append({
            'Feature': feat,
            'Group': cls_name,
            'Mean': np.mean(vals),
            'Median': np.median(vals),
            'SD': np.std(vals, ddof=1),
            'IQR': stats.iqr(vals),
            'Shapiro_p': p_shapiro,
            'Is_Normal': p_shapiro > 0.05
        })

desc_df = pd.DataFrame(desc_stats)
print(desc_df.to_string(index=False))

print("\n--- 3. NON-PARAMETRIC TESTS & 4. EFFECT SIZES (Hedge's g) ---")
stat_results = []
for feat in features:
    c_vals = df[df['Class'] == 0][feat].values
    t_vals = df[df['Class'] == 1][feat].values
    
    # Mann-Whitney U test
    u_stat, p_mwu = stats.mannwhitneyu(c_vals, t_vals, alternative='two-sided')
    
    # Hedge's g
    hedges_g = pg.compute_effsize(t_vals, c_vals, eftype='hedges')
    
    stat_results.append({
        'Feature': feat,
        'Control_Median': np.median(c_vals),
        'Tumor_Median': np.median(t_vals),
        'MWU_p': p_mwu,
        'Hedges_g': hedges_g
    })

mwu_df = pd.DataFrame(stat_results).sort_values('MWU_p')
print(mwu_df.to_string(index=False))

print("\n--- 6. CORRELATION ANALYSIS (Spearman) ---")
# Correlate XV biomarkers with TumorPercent
corrs = []
for feat in features:
    rho, p_val = stats.spearmanr(df[feat], df['TumorPercent'])
    corrs.append({'Feature': feat, 'Spearman_Rho': rho, 'p_value': p_val})
corr_df = pd.DataFrame(corrs)
print(corr_df.to_string(index=False))


print("\n--- 8 & 9. MULTIVARIATE PCA & PREDICTIVE ML (LOOCV) ---")
X = df[['VDP', 'VH', 'CTgray', 'nVDP', 'MSV']].values
y = df['Class'].values

roo = LeaveOneOut()
models = {
    "Random Forest": RandomForestClassifier(n_estimators=50, max_depth=2, random_state=42, class_weight='balanced'),
    "Logistic Regression (L2)": LogisticRegression(C=0.1, class_weight='balanced', random_state=42),
    "SVM (Linear)": SVC(kernel='linear', C=1.0, class_weight='balanced', random_state=42),
    "k-Nearest Neighbors (k=2)": KNeighborsClassifier(n_neighbors=2),
    "Naive Bayes": GaussianNB()
}

preds = {name: [] for name in models.keys()}

for train_idx, test_idx in roo.split(X):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    for name, model in models.items():
        if name in ["Random Forest", "Naive Bayes"]:
            model.fit(X_train, y_train)
            preds[name].append(model.predict(X_test)[0])
        else:
            model.fit(X_train_scaled, y_train)
            preds[name].append(model.predict(X_test_scaled)[0])

def print_metrics(y_true, y_pred, model_name):
    from sklearn.metrics import f1_score
    print(f"\n[{model_name}]")
    print(f"Accuracy: {accuracy_score(y_true, y_pred) * 100:.1f}%")
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    sens = tp / (tp + fn) if (tp+fn) > 0 else 0
    spec = tn / (tn + fp) if (tn+fp) > 0 else 0
    print(f"Sensitivity: {sens*100:.1f}% | Specificity: {spec*100:.1f}% | F1: {f1_score(y_true, y_pred)*100:.1f}%")

for name in models.keys():
    print_metrics(y, preds[name], name)

# VISUALIZATIONS
# Plot 1: Standard Boxplots
fig, axes = plt.subplots(1, 4, figsize=(20, 5))
for idx, feat in enumerate(['CTgray', 'VH', 'nVDP', 'MSV']):
    sns.boxplot(x='Class_Label', y=feat, data=df, ax=axes[idx], width=0.4, palette='Set2')
    sns.stripplot(x='Class_Label', y=feat, data=df, color='black', alpha=0.6, ax=axes[idx], jitter=True)
    axes[idx].set_title(f'{feat} Distribution')
    axes[idx].set_xlabel('')
plt.tight_layout()
plt.savefig('Postdoc_Fig_Distributions.png', dpi=300)

# Plot 2: Correlation Heatmap (Spearman)
plt.figure(figsize=(8, 6))
spearman_corr = df[features + ['TumorPercent']].corr(method='spearman')
sns.heatmap(spearman_corr, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
plt.title('Spearman Rank Correlation of XV Biomarkers')
plt.tight_layout()
plt.savefig('Postdoc_Fig_Correlation.png', dpi=300)

# Plot 3: PCA (Multivariate Dimension Reduction)
scaler = StandardScaler()
X_scaled_all = scaler.fit_transform(X)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled_all)
plt.figure(figsize=(8, 6)) # Make figure slightly wider to fit the legend outside
scatter = sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=df['Class_Label'], palette=['#1f77b4', '#d62728'], s=150, edgecolor='k')
plt.title(f'PCA of XV Biomarkers\n(PC1 & PC2 explain {np.sum(pca.explained_variance_ratio_)*100:.1f}% variance)')
plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
plt.legend(title='Group', bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True)
plt.tight_layout()
plt.savefig('Postdoc_Fig_PCA.png', dpi=300)

print("\nVisualizations saved: Postdoc_Fig_Distributions.png, Postdoc_Fig_Correlation.png, Postdoc_Fig_PCA.png")
