import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Professional plotting config
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.2)

# 1. Load Data
df = pd.read_csv('Parameters.csv')
df['Class'] = df['Genotype'].apply(lambda x: 0 if x == 'Control' else 1)
df['Class_Label'] = df['Class'].map({0: 'Control', 1: 'Tumor'})

# 2. Scatter Plots (Spearman Rank Visualizations vs TumorPercent)
fig, axes = plt.subplots(1, 4, figsize=(22, 5))
features_to_plot = ['CTgray', 'VH', 'nVDP', 'VDP']
titles = ['CT Density vs Histology', 'Vent. Heterogeneity vs Histology', 'Norm. Defect % vs Histology', 'Raw Defect % vs Histology']

for ax, feat, title in zip(axes, features_to_plot, titles):
    sns.regplot(x=df['TumorPercent'], y=df[feat], ax=ax, scatter_kws={'s':150, 'edgecolor':'k', 'alpha':0.8, 'color':'#2ca02c'}, line_kws={'color':'#d62728', 'linestyle':'--'}, ci=None)
    ax.set_title(title, fontweight='bold', fontsize=14)
    ax.set_xlabel('True Tumor Burden (%)', fontsize=12)
    ax.set_ylabel(feat, fontsize=12)

plt.tight_layout()
plt.savefig('Postdoc_Fig_Spearman_Scatters.png', dpi=300)

# 3. Grouped Bar Chart (ML Performance Comparison)
# Data extracted from our LOOCV Table 4
ml_data = {
    'Model': ['Naive Bayes', 'Random Forest', 'Logistic Regression', 'Linear SVC', 'KNN (k=2)'],
    'Accuracy': [75.0, 62.5, 62.5, 62.5, 50.0],  # From earlier scripts (Note my latest script gave KNN 62.5% or 50% depending on k, let's use the table values)
    'Sensitivity': [100.0, 83.3, 50.0, 50.0, 50.0],
    'Specificity': [0.0, 0.0, 100.0, 100.0, 100.0],
    'F1-Score': [85.7, 76.9, 66.7, 66.7, 66.7]
}

# The latest execution output had KNN at 62.5% acc, 50% Sens, 100% Spec, 66.7% F1
ml_data['Accuracy'][4] = 62.5

ml_df = pd.DataFrame(ml_data)

# Melt for seaborn grouped barplot
ml_melt = ml_df.melt(id_vars='Model', var_name='Metric', value_name='Percentage (%)')

plt.figure(figsize=(12, 6))
sns.barplot(data=ml_melt, x='Model', y='Percentage (%)', hue='Metric', palette=['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd'], edgecolor='black')
plt.title('Machine Learning Performance (LOOCV, n=8)', fontweight='bold', fontsize=16)
plt.ylabel('Percentage (%)', fontsize=14)
plt.xlabel('Algorithmic Family', fontsize=14)
plt.ylim(0, 110)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True)
plt.tight_layout()
plt.savefig('Postdoc_Fig_ML_Performance.png', dpi=300)

print("Generated Postdoc_Fig_Spearman_Scatters.png and Postdoc_Fig_ML_Performance.png")
