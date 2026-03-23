================================================================================
  X-RAY VELOCIMETRY (XV) LUNG DISEASE ANALYSIS — PROJECT FILES
================================================================================

ANALYTICAL PYTHON SCRIPTS
──────────────────────────

1. basic_to_medium_analysis.py
   Description : Entry-level statistical and machine learning analysis of
                 global XV and CT parameters from Parameters.csv.
   Input       : Parameters.csv
   Analysis    : Descriptive statistics (Mean, Median, SD, IQR),
                 Shapiro-Wilk normality tests, Mann-Whitney U group
                 comparisons, Hedge's g effect sizes, Spearman rank
                 correlations, Principal Component Analysis (PCA),
                 and supervised binary classification (Control vs Tumor)
                 using Leave-One-Out Cross-Validation (LOOCV) with five
                 models: Naive Bayes, Random Forest, Logistic Regression,
                 Linear SVM, and k-Nearest Neighbours (k=2).
   Output      : Statistical tables, boxplots, correlation heatmap,
                 PCA scatter plot, ML performance bar chart.
   Report      : Lung_Disease_XV_Analysis_Report.docx

2. advanced_xv_ml.py
   Description : Advanced machine learning analysis operating directly on
                 the raw 3D X-Ray Velocimetry point clouds
                 (*.specificVentilation.csv).
   Input       : raw_XV_data/raw_XV_data/*.specificVentilation.csv
   Analysis    : Spatial 3D grid binning (6x6x6) of the lung volume to
                 extract 13 region-level biological features per bin
                 (SV mean, median, SD, IQR, skewness, kurtosis, 10th/90th
                 percentile, defect fraction, hyper-inflation fraction, and
                 spatial XYZ bin coordinates). SMOTE oversampling applied
                 inside each training fold to address class imbalance.
                 Five models trained under strict Leave-One-Group-Out CV
                 (LOGOCV): Random Forest, XGBoost, Gradient Boosting,
                 SVM (RBF), and a Soft-Vote Ensemble.
   Output      : Confusion matrix heatmaps, feature importance chart,
                 model performance grouped bar chart.
   Report      : Lung_Disease_Advanced_XV_Analysis_Report.docx

3. advanced_xv_dl.py
   Description : Deep learning analysis of the raw XV 3D SV point clouds
                 using PyTorch on CPU. Designed and constrained to run
                 on a standard laptop (Intel i7-8650U, 16 GB RAM, no GPU).
   Input       : raw_XV_data/raw_XV_data/*.specificVentilation.csv
   Analysis    : Constructs a (1 x 6 x 6 x 6) volumetric Specific
                 Ventilation map per mouse. Applies physics-informed
                 data augmentation (Gaussian noise, axis flips, intensity
                 scaling; asymmetric 10x Control / 3x Tumor ratio to
                 address 2:6 class imbalance). Trains two architectures
                 under LOGOCV: (i) Lightweight 3D-CNN using nn.Conv3d
                 layers (~31,000 parameters), and (ii) MLP on 12 global
                 XV features. Optimal decision threshold selected per fold
                 via Youden's J statistic on training data.
   Output      : Training loss/accuracy curves, confusion matrices,
                 ROC curves, performance comparison bar chart.
   Report      : XV_Deep_Learning_Report.docx


GENERATED REPORT FILES (DOCX)
──────────────────────────────

  Lung_Disease_XV_Analysis_Report.docx
      Comprehensive results from basic_to_medium_analysis.py.
      Includes all statistical tables, figures, and ML performance metrics.

  Lung_Disease_Advanced_XV_Analysis_Report.docx
      Comprehensive results from advanced_xv_ml.py.
      Includes regional feature descriptions, confusion matrices,
      feature importance, and discussion of SMOTE augmentation.

  XV_Deep_Learning_Report.docx
      Comprehensive results from advanced_xv_dl.py.
      Includes architecture descriptions, augmentation strategy,
      training curves, confusion matrices, and ROC curve analysis.


ENVIRONMENT
───────────
  Python Virtual Environment : .XRayV  (in project root)
  Python Version             : 3.13 (Anaconda distribution)
  Activate (Windows)         : .XRayV\Scripts\activate
  Run any script             : .XRayV\Scripts\python.exe <script>.py

================================================================================
