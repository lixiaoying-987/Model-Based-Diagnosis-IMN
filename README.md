# Source Code for Manuscript: Non-invasive Model-Based Diagnosis of PLA2R-Negative Idiopathic Membranous Nephropathy

This repository contains the source code used for data analysis in the manuscript submitted to **Kidney International Reports**.

The analysis pipeline includes data cleaning, baseline characteristic analysis, feature selection (Correlation, VIF, LASSO), and machine learning modeling.

## 📁 Repository Structure 

The analysis consists of both Python and R scripts. Below is the description of each file:

### 1. Data Preprocessing & Baseline
* `data_cleaning.py`: Script for missing value imputation, outlier detection, and data standardization.
* `baseline_analysis_gtsummary.R`: R script using `gtsummary` package to generate the baseline characteristics table (Table 1) and perform statistical tests.
* `sample_resampling.py`: Script for handling data imbalance (e.g., using SMOTE or other resampling techniques).

### 2. Feature Selection
* `Collinearity_analysis_VIF.py`: Calculates Variance Inflation Factor (VIF) to detect multicollinearity among features.
* `Correlation analysis (Pearson).py`: Performs Pearson correlation analysis and generates heatmaps.
* `LASSO_feature_selection_Rwrapp...`: Implements LASSO regression for feature selection (integrating R's `glmnet` via Python or wrapper).

### 3. Machine Learning Modeling
* `rf_binary_classification.py`: Random Forest classifier for binary classification tasks.
* `feature_importance_analysis.py`: Extracts and visualizes feature importance from the trained models.
* `two_groups_multimodel_classification...`: Comprehensive script comparing multiple machine learning models (e.g., SVM, XGBoost, LR) and plotting ROC/DCA curves.

## 🛠️ Environment Requirements 

To reproduce the results, the following environments and libraries are recommended:

**Python (3.11.4)**
* numpy, pandas
* scikit-learn (sklearn)
* matplotlib, seaborn
* xgboost, lightgbm, shap
* imblearn

**R (4.2.3)**
* gtsummary
* glmnet
* dplyr

## 🚀 Usage 

1.  **Data Preparation**: Place your dataset (CSV/Excel) in the same directory as the scripts. *Note: The dataset provided in this repository (if any) is synthetic/de-identified for demonstration purposes to protect patient privacy.*
2.  **Run Order**:
    * Step 1: Run `data_cleaning.py` to preprocess raw data.
    * Step 2: Run `baseline_analysis_gtsummary.R` for Table 1.
    * Step 3: Run feature selection scripts (`LASSO`, `VIF`, `Correlation`).
    * Step 4: Run modeling scripts (`rf_binary_classification.py` or `two_groups_multimodel...`) to generate predictions and performance metrics.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## ✉️ Contact

For any questions regarding the code or the manuscript, please contact the corresponding author.

