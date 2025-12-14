# Media Campaign Cost Prediction

A machine learning regression project for predicting customer costs using ensemble methods and hyperparameter optimization.

## Overview

This project implements a regression analysis pipeline for predicting media campaign costs based on customer and store attributes. The dataset, sourced from the Kaggle Playground Series S3E11 competition, contains synthetic data generated from a deep learning model trained on real-world media campaign data.

## Dataset

**Source:** [Kaggle Playground Series S3E11](https://www.kaggle.com/competitions/playground-series-s3e11/data)

**Dataset Statistics:**
- Training samples: 360,336
- Test samples: 240,224
- Original training samples: 51,363
- Features: 14 numerical features
- Target variable: `cost` (log-transformed for modeling)

**Features:**
- `store_sales(in millions)` - Store sales volume
- `unit_sales(in millions)` - Unit sales volume
- `total_children` - Total number of children in household
- `num_children_at_home` - Number of children living at home
- `avg_cars_at home(approx).1` - Average number of cars at home
- `gross_weight` - Product gross weight
- `recyclable_package` - Binary indicator for recyclable packaging
- `low_fat` - Binary indicator for low-fat products
- `units_per_case` - Number of units per case
- `store_sqft` - Store square footage
- Store amenities: `coffee_bar`, `video_store`, `salad_bar`, `prepared_food`, `florist`

## Methodology

### Data Preprocessing
- **Target transformation:** Applied log transformation (`log1p`) to the cost variable to normalize distribution
- **Missing values:** No missing values detected in the dataset
- **Duplicate handling:** Identified 541 duplicate entries in original training data

### Feature Engineering
Created three new features to capture additional patterns:
1. **`child_ratio`** - Ratio of total children to children at home
2. **`facilities`** - Sum of all store amenities (coffee bar, video store, salad bar, prepared food, florist)
3. **`independent_child`** - Difference between total children and children at home

### Models

#### 1. Random Forest Regressor
- **Configuration:** 100 estimators, max depth of 12
- **Purpose:** Baseline model and feature importance analysis
- **Key finding:** `store_sqft` emerged as the most important feature (30% importance)

#### 2. XGBoost Regressor
- **Base parameters:**
  - Objective: `reg:squarederror`
  - Evaluation metric: RMSE
  - Tree method: `gpu_hist` (GPU acceleration)
  - Growth policy: `lossguide`

### Hyperparameter Optimization

Utilized **Optuna** for Bayesian hyperparameter optimization with 500 trials:

**Search Space:**
- Learning rate (`eta`): [1e-4, 0.1] (log-uniform)
- Min child weight: [1, 10]
- L2 regularization (`lambda`): [0, 10]
- L1 regularization (`alpha`): [0, 1]
- Gamma: [0, 1]
- Subsample: [0.5, 1]
- Column sampling: [0.1, 1] (log-uniform)
- Max depth: [5, 15]

**Optimal Parameters Found:**
```python
{
    'eta': 0.0998,
    'min_child_weight': 9,
    'lambda': 6.724,
    'alpha': 0.466,
    'gamma': 0.150,
    'subsample': 0.873,
    'colsample_bytree': 0.877,
    'max_depth': 14
}
```

### Model Validation
- **Cross-validation:** 5-fold KFold with shuffle
- **Early stopping:** 20 rounds for XGBoost training
- **Feature selection:** Removed 6 least important features based on XGBoost feature importance

## Technologies Used

- **Python 3.10**
- **Core Libraries:**
  - `pandas`, `numpy` - Data manipulation
  - `scikit-learn` - Model training and validation
  - `xgboost` - Gradient boosting implementation
  - `optuna` - Hyperparameter optimization
- **Visualization:**
  - `matplotlib`, `seaborn` - Data visualization and EDA

## Project Structure

```
Cost-Prediction/
├── README.md
└── practice2.ipynb      # Main analysis notebook
```

## Key Results

- Successfully implemented end-to-end regression pipeline
- Achieved performance improvements through systematic hyperparameter tuning using Optuna
- Identified `store_sqft` as the strongest predictor of campaign costs
- Demonstrated effective use of feature engineering to capture domain-specific relationships

## Future Improvements

- Implement ensemble stacking with multiple model types
- Explore polynomial features and interaction terms
- Conduct deeper analysis of residuals and error patterns
- Add model interpretation using SHAP values

## Author

Created as part of a machine learning practice assignment focusing on regression modeling and hyperparameter optimization techniques.
