# NHANES-Diabetes-Screening-Model

## Project Overview

This project builds a machine learning pipeline to predict whether a patient has diabetes using NHANES data from the 2009-2012 survey cycles. The task is framed as a binary classification problem, where the target variable is recoded as:

- `0`: no diabetes
- `1`: has diabetes

Because the cost of missing diabetic patients is high, the primary evaluation metric is **recall**. Precision, F1 score, and PR-AUC are used as supporting metrics to evaluate the tradeoff between identifying as many diabetic patients as possible and controlling false positives.

## Data Source

The project uses NHANES data from the 2009-2012 cycles. The feature set includes demographic and physiological variables, such as:

- Demographics: gender, age, home ownership, income, etc.
- Physiological measures: weight, height, BMI, average systolic blood pressure, urine flow, etc.
- Gender-specific variables: age at first baby, number of pregnancies, number of babies, etc.

To keep the analysis cross-sectional rather than panel-based, only one record per participant ID was kept, using the earliest available survey year.

## Preprocessing and Feature Engineering

The preprocessing pipeline included:

- Removing duplicate participant records across survey years
- Checking for non-zero variance features
- Exploratory data analysis on missingness, skewness, correlation with the target, and multicollinearity
- Removing highly collinear variables
- Removing variables with excessive missingness caused by lack of measurement
- Applying log transformations to skewed variables
- Recoding gender-specific variables
- One-hot encoding categorical variables
- Removing `SexNumPartYear` after identifying it as unsuitable for modeling

## Modeling Strategy

The project compared multiple modeling and preprocessing strategies across several experiments.

### Experiment 1: Complete-Case Analysis

A complete-case approach was tested first, but it produced low recall and only 3 out of 5 folds were successfully calculated. This showed that imputation was necessary because too much information was lost when dropping rows with missing values.

### Experiment 2: Imputation and Class Imbalance Handling

The next experiment added imputation and tested class imbalance strategies. The baseline model performed poorly, with recall around 0.077, confirming that the diabetes outcome was difficult to predict without better handling of missing data and class imbalance.

Tree-based models were paired with oversampling, while logistic regression models were tested with different imbalance strategies and regularization settings.

### Experiment 3: Model Comparison

The main models tested included:

- Baseline model
- XGBoost with oversampling
- AdaBoost with oversampling
- Random Forest with oversampling
- Logistic regression
- Logistic regression with L1 regularization
- Logistic regression with L2 regularization
- Logistic regression with different class imbalance strategies

The strongest recall performance came from:

- **XGBoost + oversampling**, with recall around **0.875**
- **Logistic regression with L1 + oversampling**, with recall around **0.846**

Given the recall-focused objective, XGBoost with oversampling was selected for further tuning.

### Experiment 4: Hyperparameter Tuning

XGBoost with oversampling was tuned using nested cross-validation instead of a simple validation split. The tuned model showed lower recall under a different threshold, suggesting that threshold choice strongly affected model behavior.

### Experiment 5: Parameter Refitting

The tuned XGBoost parameters were refit and compared with the earlier XGBoost setup. The selected tuned configuration included:

```python
{
    "colsample_bytree": 0.8,
    "gamma": 0.0,
    "learning_rate": 0.03,
    "max_depth": 4,
    "min_child_weight": 1,
    "n_estimators": 300,
    "reg_alpha": 0.0,
    "reg_lambda": 5.0,
    "subsample": 1.0
}
```

The tuned parameter setting performed better in the final cross-validation comparison, with the selected threshold around `0.205731`.

## Evaluation Focus

The project prioritized recall because false negatives are especially costly in diabetes screening. The key evaluation question was not only which model achieved the highest raw accuracy, but which model could identify the greatest number of diabetic patients while maintaining a reasonable precision and F1 score.

Threshold selection was treated as part of the modeling process, rather than assuming the default 0.5 classification threshold.

## Key Takeaways

- Recall is the most important metric for this task because missing diabetic patients has a high cost.
- Complete-case analysis was not sufficient because missingness removed too much useful data.
- Imputation was necessary for stable model evaluation.
- Class imbalance handling substantially affected model performance.
- Tree-based models, especially XGBoost with oversampling, performed best for the recall-focused objective.
- Threshold tuning was important because the default 0.5 threshold may be too conservative for screening-oriented prediction.
- Logistic regression with L1 regularization also performed competitively, suggesting that sparse linear models can be useful as interpretable baselines.

## Next Steps

Potential next steps include:

- Test the tuned model on held-out data
- Add calibration analysis to compare predicted probabilities with observed diabetes risk
- Report confusion matrices at selected thresholds
- Compare recall, precision, F1, and PR-AUC across thresholds
- Add feature importance analysis for model interpretation
- Clarify whether the model is intended for screening support, risk stratification, or exploratory analysis

## Repository Structure

A suggested repository structure:

```text
.
├── data/                 # Raw or processed NHANES data, if shareable
├── notebooks/            # Modeling and preprocessing notebooks
├── src/                  # Reusable preprocessing/modeling functions
├── outputs/              # Model results, plots, and tables
├── README.md             # Project overview
└── requirements.txt      # Python package requirements
```

## Notes

This project is a health-related predictive modeling exercise. The model should not be interpreted as a clinical diagnostic tool without external validation, calibration, and domain review.
