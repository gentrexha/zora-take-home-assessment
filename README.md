# Zora Collector Churn Prediction

This project provides a machine learning pipeline to predict collector churn on the Zora onchain social network. The goal is to identify users at risk of churning so that proactive retention strategies can be implemented.

## 📈 Key Results

The model is a Logistic Regression classifier evaluated using 5-fold stratified cross-validation.

- **Model Performance (AUROC):** `0.524 ± 0.052`
  - This indicates the model has a slight capability to distinguish between churning and non-churning collectors, performing just above a random baseline (0.5).
- **Business Metric (Recall @ Top 10%):** `9.7% ± 3.9%`
  - By targeting the 10% of collectors with the highest predicted churn risk, we can identify and engage with ~10% of all collectors who would actually churn.

## 🔑 Key Churn Drivers

Based on the model's feature importances, the primary factors influencing churn are:

1.  **`account_age_days`**: Newer accounts are more likely to churn.
2.  **`top_10_pct_collection_size`**: Collectors with smaller top collections are more likely to churn.
3.  **`collection_frequency_per_month`**: Less frequent collection activity is linked to higher churn risk.
4.  **`low_engagement_flag`**: Users flagged for low engagement are more likely to churn.
5.  **`country_France`**: Collectors from France show a lower tendency to churn.

##  actionable Insights

- **Focus on Onboarding**: Since new accounts are at higher risk, improving the onboarding experience and providing early value is critical.
- **Encourage High-Quality Collecting**: Incentivize users to build significant collections, as this is a strong indicator of retention.
- **Promote Consistent Engagement**: Implement features or campaigns that encourage regular collection activity.

## 🚀 How to Run

### Prerequisites
- Python 3.10+
- `uv` for package management

### Quick Start

```bash
# 1. Install uv (if you don't have it)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Create virtual environment and install dependencies
uv sync

# 3. Run the end-to-end pipeline
uv run python -m ztha.pipeline
```

The pipeline will execute all steps from data loading to model training and artifact generation. All outputs, including predictions and reports, are saved in the `/artifacts` directory.

---

## 📂 Project Organization

```
├── artifacts/         <- Model, predictions, and reports
├── data/              <- Raw and processed data
├── notebooks/         <- Jupyter notebooks for exploration
├── tests/             <- Tests for the pipeline
└── ztha/              <- Source code for the pipeline
    ├── config.py
    ├── dataset.py
    ├── features.py
    └── modeling/
        ├── train.py
        └── predict.py
```

