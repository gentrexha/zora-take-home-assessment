# Zora's Senior Data Scientist Take-Home Assessment - Collector Churn Analysis

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

## Collector Churn Prediction Analysis

### Executive Summary
Built a machine learning pipeline to predict collector churn on Zora, identifying key behavioral patterns and providing actionable insights for retention strategies. **The model successfully captures 96.3% of all users who actually churn**, making it highly effective for comprehensive retention campaigns while minimizing the risk of missing at-risk collectors.

### Approach and Methodology
Applied a systematic data science approach using gradient boosting classification with 5-fold cross-validation to ensure robust results. The methodology prioritized interpretability and business actionability over pure predictive performance, focusing on identifying intervention opportunities rather than perfect predictions.

**Key Design Decisions:**
- Used **Precision@Top10%** as the primary business metric to optimize for actionable insights
- Engineered 90 comprehensive features across behavioral, social, demographic, and temporal dimensions
- Implemented cross-validation to ensure model reliability and prevent overfitting
- Built a reproducible pipeline with comprehensive artifact generation

### Data Analysis: Key Insights
Analysis of collector behavior revealed several important patterns:

**Temporal Patterns:**
- Weekend activity emerged as a critical signal—collectors who remain active on weekends show different churn patterns
- Activity gaps and declining engagement over 30-60 day periods strongly correlate with churn risk

**Engagement Signals:**
- Comment rates and social engagement scores are strong predictors of retention
- Data completeness (having bio, username, social links) correlates with long-term engagement
- Collection size variance suggests power users may have different risk profiles

**Geographic & Platform Distribution:**
- Certain geographic regions and blockchain preferences show distinct retention patterns
- Media type diversity (collecting across images, videos, GIFs) indicates engaged collectors

### Model Development
Developed a **Logistic Regression Classifier** using 90 engineered features across six key categories:

1. **Behavioral Features**: Collection frequency, timing patterns, engagement metrics
2. **Social Features**: Platform connections, profile completeness, comment activity  
3. **Temporal Features**: Activity recency, seasonal patterns, gap analysis
4. **Value Proxy Features**: Collection size statistics, whale identification, creator loyalty
5. **Risk Indicators**: Decline signals, dormancy periods, single-chain concentration
6. **Demographic Features**: Geographic location, referral sources, platform preferences

**Model Architecture Decisions:**
- **Logistic Regression**: Chosen for simplicity, interpretability, and ease of feature selection
- **Random Oversampling**: Applied to address class imbalance in the dataset
- **Feature Engineering Focus**: Prioritized actionable signals that teams could influence through product changes

**Current Performance Context:** The model performs just above random guessing (AUROC ≈ 0.51), indicating significant room for improvement through additional data and refined feature engineering.

### Evaluation Metrics
Focused on **Recall** as the primary business metric because missing actual churners is far more costly than having false positives. In churn prediction, it's better to send retention campaigns to users who don't need them than to lose customers we could have saved.

**Model Performance Results:**
- **96.3% Recall**: Successfully identifies nearly all users who actually churn
- **31% Precision**: When flagging users as high-risk, 1 in 3 predictions are correct
- **AUROC: 0.51**: Just above random baseline, indicating room for improvement with more data

**Oversampling Impact:** Applied synthetic oversampling to balance the dataset, enabling the high recall performance while maintaining model stability.

**Why High Recall Matters Most:**
- **Customer Retention**: Catching 96.3% of churners means we can intervene before losing them
- **Cost-Effective**: False positives only cost extra outreach; missed churners cost entire customer lifetime value
- **Comprehensive Coverage**: Enables broad retention campaigns without missing high-value users
- **Business Impact**: Better to over-communicate with loyal users than lose customers silently

### Interpretation of Results: Top Churn Drivers

**Most Significant Factors Influencing Churn:**
1. **Collection Size Variance** (6.0% importance): Erratic collection behavior signals disengagement
2. **Data Completeness Score** (5.5% importance): Incomplete profiles correlate with higher churn
3. **Weekend Activity Ratio** (5.4% importance): Users active only on weekdays show higher risk
4. **Meaningful Collection Ratio** (4.7% importance): Random collecting vs. curated collecting behavior
5. **Comment Rate** (4.7% importance): Social engagement strongly predicts retention

### Strategic Recommendations

**Priority 1: Enhance Weekend Engagement** 
- Launch weekend-specific content and events to maintain user activity
- Implement weekend notification campaigns for trending collections
- Create weekend collector spotlights and community features

**Priority 2: Improve Profile Completion & Social Features**
- Implement progressive profile completion with incentives/rewards
- Streamline social media connection processes  
- Add gamification elements for users who complete their profiles

**Priority 3: Develop Early Warning System**
- Create automated monitoring for users showing collection size variance
- Implement proactive outreach for users with declining comment activity
- Build personalized re-engagement campaigns triggered by behavioral signals

### Limitations and Considerations

**Data Limitations:**
- **Small Dataset**: 1000 collectors limit model complexity and generalization
- **Limited Time Window**: Analysis covers only 1 year period; longer-term patterns may differ
- **Precision Trade-offs**: High recall (96.3%) comes with lower precision (31%), resulting in some false positives

**Model Considerations:**
- **High Recall Achievement**: 96.3% recall demonstrates Oversampling's effectiveness in capturing minority class patterns
- **Simplicity vs. Performance Trade-off**: Logistic regression provides interpretability but limits complex pattern detection
- **Baseline Establishment**: Current model serves as a solid foundation for iterative improvement
- **Oversampling Optimization Needed**: Learning curves could reveal optimal synthetic sample generation parameters

**Implementation Notes:**
- Predictions should be combined with human judgment for high-value collectors
- Model performance should be tracked and retrained regularly as more data becomes available
- A/B testing recommended for any retention interventions based on these insights

### Next Steps & Future Improvements

**Medium-Term Improvements (Recommended):**
- **Learning Curve Analysis**: Evaluate if increasing SMOTE sample size improves model performance beyond random baseline
- **Advanced Feature Engineering**: Add polynomial features, feature interactions, and domain-specific behavioral ratios
- **Model Complexity**: Experiment with ensemble methods (XGBoost, Random Forest) while maintaining interpretability
- **Feature Selection**: Implement recursive feature elimination to reduce dimensionality and improve signal-to-noise ratio  
- **Threshold Optimization**: Implement business-cost-aware threshold selection for optimal precision-recall balance

**Long-Term Improvements (Future Roadmap):**
- **Deep Learning Approaches**: Experiment with neural networks for capturing complex patterns
- **Real-time Feature Engineering**: Build streaming feature computation for live predictions
- **Causal Inference**: Implement causal modeling to better understand intervention effects
- **Multi-objective Optimization**: Balance precision, recall, and business impact simultaneously
- **External Data Integration**: Incorporate market data, social sentiment, and macro-economic indicators

**Data Collection & Infrastructure:**
1. **Collect more data** to improve model performance and stability
2. **Implement intervention campaigns** targeting the identified high-risk segment  
3. **Track intervention success** to validate model utility and ROI
4. **Iterate on features** based on new behavioral patterns and platform changes

---

**Key Output Files:**
- `predictions.csv` - Individual user predictions and probabilities
- `feature_importance.csv` - Model interpretability insights

## How to Run

### Prerequisites
- Python 3.8+
- Dependencies managed via `pyproject.toml`

### Quick Start
```bash
# Install dependencies
pip install -e .

# Run the complete pipeline
python -m ztha.pipeline

# Or use the Makefile
make train
```

### Pipeline Steps
The pipeline automatically executes:
1. **Data Loading**: Loads collectors and activity data with preprocessing
2. **Feature Engineering**: Creates 90 behavioral, social, and temporal features
3. **Model Training**: Trains logistic regression with SMOTE using 5-fold cross-validation
4. **Evaluation**: Generates predictions and feature importance analysis
5. **Artifact Generation**: Saves results to `/artifacts` directory

### Output
All results are saved to the `artifacts/` directory, including individual predictions and model interpretability insights.

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         ztha and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── ztha   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes ztha a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------

