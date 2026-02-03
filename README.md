
# Credit Default Analysis (Taiwan credit cards)

This repo now focuses on the UCI Taiwan credit card default dataset (April–September 2005) to explore drivers of default and build predictive models.

## Revisions after feedback
The code has been revised to address key issues:
- **Data leakage fixed**: Train/test split now occurs *before* scaling
- **OneHotEncoder**: Added `drop='if_binary'` for binary features (SEX)
- **Best model corrected**: XGBoost (not Gradient Boosting) is tuned and selected
- **Optimization metric**: Changed from ROC AUC to F2-score to prioritize recall/minimize false negatives
- **Fairness analysis added**: Demographic fairness metrics (TPR, FPR, selection rates by sex/education/marriage)
- **Consistent tree depth**
- **PAY_0 dominance addressed**

## Key artifacts
- `credit_default_analysis.ipynb` — end-to-end notebook: load/clean, EDA (demographics, credit, repayment), preprocessing, baseline models, tuned RF/GB/XGBoost, feature importance, threshold tuning, fairness analysis, ROC/confusion.

- Data: `CreditData/UCI_Credit_Card.csv` (30,000 clients, target `default.payment.next.month`).

## How to run
1) Open `credit_default_analysis.ipynb` in VS Code/Jupyter.
2) Run all cells (Python 3.9+). Required libs: pandas, numpy, scikit-learn, matplotlib, seaborn, xgboost.

```bash
pip install -r requirements.txt  # or pip install pandas numpy scikit-learn matplotlib seaborn xgboost
```
