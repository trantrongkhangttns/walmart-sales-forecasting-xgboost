# Walmart Sales Forecasting: End-to-End Time Series Optimization

## Project Overview
This project focuses on predicting weekly sales for 45 Walmart stores across 81 departments. The challenge lies in handling strong seasonal trends, holiday effects, and mitigating data leakage in a complex panel time-series dataset. 

By leveraging **XGBoost** and **Bayesian Optimization (Optuna)**, the final model significantly outperformed the traditional Seasonal Naive baseline, reducing the Weighted Mean Absolute Error (WMAE) by over **32%**.

## Key Highlights & Technical Decisions

* **Data Leakage Prevention:** Implemented strict Walk-Forward Validation. Time-series features (Lags, Rolling Means/Stds) were carefully constructed using isolated grouping (`groupby(['Store', 'Dept'])`) and shifted windows to ensure zero future data leaked into the training phase.
* **Ablation Study on Feature Engineering:** Conducted deep Error Analysis revealing massive variance in Dept 72 and during October. However, an ablation study proved that heavily hardcoding outlier flags (e.g., `Is_High_Error_Dept`, `Oct_Risk`) introduced noise and local overfitting. The model performed best when trusting pure historical signals (`lag_52`, `rolling_mean`).
* **Hyperparameter Tuning:** Utilized **Optuna** with the TPE algorithm and GPU acceleration (`tree_method='hist'`) to efficiently search the hyperparameter space. Early pruning callbacks were integrated to save computational resources.

## Results & Evaluation

The project uses **WMAE** (Weighted Mean Absolute Error), where holiday weeks are penalized 5x more than regular weeks.

| Model | WMAE Score | Improvement |
| :--- | :--- | :--- |
| **Seasonal Naive (Baseline)** | 1785.66 | - |
| **XGBoost (Default)** | 1337.50 | ~25.1% |
| **XGBoost (Optuna Tuned + Full Data)** | **1200.60** | **~32.7%** |

## Project Structure

- Data/
- Model/
- Notebook/
- Submission/
- .gitignore
- LICENSE
- README.md

## Tech Stack
* **Modeling & Tuning:** XGBoost, Optuna, Scikit-Learn
* **Data Processing:** Pandas, NumPy
* **Environment:** Python 3, Google Colab (NVIDIA T4/RTX 4050 GPU)

## 💡 How to Run
1. Clone the repository: `git clone https://github.com/trantrongkhangttns/walmart-sales-forecasting-xgboost.git`
2. Download the datasets from [Kaggle](https://www.kaggle.com/datasets/anggundwilestari/walmart-sales-forecasting?select=stores+-+Walmart+Sales+Forecast.csv) and place them in the `Data/` directory.
3. Run the notebook in the `Notebook/` folder to view the full pipeline, or use the pre-trained model in the `Model/` folder for immediate inference.
