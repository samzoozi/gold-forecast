# gold-price-forecast

The goal of this project is to train a model to forecast price of gold on the market for arbitrary days ahead. 2 main approaches are explored. 1) univariate timeseries forecasting using ARIMA model, 2) machine learning based approach using feature engineering and selection. A linear regression and non-liner gradient boosting decision trees are used for modeling the data.

## Setup

For running the notebooks on this repo Python 3.6 or above is recommended.
You can install all the required libraries using:

```
pip install -r requirements.txt
```

## File Structure

- **gold_price_arima.ipynb**: The code for forecasting gold price using linear ARIMA model.

- **gold_price_ML_model.ipynb**: The pipeline for training the forecasting ML model for gold price prediction. Data extraction, exploratory data analysis and visualizations, feature engineering and selection, modeling and evaluation is done in this notebook. In the last step a binary model file is saved in the model directory of this repo.

- **gold_price_inference.ipynb**: This notebook loads the saved model from the gold_price_ML_models notebook and performs forecasting.

- **helper.py**: This is a helper module containing logics related to data extraction, features engineering, plotting, etc.

- **model/**: binary model files and feature list required for modeling are stored in this directory.

- **tickers.json**: simple dictionary containing set of financial indexes used in the data extraction stage.

# Evaluation Results

We can see rmse scores for linear regression model and non-linear xgboost models in prediction of gold prices at n days ahead. We also have used ARMA time series forecasting model (explained in the gold_price_arima notebook) for forecasting of 1 day ahead.

| Model             | Days ahead | RMSE   |
| ----------------- | ---------- | ------ |
| XGBoost           | 1          | 0.0074 |
| Linear Regression | 1          | 0.0075 |
| XGBoost           | 2          | 0.0101 |
| Linear Regression | 2          | 0.0102 |
| XGBoost           | 3          | 0.0122 |
| Linear Regression | 3          | 0.0121 |
| XGBoost           | 4          | 0.0137 |
| Linear Regression | 4          | 0.0140 |

Evaluation results of the gold price value prediction on the test set are as follows:

| Model             | Days ahead | RMSE  |
| ----------------- | ---------- | ----- |
| XGBoost           | 1          | 13.67 |
| Linear Regression | 1          | 13.26 |
| ARIMA             | 1          | 12.15 |

The final results provides an interesting intuition that for 1 day ahead predicion, simple linear models are outperforming a complex non-linear model like XGboost. This can be also attributed to relatively small size of the dataset and also high volatility in financial data. Evaluation on absolute gold price data on validation set showed that ARMA model using only the gold price signal outperformed the multivariate linear regression model using various number of other indicators. Another interesting approach would be using vector auto regression model on this problem, which unfortunately I did not find the time to implement and compare.
