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
