# gold-price-forecast

The goal of this project is to train a model to forecast price of gold on the market for arbitrary days ahead. In this project we have set n=4 to do 4 days ahead forecasting.

## Setup

For running the notebooks on this repo Python 3.6 or above is recommended. 
You can install all the required libraries using:
```
pip install -r requirements.txt
```

## File Structure

- **gold_price_modeling.ipynb**: This notebook contains the pipeline for training the forecasting model. Data extraction, exploratory data analysis and visualizations, feature engineering and selection, modeling and evaluation is done in this notebook. In the last step a binary model file is saved in the model directory of this repo.

- **gold_price_inference.ipynb**: This notebook loads the saved model from the gold_price_modeling notebook and performs forecasting.

- **helper.py**: This is a helper module containing logics related to data extraction, features engineering, plotting, etc.

- **model/**: binary model files and feature list required for modeling are stored in this directory.

- **tickers.json**: simple dictionary containing set of financial indexes used in the data extraction stage.
