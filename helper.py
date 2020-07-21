from yahoofinancials import YahooFinancials
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def extract_data(start_date, end_date, tickers):
    values = pd.DataFrame({'date': pd.date_range(start=start_date, end=end_date)}).iloc[:-1]
    values['date'] = pd.to_datetime(values['date'])

    for ticker in tickers:
        data = YahooFinancials(tickers[ticker])
        data = data.get_historical_price_data(start_date, end_date, "daily")
        df = pd.DataFrame(data[tickers[ticker]]['prices'])[['formatted_date', 'adjclose']]
        df.columns = ['ticker_date', ticker]
        df['ticker_date'] = pd.to_datetime(df['ticker_date'])
        values = values.merge(df, how='left', left_on='date', right_on='ticker_date')
        values = values.drop(labels='ticker_date', axis=1)
    values = values.fillna(method="ffill", axis=0)
    values = values.fillna(method="bfill", axis=0)
    cols = values.columns.drop('date')
    values[cols] = values[cols].apply(pd.to_numeric).round(decimals=2)
    return values


def sorted_correlations_with_gold(corr):
    corr_sorted = abs(corr['Gold']).sort_values(ascending=False)
    return corr_sorted


def create_features(values):
    # imp = sorted_correlations_with_gold(values.corr()).index[:6].values
    imp = ['Gold', 'Silver', '10 Yr US T-Note futures', 'Copper',
           'Euronext100', 'Platinum', 'Volatility Index']
    # short term roc
    features = pd.DataFrame(values['date'])
    for lag in [1, 2, 3, 5, 7, 14, 21, 30]:
        features = pd.concat([features, values[values.columns[1:]].pct_change(
            lag).add_suffix("_t-%d" % lag)], axis=1)
    # long term roc
    for month in [2, 3, 6]:
        features = pd.concat([features, values[imp].pct_change(30*month).add_suffix("_%dmonths" % month)], axis=1)
    # moving avg features for gold
    moving_avg = pd.DataFrame(values['date'], columns=['date'])
    moving_avg['date'] = pd.to_datetime(moving_avg['date'], format='%Y-%b-%d')
    for lag in [15, 30, 60, 90, 180]:
        moving_avg['Gold_%dSMA' % lag] = (values['Gold']/(values['Gold'].rolling(window=lag).mean()))-1
    for lag in [30, 60, 90, 180]:
        moving_avg['Gold_%dEMA' % lag] = (
            values['Gold']/(values['Gold'].ewm(span=lag, adjust=True, ignore_na=True).mean()))-1
    moving_avg = moving_avg.dropna(axis=0)
    features = pd.merge(left=features, right=moving_avg, how='left', on='date')

    # stochastic oscilator features (14 days)
    oscilator = pd.DataFrame(values['date'], columns=['date'])
    oscilator['date'] = pd.to_datetime(oscilator['date'], format='%Y-%b-%d')
    for feat in imp:
        oscilator['%s_osc' % feat] = (values['%s' % feat] - (values['%s' % feat].rolling(window=14).min())) / (
            (values['%s' % feat].rolling(window=14).max()) - (values['%s' % feat].rolling(window=14).min()))
    features = pd.merge(left=features, right=oscilator, how='left', on='date')
    features = features.dropna()

    # rate of change ratio features
    roc_ratio = pd.DataFrame(features['date'], columns=['date'])
    roc_ratio['date'] = pd.to_datetime(roc_ratio['date'], format='%Y-%b-%d')
    for feat in imp:
        for n in [2, 3, 5, 7, 14, 21]:
            roc_ratio['%s_rocr_%d' % (feat, n)] = features['%s_t-1' % feat]/features['%s_t-%d' % (feat, n)]
    roc_ratio = roc_ratio.fillna(method="ffill", axis=0)
    roc_ratio = roc_ratio.fillna(method="bfill", axis=0)
    roc_ratio = roc_ratio.replace([np.inf, -np.inf], 0)

    features = pd.merge(left=features, right=roc_ratio, how='left', on='date')
    return features


def remove_correlated_features(values, correlation_matrix, threshold=0.9):
    correlated_features = set()
    for i in range(len(correlation_matrix.columns)):
        # excluding gold feature (index 0)
        for j in range(1, i):
            if abs(correlation_matrix.iloc[i, j]) > threshold:
                colname = correlation_matrix.columns[i] if abs(correlation_matrix['Gold'][correlation_matrix.columns[i]]) < abs(
                    correlation_matrix['Gold'][correlation_matrix.columns[j]]) else correlation_matrix.columns[j]
                correlated_features.add(colname)
    print("corelated features to be removed: %s" % correlated_features)
    newcols = [x for x in values.columns if x not in correlated_features]
    values = values[newcols]
    return values


def create_lr_plot(validation_scores, training_scores, validation_loss, training_loss, n_features_used):
    f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharex=True, figsize=(18, 6))

    ax1.plot(validation_scores, label='validation')
    ax1.plot(training_scores, label='training')
    ax1.set_title("R2 scores of training and validation")
    ax1.set_xlabel("feature index")
    ax1.set_ylabel("R2 Score")
    ax1.set_ylabel("n_features")
    ax1.legend()

    ax2.plot(validation_loss, label='validation loss')
    ax2.plot(training_loss, label='training loss')
    ax2.set_title("MSE of training and validation")
    ax2.legend()

    ax3.plot(n_features_used, label="n_features_used")
    ax3.set_title("number of features used in the model")
    ax3.set_xlabel("feature index")


def create_xgb_plot(validation_scores, training_scores, n_features_used, feature_columns_sorted):
    f, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(15, 6))
    ax1.plot(validation_scores, label='validation loss')
    ax1.plot(training_scores, label='training loss')
    ax1.set_title("MSE scores of training and validation by varying the indicators")
    ax1.set_xlabel("feature index")
    ax1.set_ylabel("MSE Score")
    ax2.plot(n_features_used, label="n_features_used")
    ax2.set_title("number of features used in the model")
    ax2.set_xlabel("feature index")
    idx = [int(x) for x in np.linspace(0, len(feature_columns_sorted)-1, 30)]
    ax2.set_xticks(idx)
    ax2.set_xticklabels([feature_columns_sorted[x] for x in idx], minor=False, rotation=90, size=10)
    ax1.set_ylabel("n_features")
    ax1.legend()
