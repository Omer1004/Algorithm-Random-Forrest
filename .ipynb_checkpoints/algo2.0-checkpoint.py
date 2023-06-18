import yfinance as yf
import pandas as pd
import talib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.impute import SimpleImputer

def preprocess_data(data):
    for stock in data:
        data[stock].reset_index(inplace=True)
        data[stock].rename(columns={"Date": "date"}, inplace=True)
        data[stock] = data[stock][['date', 'Open', 'High', 'Low', 'Close', 'Volume']]
    return data

def add_technical_indicators(data):
    for stock in data:
        # Add EMA_9 and EMA_72
        data[stock]['EMA_9'] = talib.EMA(data[stock]['Close'], timeperiod=9)
        data[stock]['EMA_72'] = talib.EMA(data[stock]['Close'], timeperiod=72)

        # Add RSI
        data[stock]['RSI'] = talib.RSI(data[stock]['Close'], timeperiod=14)

        # Add MACD, MACD_signal, and MACD_hist
        macd, macdsignal, macdhist = talib.MACD(data[stock]['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
        data[stock]['MACD'] = macd
        data[stock]['MACD_signal'] = macdsignal
        data[stock]['MACD_hist'] = macdhist

        # Add ADX
        data[stock]['ADX'] = talib.ADX(data[stock]['High'], data[stock]['Low'], data[stock]['Close'], timeperiod=14)

        # Add ATR
        data[stock]['ATR'] = talib.ATR(data[stock]['High'], data[stock]['Low'], data[stock]['Close'], timeperiod=14)

        # Drop rows with NaN values
        data[stock].dropna(inplace=True)

    return data

def get_current_features(stock_list, vix_data, X):
    current_data = []
    for stock in stock_list:
        stock_data = yf.download(stock, period="3mo")

        stock_data = preprocess_data({stock: stock_data})[stock]
        stock_data = add_technical_indicators({stock: stock_data})[stock]
        stock_data = stock_data.merge(vix_data, how='left', on='date')
        stock_data = stock_data.tail(1)

        if not stock_data.empty:  # Check if the DataFrame is not empty
            current_data.append(stock_data)
    if not current_data:  # Check if the list is not empty
        raise ValueError("No valid data available.")
    current_df = pd.concat(current_data)
    current_df = current_df[X.columns]
    imputer = SimpleImputer(strategy="mean")
    current_df = pd.DataFrame(imputer.fit_transform(current_df), columns=current_df.columns)
    return current_df


def download_vix_data(period='5y', interval='1d'):
    vix_ticker = '^VIX'
    vix_data = yf.download(vix_ticker, period=period, interval=interval)
    vix_data.reset_index(inplace=True)
    vix_data.rename(columns={"Date": "Date"}, inplace=True)
    return vix_data


def add_vix_data(data, vix_data):
    vix_data = vix_data[['Date', 'Close']]
    vix_data.columns = ['Date', 'VIX_Close']

    for stock in data:
        data[stock] = data[stock].merge(vix_data, how='left', on='Date')
        data[stock]['VIX_Close'] = data[stock]['VIX_Close'].fillna(method='ffill')

    return data


def create_target_labels(data, shift=5):
    for stock in data:
        data[stock]['Future_Price'] = data[stock]['Close'].shift(-shift)
        data[stock].dropna(inplace=True)
    return data

def prepare_xy(data):
    X = pd.concat([df.drop(['Future_Price'], axis=1) for df in data.values()], axis=0)
    y = pd.concat([df['Future_Price'] for df in data.values()], axis=0)
    return X, y


def calculate_indicators(data):
    for stock in data:
        # Calculate EMA
        data[stock]['EMA_9'] = talib.EMA(data[stock]['Close'], timeperiod=9)
        data[stock]['EMA_72'] = talib.EMA(data[stock]['Close'], timeperiod=72)

        # Calculate RSI
        data[stock]['RSI'] = talib.RSI(data[stock]['Close'], timeperiod=14)

        # Calculate MACD
        macd, macdsignal, macdhist = talib.MACD(data[stock]['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
        data[stock]['MACD'] = macd
        data[stock]['MACD_signal'] = macdsignal
        data[stock]['MACD_hist'] = macdhist

        # Calculate ADX
        data[stock]['ADX'] = talib.ADX(data[stock]['High'], data[stock]['Low'], data[stock]['Close'], timeperiod=14)

        # Calculate ATR
        data[stock]['ATR'] = talib.ATR(data[stock]['High'], data[stock]['Low'], data[stock]['Close'], timeperiod=14)

        # Drop rows with NaN values
        data[stock].dropna(inplace=True)

    return data






def download_stock_data(stock_list, period, interval):
    data = {}
    for stock in stock_list:
        data[stock] = yf.download(stock, period=period, interval=interval)
    return data



def main():
    stock_list = ['QQQ', 'AAPL', 'TSLA', 'GOOGL', 'NVDA', 'META', 'MSFT', 'AMZN', 'AMD']
    period = '5y'
    interval = '1d'

    data = download_stock_data(stock_list, period, interval)
    # Calculate technical indicators, preprocess data, train and evaluate model, and generate predictions
    vix_data = download_vix_data()
    data = calculate_indicators(data)
    data = add_vix_data(data, vix_data)
    data = create_target_labels(data)

    X, y = prepare_xy(data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

    rf_regressor = RandomForestRegressor(random_state=42)
    rf_regressor.fit(X_train, y_train)

    y_pred = rf_regressor.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error of the model: {mse}")
    r2 = r2_score(y_test, y_pred)
    print(f"R^2 score of the model: {r2:.2f}")
    print(f"Accuracy of the model: {r2 * 100:.2f}%")

    current_features = get_current_features(stock_list, vix_data, X)
    predictions = rf_regressor.predict(current_features)

    for i, stock in enumerate(stock_list):
        print(f"Predicted price for {stock} next week: {predictions[i]}")

if __name__ == '__main__':
    main()
