import yfinance as yf
import pandas as pd
import talib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler





def preprocess_data(data):
    for stock in data:
        data[stock].reset_index(inplace=True)
        data[stock].rename(columns={"Date": "date"}, inplace=True)
        data[stock] = data[stock][['date', 'Open', 'High', 'Low', 'Close', 'Volume']]
    return data

def add_technical_indicators(data):
    for stock in data:
        # Add EMA_9 and EMA_72
        print(f"Data before calculating technical indicators for {stock}:")
        print(data[stock])
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


def add_technical_indicators_for_current_features(data, period='10d'):
    for stock in data:
        # Download more data for the calculation of the indicators
        stock_data = yf.download(stock, period=period, interval='1d')

        # Calculate the indicators
        stock_data['EMA_9'] = talib.EMA(stock_data['Close'], timeperiod=9)
        stock_data['EMA_72'] = talib.EMA(stock_data['Close'], timeperiod=72)
        stock_data['RSI'] = talib.RSI(stock_data['Close'], timeperiod=14)
        macd, macdsignal, macdhist = talib.MACD(stock_data['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
        stock_data['MACD'] = macd
        stock_data['MACD_signal'] = macdsignal
        stock_data['MACD_hist'] = macdhist
        stock_data['ADX'] = talib.ADX(stock_data['High'], stock_data['Low'], stock_data['Close'], timeperiod=14)
        stock_data['ATR'] = talib.ATR(stock_data['High'], stock_data['Low'], stock_data['Close'], timeperiod=14)

        # Ensure that stock_data has the same columns as data[stock]
        for column in data[stock].columns:
            if column not in stock_data.columns:
                stock_data[column] = np.nan

        # Update the original data with the indicators of the most recent day
        data[stock] = stock_data.tail(1)[data[stock].columns]

    return data



def get_current_features(stock_list, vix_data, X,scaler):
    current_features = {}
    for stock in stock_list:
        # Download the most recent day's data
        stock_data = yf.download(stock, period='1d', interval='1d')
        stock_data = add_technical_indicators_for_current_features(stock_data)

        # Reset the index before the merge operation
        stock_data.reset_index(inplace=True)

        # Merge the stock data with the VIX data
        stock_data = stock_data.merge(vix_data, how='left', on='date')

        # Standardize the stock data
        stock_data = scaler.transform(stock_data[X.columns])

        # Add the standardized data to the current_features dictionary
        current_features[stock] = stock_data
    return current_features


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


    X_train['Date'] = X_train['Date'].astype(int) / 10 ** 9
    X_test['Date'] = X_test['Date'].astype(int) / 10 ** 9

    # Assume X_train is your training data
    scaler = StandardScaler()
    scaler.fit(X_train)
    rf_regressor = RandomForestRegressor(random_state=42)
    rf_regressor.fit(X_train, y_train)

    y_pred = rf_regressor.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error of the model: {mse}")
    r2 = r2_score(y_test, y_pred)
    print(f"R^2 score of the model: {r2:.2f}")
    print(f"Accuracy of the model: {r2 * 100:.2f}%")

    current_features = get_current_features(stock_list, vix_data, X,scaler)
    predictions = rf_regressor.predict(current_features)

    for i, stock in enumerate(stock_list):
        print(f"Predicted price for {stock} next week: {predictions[i]}")

if __name__ == '__main__':
    main()



#notes of algo1.0
'''
def run_trading_strategy(model, data_dict):
    """
    This function simulates a trading strategy using the trained model.

    Parameters:
    model (Model): The trained predictive model.
    data_dict (dict): The dictionary of preprocessed data with technical indicators.

    Returns:
    portfolio_value (float): The final value of the portfolio after running the trading strategy.
    """
    # Initialize portfolio
    portfolio = {stock: 0 for stock in data_dict.keys()}
    initial_cash = 10000  # starting amount in dollars
    cash = initial_cash

    # Iterate over each stock in the data
    for ticker, data in data_dict.items():
        # Iterate over each day in the data
        for i in range(len(data) - 1):
            current_day = data.iloc[i]
            next_day = data.iloc[i + 1]

            # Get model's prediction for the next day
            X_current = current_day[['EMA_9', 'EMA_72', 'EMA_89', 'RSI', 'MACD', 'MACD_signal', 'MACD_hist']].values.reshape(1, -1)
            prediction = model.predict(X_current)[0]

            # If the model predicts an up movement, buy the stock
            if prediction == 1:
                if cash >= next_day['Close']:
                    # Buy as much as possible
                    shares_to_buy = cash // next_day['Close']
                    portfolio[ticker] += shares_to_buy
                    cash -= shares_to_buy * next_day['Close']

            # If the model predicts a down movement, sell the stock
            elif prediction == -1:
                if portfolio[ticker] > 0:
                    # Sell all shares
                    cash += portfolio[ticker] * next_day['Close']
                    portfolio[ticker] = 0

    # At the end of the simulation, sell all remaining shares
    for ticker, shares in portfolio.items():
        last_day = data_dict[ticker].iloc[-1]
        cash += shares * last_day['Close']

    portfolio_value = cash
    return portfolio_value

'''