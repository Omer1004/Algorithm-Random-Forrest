import yfinance as yf
import pandas as pd
import talib
import numpy as np


class StockData:
    def __init__(self, stock_list, period, interval):
        self.stock_list = stock_list
        self.period = period
        self.interval = interval
        self.data = self.download_stock_data()

    def download_stock_data(self):
        data = {}
        for stock in self.stock_list:
            data[stock] = yf.download(stock, period=self.period, interval=self.interval)
        return data

    def preprocess_data(self):
        for stock in self.data:
            self.data[stock]['Return'] = self.data[stock]['Close'].pct_change()
            self.data[stock]['EMA_9'] = self.data[stock]['Close'].ewm(span=9).mean()
            self.data[stock]['EMA_72'] = self.data[stock]['Close'].ewm(span=72).mean()
            self.data[stock]['EMA_89'] = self.data[stock]['Close'].ewm(span=89).mean()

            self.data[stock].dropna(inplace=True)
            self.data[stock]['Target'] = np.where(self.data[stock]['Return'] > 0, 1, 0)

        return self.data

    def add_technical_indicators(self):
        for stock in self.data:
            # Calculate RSI
            self.data[stock]['RSI'] = talib.RSI(self.data[stock]['Close'], timeperiod=14)

            # Calculate MACD
            macd, macdsignal, macdhist = talib.MACD(self.data[stock]['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
            self.data[stock]['MACD'] = macd
            self.data[stock]['MACD_signal'] = macdsignal
            self.data[stock]['MACD_hist'] = macdhist

            # Drop rows with NaN values
            self.data[stock].dropna(inplace=True)

        return self.data

    def prepare_xy(self):
        X = pd.concat([df.loc[:, ['EMA_9', 'EMA_72', 'EMA_89', 'RSI', 'MACD', 'MACD_signal', 'MACD_hist']] for df in self.data.values()], axis=0)
        y = pd.concat([df.loc[:, 'Target'] for df in self.data.values()], axis=0)
        return X, y


