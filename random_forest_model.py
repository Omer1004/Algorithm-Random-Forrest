import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
import yfinance as yf
from stock_data import StockData

class RandomForestModel:
    def __init__(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.model = RandomForestClassifier(random_state=42)

    def tune_hyperparameters(self):
        n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
        max_features = ['auto', 'sqrt']
        max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
        max_depth.append(None)
        min_samples_split = [2, 5, 10]
        min_samples_leaf = [1, 2, 4]
        bootstrap = [True, False]
        random_grid = {'n_estimators': n_estimators,
                       'max_features': ['sqrt'],
                       'max_depth': max_depth,
                       'min_samples_split': min_samples_split,
                       'min_samples_leaf': min_samples_leaf,
                       'bootstrap': bootstrap}

        rf_random = RandomizedSearchCV(estimator=self.model, param_distributions=random_grid, n_iter=100, cv=3, verbose=2,
                                       random_state=42, n_jobs=-1)
        rf_random.fit(self.X_train, self.y_train.ravel())
        best_params = rf_random.best_params_
        self.model.set_params(**best_params)

        return best_params

    def train_and_evaluate(self, X_test, y_test):
        self.model.fit(self.X_train, self.y_train.ravel())
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        return accuracy

    def get_current_features(self, stock_list, X):
        current_data = []
        for stock in stock_list:
            stock_data = yf.download(stock, period="3mo")

            stock_data = StockData.preprocess_data({stock: stock_data})[stock]
            stock_data = StockData.add_technical_indicators({stock: stock_data})[stock]
            stock_data = stock_data.tail(1)

            if not stock_data.empty:
                current_data.append(stock_data)
        if not current_data:
            raise ValueError("No valid data available.")
        current_df = pd.concat(current_data)
        current_df = current_df[X.columns]
        imputer = SimpleImputer(strategy="mean")
        current_df = pd.DataFrame(imputer.fit_transform(current_df), columns=current_df.columns)
        return current_df

    def predict(self, X):
        return self.model.predict(X)
