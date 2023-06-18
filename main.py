import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


from stock_data import StockData
from random_forest_model import RandomForestModel

stock_list = ['TSLA', 'AAPL', 'MSFT', 'AMZN', 'META']
period = '5y'
interval = '1d'

stock_data = StockData(stock_list, period, interval)
preprocessed_data = stock_data.preprocess_data()
data_with_indicators = stock_data.add_technical_indicators()
X, y = stock_data.prepare_xy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_model = RandomForestModel(X_train, y_train)
best_params = rf_model.tune_hyperparameters()
accuracy = rf_model.train_and_evaluate(X_test, y_test)
print(f"Accuracy of the model: {accuracy:.2f}")

current_features = rf_model.get_current_features(stock_list, X)
predictions = rf_model.predict(current_features)

for i, stock in enumerate(stock_list):
    if predictions[i] == 1:
        print(f"We should buy {stock} today.")
    else:
        print(f"We should sell {stock} today.")
