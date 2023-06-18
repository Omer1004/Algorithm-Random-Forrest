import os
from tkinter import *
import tkinter.ttk as ttk
from tkinter import messagebox
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import requests
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.metrics import accuracy_score
from datetime import datetime, timedelta
import time
import json
import talib
import logging
from sklearn.impute import SimpleImputer
from ta.trend import macd
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

'''
Data Collection: The algorithm begins by collecting stock price data for specified stocks over a specific period and interval using Yahoo Finance API. It also gathers the Fear & Greed Index data.
Data Preprocessing: The data is then preprocessed to calculate the daily return and Exponential Moving Averages (EMA) for specific periods. The algorithm also includes the Fear & Greed Index data in the stock data.
Adding Technical Indicators: Following the preprocessing step, the algorithm adds technical indicators to the data such as the Relative Strength Index (RSI) and the Moving Average Convergence Divergence (MACD).
Preparing Training and Testing Datasets: The algorithm prepares training and testing datasets. The features (X) are the technical indicators and the target (y) is a binary variable indicating whether the stock price increased (1) or decreased (0) the next day.
Hyperparameter Tuning: The algorithm then tunes the hyperparameters of a Random Forest Classifier using RandomizedSearchCV, which randomly samples from a grid of hyperparameters to find the best combination.
Model Training and Evaluation: Using the best hyperparameters found, the Random Forest Classifier is trained on the training dataset and evaluated on the testing dataset. The model's performance is evaluated using various metrics like accuracy, precision, recall, F1 score, and ROC AUC score.
Model Deployment: Finally, the model is used to predict whether we should buy or sell each stock in the list based on their most recent data. The prediction is based on the assumption that if the model predicts a positive return (1), we should buy, and if it predicts a negative return (0), we should sell.
'''




def download_stock_data(stock_list, period, interval):
    data = {}
    for stock in stock_list:
        data[stock] = yf.download(stock, period=period, interval=interval)
    return data


def preprocess_data(data):
    for stock in data:
        data[stock]['Return'] = data[stock]['Close'].pct_change()
        data[stock]['EMA_9'] = data[stock]['Close'].ewm(span=9).mean()
        data[stock]['EMA_72'] = data[stock]['Close'].ewm(span=72).mean()
        data[stock]['EMA_89'] = data[stock]['Close'].ewm(span=89).mean()

        data[stock].dropna(inplace=True)

        # Calculate two-day returns
        data[stock]['TwoDay_Return'] = data[stock]['Close'].pct_change(periods=2)

        # Set the target based on two-day returns
        data[stock]['Target'] = np.where(data[stock]['TwoDay_Return'] > 0, 1, 0)

    return data


def prepare_xy(preprocessed_data):
    X = pd.concat([df.assign(Ticker=stock).loc[:, ['Ticker', 'EMA_9', 'EMA_72', 'EMA_89', 'RSI', 'MACD', 'MACD_signal', 'MACD_hist']] for stock, df in
                   preprocessed_data.items()], axis=0)
    y = pd.concat([df.assign(Ticker=stock).loc[:, ['Ticker', 'Target']] for stock, df in preprocessed_data.items()], axis=0)
    return X.set_index('Ticker', append=True), y.set_index('Ticker', append=True)



def tune_hyperparameters3(X_train, y_train):# by RandomizedSearchCV 20 min to run
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

    rf = RandomForestClassifier(random_state=42)
    rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid, n_iter=100, cv=3, verbose=2,
                                   random_state=42, n_jobs=-1)
    rf_random.fit(X_train, y_train.values.ravel())

    best_params = rf_random.best_params_

    return best_params



def train_and_evaluate(X_train, y_train, X_test, y_test):
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train, y_train['Target'].to_numpy().ravel())
    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test['Target'].to_numpy(), y_pred)

    return rf, accuracy, y_pred  # return y_pred as well



def get_current_features(stock_list, X):
    current_data = []
    for stock in stock_list:
        stock_data = yf.download(stock, period="3mo")
        #vix_data = download_vix_data('1d', '1d')
        #stock_data = add_vix_data(stock_data, vix_data)
        stock_data = preprocess_data({stock: stock_data})[stock]
        stock_data = add_technical_indicators({stock: stock_data})[stock]
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


def add_technical_indicators(data):
    for stock in data:
        # Calculate RSI
        data[stock]['RSI'] = talib.RSI(data[stock]['Close'], timeperiod=14)
        # Calculate MACD
        macd, macdsignal, macdhist = talib.MACD(data[stock]['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
        data[stock]['MACD'] = macd
        data[stock]['MACD_signal'] = macdsignal
        data[stock]['MACD_hist'] = macdhist

        # Drop rows with NaN values
        data[stock].dropna(inplace=True)

    return data


#VIX NOT IN USE YET
def download_vix_data(period='5y', interval='1d'):
    vix_ticker = '^VIX'
    vix_data = yf.download(vix_ticker, period=period, interval=interval)
    vix_data.reset_index(inplace=True)
    vix_data.rename(columns={"Date": "Date"}, inplace=True)
    return vix_data


def add_vix_data(data, vix_data):
    vix_data = vix_data[['Date', 'Close']]
    vix_data.columns = ['Date', 'VIX_Close']
    vix_data.set_index('Date', inplace=True)

    for stock in data:
        data[stock] = pd.concat([data[stock], vix_data], axis=1, join='inner')
        data[stock]['VIX_Close'] = data[stock]['VIX_Close'].fillna(method='ffill')
    return data



def calculate_model_metrics(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)

    return accuracy, precision, recall, f1, roc_auc, y_pred


def print_model_metrics(accuracy, precision, recall, f1, roc_auc, model_type=""):
    print(f"{model_type} Accuracy: {accuracy}")
    print(f"{model_type} Precision: {precision}")
    print(f"{model_type} Recall: {recall}")
    print(f"{model_type} F1 Score: {f1}")
    print(f"{model_type} ROC AUC Score: {roc_auc}")


def make_predictions(model, current_features, stock_list):
    predictions = model.predict(current_features)
    s = ""
    for i, stock in enumerate(stock_list):
        if predictions[i] == 1:
            s = s+"\n"+f"We should buy {stock} today."
        else:
            s = s + "\n" + f"We should sell {stock} today."
    return s


def run_trading_strategy3(model, data, X_test):
    # Initial cash
    initial_cash = 10000
    cash = initial_cash
    max_date = max([max(v.index) for v in data.values()])

    # Initialize trade records
    trade_records = pd.DataFrame(
        columns=['date', 'ticker', 'entry_price', 'exit_price', 'position_size', 'long_or_short', 'profit_loss',
                 'profit_loss_pct', 'number_of_shares'])

    num_of_trades = 0
    total_profit_loss = 0

    # Iterate through each day in the X_test DataFrame
    for date in X_test.index.get_level_values('Date').unique():
        # Get the features for this day
        X_day = X_test.loc[date]

        # Make predictions for this day
        y_pred = model.predict(X_day)

        # Iterate through each ticker
        for ticker in X_day.index:
            # Check if the model's prediction is 1 (buy)
            if y_pred[X_day.index.get_loc(ticker)] == 1:
                # Get the data for this ticker on this day
                current_day = data[ticker].loc[date]

                # Calculate position size and number of shares to buy, making sure it's a whole number
                max_cash_to_use = max(0.1 * cash, current_day['Open'])
                shares_to_buy = min(int(cash // current_day['Open']), int(max_cash_to_use // current_day['Open']))

                # Check if we can afford at least one share
                if shares_to_buy >= 1:
                    cash -= shares_to_buy * current_day['Open']

                    # Record the trade entry
                    new_row = {'date': date,
                               'ticker': ticker,
                               'entry_price': current_day['Open'],
                               'exit_price': np.nan,
                               'position_size': shares_to_buy * current_day['Open'],
                               'long_or_short': 'long',
                               'profit_loss': np.nan,
                               'profit_loss_pct': np.nan,
                               'number_of_shares': shares_to_buy}
                    trade_records = pd.concat([trade_records, pd.DataFrame([new_row])], ignore_index=True)

                    # Calculate the next day (1 day after current date)
                    next_day_date = date + pd.Timedelta(days=1)

                    # Check if the next day's data exists. If not, continue to the day after.
                    while next_day_date not in data[ticker].index:
                        # If next_day_date exceeds the max_date, break the loop
                        if next_day_date > max_date:
                            break
                        next_day_date += pd.Timedelta(days=1)

                    if next_day_date > max_date:
                        cash += shares_to_buy * current_day['Open']  # Return the cash back
                        trade_records = trade_records.drop(trade_records[(trade_records['date'] == date) & (
                                    trade_records['ticker'] == ticker)].index)  # Remove the last position
                        continue  # Skip this iteration and move to the next ticker

                    next_day = data[ticker].loc[next_day_date]

                    # Calculate profit/loss
                    profit_loss = ((next_day['Close'] - current_day['Open']) * shares_to_buy)-5 #5 for commision
                    cash_inflow = (shares_to_buy * next_day['Close']) -5
                    cash += cash_inflow

                    # Record the trade exit and calculate profit/loss
                    trade_records.loc[
                        (trade_records['date'] == date) & (trade_records['ticker'] == ticker), 'exit_price'] = next_day[
                        'Close']
                    trade_records.loc[
                        (trade_records['date'] == date) & (trade_records['ticker'] == ticker), 'profit_loss'] = \
                        cash_inflow - trade_records.loc[
                            (trade_records['date'] == date) & (trade_records['ticker'] == ticker), 'position_size']

                    # Calculate profit/loss percentage
                    trade_records.loc[
                        (trade_records['date'] == date) & (trade_records['ticker'] == ticker), 'profit_loss_pct'] = \
                        (trade_records.loc[
                             (trade_records['date'] == date) & (trade_records['ticker'] == ticker), 'exit_price'] - \
                         trade_records.loc[
                             (trade_records['date'] == date) & (trade_records['ticker'] == ticker), 'entry_price']) / \
                        trade_records.loc[
                            (trade_records['date'] == date) & (trade_records['ticker'] == ticker), 'entry_price'] * 100

                    # Update trade statistics
                    num_of_trades += 1
                    total_profit_loss += profit_loss

    # Calculate average profit/loss
    avg_profit_usd = trade_records['profit_loss'].mean()
    avg_profit_pct = trade_records['profit_loss_pct'].mean()

    # Calculate max loss and profit
    max_loss = trade_records['profit_loss'].min()
    max_profit = trade_records['profit_loss'].max()

    cash_dict = {
        "initial cash": initial_cash,
        "cash": cash,
        "profit_loss": cash - initial_cash,
        "avg_profit_usd": avg_profit_usd,
        "avg_profit_pct": avg_profit_pct,
        "max_loss": max_loss,
        "max_profit": max_profit,
        "number of trades": num_of_trades
    }

    # Convert dates to string format before writing to JSON
    trade_records['date'] = trade_records['date'].dt.strftime('%Y-%m-%d')

    with open('trading_records.json', 'w') as f:
        f.write(trade_records.to_json(orient='records'))

    with open('totalcashAPNL.json', 'w') as fcash:
        json.dump(cash_dict, fcash)

    return trade_records, cash_dict



def main():
    stprog = time.time()
    stock_list = ['QQQ', 'AAPL', 'TSLA', 'GOOGL', 'NVDA', 'META', 'MSFT', 'AMZN', 'AMD','NFLX']#'BABA' REMOVED
    period = '5y'
    interval = '1d'

    pd.set_option('display.max_columns', None)

    data = download_stock_data(stock_list, period, interval)
    vix_data = download_vix_data(period, interval)
    data = add_vix_data(data, vix_data)

    preprocessed_data = preprocess_data(data)

    print("data with indicators")
    data_with_indicators = add_technical_indicators(preprocessed_data)
    print(data_with_indicators)

    X, y = prepare_xy(data_with_indicators)

    X_trains, X_tests, y_trains, y_tests = [], [], [], []

    for stock in stock_list:
        X_stock = X[X.index.get_level_values('Ticker') == stock]
        y_stock = y[y.index.get_level_values('Ticker') == stock]
        X_train_stock, X_test_stock, y_train_stock, y_test_stock = train_test_split(X_stock, y_stock, test_size=0.02,
                                                                                    random_state=42, shuffle=False)
        X_trains.append(X_train_stock)
        X_tests.append(X_test_stock)
        y_trains.append(y_train_stock)
        y_tests.append(y_test_stock)

    X_train = pd.concat(X_trains)
    X_test = pd.concat(X_tests)
    y_train = pd.concat(y_trains)
    y_test = pd.concat(y_tests)

    print(X_test)
    print(y_test)

    model, accuracy, y_pred = train_and_evaluate(X_train, y_train, X_test, y_test)  # store y_pred

    #For normal model
    accuracy, precision, recall, f1, roc_auc, y_pred = calculate_model_metrics(model, X_test, y_test)
    print_model_metrics(accuracy, precision, recall, f1, roc_auc, model_type="Normal model")
    current_features = get_current_features(stock_list, X)
    normal_model_predictions = make_predictions(model, current_features, stock_list)
    print(normal_model_predictions)
    # get the start time
    st = time.time()

    best_params_load = False
    if best_params_load:
        best_params = tune_hyperparameters3(X_train, y_train)
        #best_params = random_search(X_train, y_train)
        # Save best_params
        with open('best_params.json', 'w') as f:
            json.dump(best_params, f)
    else:
        # Load best_params
        with open('best_params.json', 'r') as f:
            best_params = json.load(f)
    # get the end time
    et = time.time()
    # get the execution time
    elapsed_time = et - st
    print('Execution time of the hypertuning of the parameters:', elapsed_time, 'seconds')
    print('which is about ', elapsed_time/60)
    rf_best = RandomForestClassifier(**best_params)
    rf_best.fit(X_train, y_train['Target'].to_numpy().ravel())
    y_pred_best = rf_best.predict(X_test)


    accuracy_best, precision_best, recall_best, f1_best, roc_auc_best, y_pred_best = calculate_model_metrics(rf_best,X_test,y_test)
    print_model_metrics(accuracy_best, precision_best, recall_best, f1_best, roc_auc_best,model_type="Best params model")
    current_features = get_current_features(stock_list, X)

    best_model_predictions = make_predictions(rf_best, current_features, stock_list)
    print(best_model_predictions)
    # Initialize trade_records DataFrame
    trade_records = pd.DataFrame(
        columns=['date', 'ticker', 'entry_price', 'exit_price', 'position_size', 'long_or_short', 'profit_loss',
                 'profit_loss_pct','number_of_shares'])



    accuracy, precision, recall, f1, roc_auc, y_pred = calculate_model_metrics(model, X_test, y_test)
    print_model_metrics(accuracy, precision, recall, f1, roc_auc, model_type="Normal model")
    accuracy_best, precision_best, recall_best, f1_best, roc_auc_best, y_pred_best = calculate_model_metrics(rf_best,
                                                                                                             X_test,
                                                                                                             y_test)
    print_model_metrics(accuracy_best, precision_best, recall_best, f1_best, roc_auc_best,
                        model_type="Best params model")

    run_stratagy_load = True
    if run_stratagy_load:
        print("Started trading...")
        #trade_records_dict, cashpnl = run_trading_strategy4(rf_best, data_with_indicators, X_test)
        trade_records_dict, cashpnl = run_trading_strategy3(rf_best, data_with_indicators,X_test)
        #trade_records_dict, cashpnl = run_trading_strategy3(model, data_with_indicators, X_test)
        print("trade records: ", trade_records_dict)
        print(cashpnl)


    with open('trading_records.json', 'r') as f:
        trade_records_dict = json.load(f)

    with open('totalcashAPNL.json', 'r') as f:
        totalcashAPNL = json.load(f)
    print('Execution time of the hypertuning of the parameters:', elapsed_time, 'seconds')
    print('which is about ', elapsed_time / 60)
    etprog = time.time()
    # get the execution time
    elapsed_timeprog = etprog - stprog
    print('LOAD HYPERPARAMS:',best_params_load)
    print('LOAD TRADING STRATEGY:',run_stratagy_load)
    print('Execution time of the program until gui:', elapsed_timeprog, 'seconds')
    print('which is about ', elapsed_timeprog / 60)

    main_gui(model,stock_list,X_test,y_test,rf_best,normal_model_predictions,best_model_predictions,trade_records_dict,totalcashAPNL)


def show_trade_records_screen(trade_records, cash_dict):
    # Create a new window for trade records
    records_window = Toplevel()
    records_window.title("Trade Records")

    # Create a Treeview widget to display trade records
    tree = ttk.Treeview(records_window, columns=['date', 'ticker', 'entry_price', 'exit_price', 'position_size', 'long_or_short', 'profit_loss', 'profit_loss_pct', 'number_of_shares'], show="headings")
    for column in ['date', 'ticker', 'entry_price', 'exit_price', 'position_size', 'long_or_short', 'profit_loss', 'profit_loss_pct', 'number_of_shares']:
        tree.heading(column, text=column)

    # Add trade records to the Treeview
    for record in trade_records:
        tree.insert("", "end", values=list(record.values()))

    tree.pack(expand=True, fill="both")

    # Display total cash and profit/loss
    cash_label = Label(records_window, text=f"Initial Cash: {cash_dict['initial cash']}")
    cash_label.pack()

    final_cash_label = Label(records_window, text=f"Final Cash: {cash_dict['cash']}")
    final_cash_label.pack()

    profit_loss_label = Label(records_window, text=f"Total Profit/Loss: {cash_dict['profit_loss']}")
    profit_loss_label.pack()

    commission_cost_label = Label(records_window, text=f"Commission Cost: {cash_dict['number of trades'] * 5}")
    commission_cost_label.pack()

    avg_profit_usd_label = Label(records_window, text=f"Average Profit per Trade (USD): {cash_dict['avg_profit_usd']}")
    avg_profit_usd_label.pack()

    avg_profit_pct_label = Label(records_window, text=f"Average Profit per Trade (%): {cash_dict['avg_profit_pct']}")
    avg_profit_pct_label.pack()

    max_loss_label = Label(records_window, text=f"Max Loss on a Single Trade: {cash_dict['max_loss']}")
    max_loss_label.pack()

    max_profit_label = Label(records_window, text=f"Max Profit on a Single Trade: {cash_dict['max_profit']}")
    max_profit_label.pack()

    num_trades_label = Label(records_window, text=f"Number of Trades: {cash_dict['number of trades']}")
    num_trades_label.pack()

    profit_loss_button = ttk.Button(records_window, text="Show Profit/Loss Graph",
        command=lambda: show_profit_loss_graph(trade_records,cash_dict['initial cash']))
    profit_loss_button.pack()

    # Add a return button to go back to the main screen
    return_button = ttk.Button(records_window, text="Return to Main", command=records_window.destroy)
    return_button.pack()

def show_profit_loss_graph(trade_records, initial_cash):
    # Create a new window for the graph
    graph_window = Toplevel()
    graph_window.title("Profit/Loss Graph")

    # Create a Figure and a Canvas to display the graph
    fig = plt.Figure(figsize=(8, 5))
    canvas = FigureCanvasTkAgg(fig, master=graph_window)
    canvas.draw()
    canvas.get_tk_widget().pack(side="top", fill="both", expand=1)

    # Add the graph to the Figure
    ax = fig.add_subplot(111)

    # Sort trade records by date
    trade_records_sorted = sorted(trade_records, key=lambda x: x['date'])
    dates = [record['date'] for record in trade_records_sorted]
    profits_losses = [record['profit_loss'] for record in trade_records_sorted]

    # Calculate daily cash
    cash_values = [initial_cash]
    for profit_loss in profits_losses:
        cash_values.append(cash_values[-1] + profit_loss)

    # Trim the extra element from either dates or cash_values
    if len(dates) > len(cash_values):
        dates = dates[:len(cash_values)]
    elif len(cash_values) > len(dates):
        cash_values = cash_values[:len(dates)]

    # Plot the cash values
    ax.plot(dates, cash_values, label='Cash')

    # Set the labels and the title
    ax.set_xlabel('Date')
    ax.set_ylabel('Cash')
    ax.set_title('Cash over time')
    ax.legend()

    # Add a return button to go back to the main screen
    return_button = ttk.Button(graph_window, text="Return to Main", command=graph_window.destroy)
    return_button.pack()




def show_predictions_screen(predictions):
    # Create a new window for predictions
    pred_window = Toplevel()
    pred_window.title("Predictions")

    # Create a Label widget to display the predictions
    label = Label(pred_window, text=predictions)
    label.pack()

    # Add a return button to go back to the main screen
    return_button = ttk.Button(pred_window, text="Return to Main", command=pred_window.destroy)
    return_button.pack()

def main_gui(model, stock_list, X_test, y_test, rf_best, normal_model_predictions, best_model_predictions, trade_records,totalcashAPNL):
    root = Tk()
    root.title("Stock Prediction")

    Label(root, text="Normal Model Predictions:").pack()
    Label(root, text=normal_model_predictions).pack()

    Label(root, text="Best Params Model Predictions:").pack()
    Label(root, text=best_model_predictions).pack()

    Button(root, text="Show Trade Records",
           command=lambda: show_trade_records_screen(
               trade_records,
               totalcashAPNL
           )).pack()

    Button(root, text="Show Predictions (Normal Model)",
           command=lambda: show_predictions_screen(normal_model_predictions)).pack()

    Button(root, text="Show Accuracy (Normal Model)",
           command=lambda: show_accuracy(model, X_test, y_test)).pack()

    Button(root, text="Show Predictions (Best Params Model)",
           command=lambda: show_predictions_screen(best_model_predictions)).pack()

    Button(root, text="Show Accuracy (Best Params Model)",
           command=lambda: show_accuracy(rf_best, X_test, y_test)).pack()

    root.mainloop()



def show_accuracy(model, X_test, y_test):
    accuracy, _, _, _, _, _ = calculate_model_metrics(model, X_test, y_test)
    messagebox.showinfo("Accuracy", f"The model accuracy is {accuracy}")


def random_search(X_train, y_train):
    from sklearn.model_selection import RandomizedSearchCV
    '''
    Fitting 3 folds for each of 100 candidates, totalling 300 fits
    takes 12 min to run 
    Normal model Accuracy: 0.7885756676557863
    Normal model Precision: 0.7736625514403292
    Normal model Recall: 0.8245614035087719
    Normal model F1 Score: 0.7983014861995754
    Normal model ROC AUC Score: 0.7880337138025787
    Best params model Accuracy: 0.7841246290801187
    Best params model Precision: 0.7717842323651453
    Best params model Recall: 0.8157894736842105
    Best params model F1 Score: 0.7931769722814499
    Best params model ROC AUC Score: 0.7836477488902981
    '''

    random_grid = {
        'n_estimators': list(range(100, 2000)),
        'max_depth': list(range(1, 10)) + [None],
        'min_samples_split': list(range(2, 21)),
        'min_samples_leaf': list(range(1, 9)),
        'bootstrap': [True, False],
        'max_features': ['sqrt', 'log2', None]
    }

    rf = RandomForestClassifier(random_state=42)
    rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid, n_iter=100, cv=3, verbose=2, random_state=42, n_jobs=-1)
    rf_random.fit(X_train, y_train)

    best_params = rf_random.best_params_

    return best_params


def run_trading_strategy4(model, data, X_test):
    # Initialize variables
    initial_cash = 100000
    cash = initial_cash
    commission_cost = 5  # USD
    active_positions = {}
    max_date = max([max(v.index) for v in data.values()])
    trade_records = pd.DataFrame(columns=['date', 'ticker', 'entry_price', 'exit_price', 'position_size',
                                          'long_or_short', 'profit_loss', 'profit_loss_pct', 'number_of_shares'])

    # Iterate through each day in the X_test DataFrame
    for date in X_test.index.get_level_values('Date').unique():

        # Next, handle new entries for today
        X_day = X_test.loc[date]  # Get the features for this day
        y_pred = model.predict(X_day)  # Make predictions for this day

        # Iterate through each ticker
        for ticker in X_day.index:
            # Check if the model's prediction is 1 (buy), and that we have enough cash for at least one share
            current_day = data[ticker].loc[date]  # Get the data for this ticker on this day
            max_cash_to_use = max(0.1 * cash, current_day['Open'] + commission_cost)  # Only use 10% of available cash

            if y_pred[X_day.index.get_loc(ticker)] == 1 and cash > max_cash_to_use:
                # Calculate number of shares to buy, making sure it's a whole number and within our cash limit
                shares_to_buy = int((max_cash_to_use - commission_cost) // current_day['Open'])

                if shares_to_buy >= 1:  # Check if we can afford at least one share
                    position_size = shares_to_buy * current_day['Open']  # Calculate position size
                    cash -= position_size + commission_cost  # Deduct position's cost from our cash

                    # Add the new position to active positions
                    active_positions[ticker] = {'entry_date': date,
                                                'entry_price': current_day['Open'],
                                                'number_of_shares': shares_to_buy,
                                                'position_size': position_size,
                                                'exit_date': date + pd.Timedelta(
                                                    days=1)}  # Assume we hold the position for one day

                    # Record the trade entry
                    new_row = {'date': date,
                               'ticker': ticker,
                               'entry_price': current_day['Open'],
                               'exit_price': np.nan,
                               'position_size': position_size,
                               'long_or_short': 'long',
                               'profit_loss': np.nan,
                               'profit_loss_pct': np.nan,
                               'number_of_shares': shares_to_buy}
                    trade_records = pd.concat([trade_records, pd.DataFrame([new_row])], ignore_index=True)

        # Then, handle exits from active positions
        for ticker, position in active_positions.copy().items():  # Use copy() because we might change the dict size inside the loop
            if position['exit_date'] <= date:  # Only consider positions that should have been exited by now
                # Check if the exit day's data exists. If not, continue to the day after.
                next_day_date = position['exit_date']
                while next_day_date not in data[ticker].index:
                    # If next_day_date exceeds the max_date, break the loop
                    if next_day_date > max_date:
                        break
                    next_day_date += pd.Timedelta(days=1)

                # If the next_day_date exceeds max_date, return the cash and remove the position
                if next_day_date > max_date:
                    cash += position['position_size']  # Return the cash
                    del active_positions[ticker]  # Remove the position
                    continue

                current_day = data[ticker].loc[next_day_date]  # Get the data for this ticker on this day

                # Calculate profit/loss
                profit_loss = (current_day['Close'] - position['entry_price']) * position[
                    'number_of_shares'] - commission_cost
                cash += position['position_size'] + profit_loss  # Add position's worth and profit/loss to our cash

                # Update the record of this position with exit information
                trade_records.loc[
                    (trade_records['date'] == position['entry_date']) & (trade_records['ticker'] == ticker),
                    ['exit_price', 'profit_loss', 'profit_loss_pct']] = [current_day['Close'], profit_loss,
                                                                         profit_loss / position['position_size'] * 100]

                # Remove this position from active positions
                del active_positions[ticker]

    # Calculate final statistics
    cash_dict = {
        "initial cash": initial_cash,
        "cash": cash,
        "profit_loss": cash - initial_cash,
        "avg_profit_usd": trade_records['profit_loss'].mean(),
        "avg_profit_pct": trade_records['profit_loss_pct'].mean(),
        "max_loss": trade_records['profit_loss'].min(),
        "max_profit": trade_records['profit_loss'].max(),
        "number of trades": len(trade_records)
    }

    # Convert dates to string format before writing to JSON
    trade_records['date'] = trade_records['date'].dt.strftime('%Y-%m-%d')

    # Save results to JSON files
    with open('trading_records.json', 'w') as f:
        f.write(trade_records.to_json(orient='records'))

    with open('totalcashAPNL.json', 'w') as fcash:
        json.dump(cash_dict, fcash)

    return trade_records, cash_dict


main()




'''
to understand the error of the article 
to explain the user needs and that its testing is incorrect 
to add GUI
bonus to find an article with the correct mesuring metric
'''


'''
to explain the progress from 67% to the end


word 
to explain the process 

'''


'''
TO CHECK ON NORMALIZATIONS 
TO CHECK ON CHANGE OF STOCKS CHANGE NOT NECESSARLY 
TO ADD MORE MODELS AND COMPARE IT
'''