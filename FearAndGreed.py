import pandas as pd
import fear_and_greed
from datetime import datetime, timedelta

# Read the existing CSV file into a DataFrame
fear_greed_df = pd.read_csv('fear_and_greed_index.csv')
fear_greed_df['date'] = pd.to_datetime(fear_greed_df['date'], format="%d/%m/%Y")
fear_greed_df.set_index('date', inplace=True)

# Create a date range for the missing dates
start_date = datetime(2018, 2, 1)
missing_data_end_date = datetime(2022, 9, 8)
missing_data_date_range = pd.date_range(start=start_date, end=missing_data_end_date)

# Create a new DataFrame with the missing date range
missing_data_df = pd.DataFrame(missing_data_date_range, columns=['date'])
missing_data_df.set_index('date', inplace=True)

# Concatenate the missing data DataFrame with the existing data
fear_greed_df = pd.concat([missing_data_df, fear_greed_df], axis=0)

# Reset index
fear_greed_df.reset_index(inplace=True)

# Fill missing values with None
fear_greed_df = fear_greed_df.fillna(value={"fng_value": None, "fng_classification": None})

# Get the current Fear and Greed Index value
current_fear_greed = fear_and_greed.get()

# Print the current Fear and Greed Index value
print("Current Fear and Greed Index Value:", current_fear_greed.value)
print("Current Fear and Greed Index Description:", current_fear_greed.description)
print("Last Update:", current_fear_greed.last_update)
print(fear_greed_df)

# Use the DataFrame for your learning algorithm


'''FUNCTIONS FROM MAIN NOT IN USE'''
def get_fear_greed_data():
    fear_greed_df = pd.read_csv('fear_and_greed_index.csv')
    fear_greed_df['date'] = pd.to_datetime(fear_greed_df['date'], format="%d/%m/%Y")
    fear_greed_df.set_index('date', inplace=True)

    start_date = datetime(2018, 2, 1)
    end_date = datetime.now()
    date_range = pd.date_range(start=start_date, end=end_date)

    fear_greed_df = fear_greed_df.reindex(date_range, fill_value=None)
    fear_greed_df.index.name = 'date'
    fear_greed_df.reset_index(inplace=True)

    return fear_greed_df


def preprocess_data_withFG(data,fear_greed_data):
    for stock in data:
        data[stock].reset_index(inplace=True)
        data[stock].rename(columns={"Date": "date"}, inplace=True)
        data[stock] = data[stock].merge(fear_greed_data, how='left', on='date')
        data[stock]['Return'] = data[stock]['Close'].pct_change()
        data[stock]['EMA_9'] = data[stock]['Close'].ewm(span=9).mean()
        data[stock]['EMA_72'] = data[stock]['Close'].ewm(span=72).mean()
        data[stock]['EMA_89'] = data[stock]['Close'].ewm(span=89).mean()

        data[stock].dropna(inplace=True)
        data[stock]['Target'] = np.where(data[stock]['Return'] > 0, 1, 0)

    return data