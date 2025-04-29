# Importing dependencies
import csv

import requests   # The requests library is used to send HTTP requests to websites or APIs.
# requests.get() sends a GET request to the specified URL.
# Headers in requests : Headers are metadata sent alon  with your request. They help the server understand :
# Who is making the request
# What data format is accepted (JSON, XML)
# If you're logged in (authorization token)
# Cache control, language preference, and more.

import time
import random
import pandas as pd

# Creating a fake browser header otherwise nse server will block the request
user_agents = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.1 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0 Safari/537.36",
]

header = {
    "User-Agent" : random.choice(user_agents)
}

# Function to get the stock data from nse
def get_nse_stock_price(symbol : str = "Reliance"):
    """Function to get live stock prices of a particular symbol calling the api.
    symbol : str = 'Reliance' """

    # NSE's quote API returns live data about the stock
    url = f"https://www.nseindia.com/api/quote-equity?symbol={symbol.upper()}"
    try:
        # This sends a GET request to the server (like visiting the stock's page)
        response = requests.get(url, headers = header)

        # Parse the response as JSON
        print(response)
        print('========================================================================================')
        data = response.json()
        print(data)

        # Getting the relevant price details
        price_info = data['priceInfo']
        return {
            "symbol" : symbol.upper(),
            "last price" : price_info["lastPrice"],   # Current price
            "change" : price_info["change"],  # absolute change from previous close
            "pChange" : price_info["pChange"],   # % change
            "timestamp" : data['metadata']['lastUpdateTime']  # time of last update
        }
    except Exception as e:
        return {"error" : str(e)}

# Calling the function every 1 min to stream live data to the model

def stream_stock_data(symbol : str = "INFY", interval : float = 60):
    with open(file = f"{symbol}_stock_data.csv", mode = 'a', newline = '') as f:
        writer = csv.DictWriter(f, fieldnames = ["timestamp", "symbol", "last_price", "change", "pChange", "last price"])
        if f.tell() == 0:
            writer.writeheader()
        while True:
            data = get_nse_stock_price(symbol = symbol)
            if "error" not in data:   # Checking if the return value from the previous function is not an error.
                print(data)
                writer.writerow(data)
            else:
                print("Error Fetching", data["error"])
                time.sleep(interval)

# Collecting data of reliance using yfinance for training the model
if __name__ == '__main__':
    import yfinance as yf
    import pandas as pd

    # Fetching 5-min candle data for reliance.Ns
    reliance_5min = yf.download(
        'RELIANCE.NS',
        period = '60d',
        interval = '5m'
    )

    # Saving it to csv
    reliance_5min.to_csv('Data/reliance_5min_60days.csv')



