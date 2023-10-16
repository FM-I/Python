import datetime
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn import covariance, cluster
import mplfinance as mpf
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

input_file = 'company_symbol_mapping.json'

with open(input_file, 'r') as f:
    company_symbols_map = json.loads(f.read())

symbols, names = np.array(list(company_symbols_map.items())).T

start_date = datetime.datetime (2003, 7, 3)
end_date = datetime.datetime (2007, 5, 4)
quotes = [yf.download(symbol, start=start_date, end=end_date) for symbol in symbols]


dtype = [('symbol', '<U5'), ('diff', float)]  # типи даних для структурованого масиву
quotes_diff = np.array([(symbol, quote['Close'].mean() - quote['Open'].mean()) for symbol, quote in zip(symbols, quotes)], dtype=dtype)

normalized_diff_values = quotes_diff['diff'].reshape(-1, 1)

# Виконайте операції з цими нормалізованими значеннями
# Наприклад, якщо ви хочете обчислити суму нормалізованих значень:
sum_normalized_diff = np.sum(normalized_diff_values)

X = normalized_diff_values.copy().T
X /= np.std(X)


edge_model = covariance.GraphicalLassoCV()

with np.errstate(invalid='ignore'):
    edge_model.fit(X)
