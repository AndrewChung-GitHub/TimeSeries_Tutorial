# -*- coding: utf-8 -*- 
"""
Project: ts_arima
Create time: 2019-12-02
Introduction:
"""
import pandas as pd
from grid_search_ARIMA.funcs.evaluate_arima import evaluate_models
import warnings


# load dataset
series = pd.read_csv(r'../data/daily-total-female-births.csv', header=0)
series['Date'] = pd.to_datetime(series['Date'])
series.set_index('Date', inplace=True)

# evaluate parameters
p_values = [0, 1, 2, 4, 6, 8, 10]
d_values = range(0, 3)
q_values = range(0, 3)
warnings.filterwarnings("ignore")
evaluate_models(series.values, p_values, d_values, q_values)
