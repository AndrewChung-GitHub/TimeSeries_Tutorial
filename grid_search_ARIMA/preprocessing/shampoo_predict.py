# -*- coding: utf-8 -*- 
"""
Project: ts_arima
Create time: 2019-12-02
Introduction:
"""
import pandas as pd
from datetime import datetime
from grid_search_ARIMA.funcs.evaluate_arima import evaluate_models
import warnings
warnings.filterwarnings('ignore')


# load dataset
def parser(x):
    return datetime.strptime('190' + x, '%Y-%m')


series = pd.read_csv(r'../data/shampoo.csv', header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)

# evaluate parameters
p_values = [0, 1, 2, 4, 6, 8, 10]
d_values = range(0, 3)
q_values = range(0, 3)
evaluate_models(series, p_values, d_values, q_values)
