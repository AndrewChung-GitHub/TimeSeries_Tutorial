# -*- coding: utf-8 -*- 
"""
Project: cold_chain_02
Create time: 2019-10-31
Introduction:
"""
import statsmodels.api as sm
import pandas as pd
from cold_chain.preprocessing.data_handle import filter_data
import joblib
import os


class TemperaturePredict(object):
    def __init__(self, **sarimax_kwargs):
        sarimax_kwargs['order'] = sarimax_kwargs.get('order', (0, 0, 0))
        sarimax_kwargs['seasonal_order'] = sarimax_kwargs.get('seasonal_order', (0, 0, 0, 0))
        sarimax_kwargs['trend'] = sarimax_kwargs.get('trend', None)

        self.sarimax_kwargs = sarimax_kwargs
        self.ts = None
        self.model = None
        self.predict = {}

    def fit(self, df):
        self.ts = df.set_index('Timestamp')
        self.model = sm.tsa.statespace.SARIMAX(self.ts, ** self.sarimax_kwargs).fit()
        return self

    def forecast(self, number):
        self.predict['yhat'] = self.model.forecast(number)
        self.predict['yhat_lower'] = self.model.get_forecast(number).conf_int().iloc[:, 0]
        self.predict['yhat_upper'] = self.model.get_forecast(number).conf_int().iloc[:, 1]
        return self

    def save(self):
        joblib.dump(
            self.model,
            filename=os.path.join(
                os.path.dirname(
                    os.path.abspath(__file__)),
                'pkl_files',
                'temperature_predict.pkl'))

    def load(self):
        self.model = joblib.load(
            filename=os.path.join(
                os.path.dirname(
                    os.path.abspath(__file__)),
                'pkl_files',
                'temperature_predict.pkl'))


if __name__ == '__main__':
    FORECAST_LENGTH = 200
    df_total = pd.read_csv(r'../../preprocessing/raw_data.csv', encoding='gb2312')
    df = filter_data(df=df_total, category='temperature')
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df.set_index('Timestamp', inplace=True)
    df_new = df['2019-10-10': '2019-10-15'][['B TEMP_Value (°F)']]
    df_new.reset_index(inplace=True)

    # Training model
    # tp = TemperaturePredict(order=(0, 0, 0), seasonal_order=(4, 1, 6, 36))
    # tp.fit(df_new)
    # tp.save()

    # Loading trained model
    tp = TemperaturePredict()
    tp.load()
    tp.forecast(FORECAST_LENGTH)

    df_forecast = pd.DataFrame({
        'Timestamp': pd.date_range(start='2019-10-16 00:00:00', freq='10T', periods=FORECAST_LENGTH),
        'Predict': tp.predict['yhat'],
        'Lower': tp.predict['yhat_lower'],
        'Upper': tp.predict['yhat_upper']
    })

    df_all = pd.concat([df_new, df_forecast], sort=False)
    df_all.to_csv(r'../../preprocessing/data_arima.csv', index=False)

