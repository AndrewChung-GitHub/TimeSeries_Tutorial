# -*- coding: utf-8 -*- 
"""
Project: cold_chain_02
Create time: 2019-10-23
Introduction:
"""
from fbprophet import Prophet
import joblib
import os


class WorkingPredict(object):
    def __init__(self, **prophet_kwargs):
        prophet_kwargs['growth'] = prophet_kwargs.get('growth', 'linear')
        prophet_kwargs['yearly_seasonality'] = prophet_kwargs.get(
            'yearly_seasonality', False)
        prophet_kwargs['weekly_seasonality'] = prophet_kwargs.get(
            'weekly_seasonality', False)
        prophet_kwargs['daily_seasonality'] = prophet_kwargs.get(
            'daily_seasonality', True)
        prophet_kwargs['changepoint_prior_scale'] = prophet_kwargs.get(
            'changepoint_prior_scale', 0.05)

        self.prophet_kwargs = prophet_kwargs
        self.ts = None
        self.model = None
        self.future = None
        self.forecast = None

    def fit(self, df):
        df.columns = ['ds', 'y']
        self.model = Prophet(** self.prophet_kwargs)
        self.ts = df[['ds', 'y']]
        self.model = self.model.fit(self.ts)
        return self

    def predict(self, periods=24, freq='H'):
        self.future = self.model.make_future_dataframe(
            periods=periods, freq=freq)
        self.forecast = self.model.predict(self.future)
        self.forecast['y'] = self.ts['y']
        return self

    def save(self):
        joblib.dump(
            self.model,
            filename=os.path.join(
                os.path.dirname(
                    os.path.abspath(__file__)),
                'pkl_files',
                'working_predict.pkl'))

    def load(self):
        self.model = joblib.load(
            filename=os.path.join(
                os.path.dirname(
                    os.path.abspath(__file__)),
                'pkl_files',
                'working_predict.pkl'))


if __name__ == '__main__':
    from cold_chain.preprocessing.data_handle import filter_data
    from cold_chain.status_monitor.charts.charts_plot import plot_predict
    import pandas as pd

    df_total = pd.read_csv(r'../../preprocessing/raw_data.csv', encoding='gb2312')
    df = filter_data(df=df_total, category='pressure')
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df.set_index('Timestamp', inplace=True)
    df_new = df['2019-10-10': '2019-10-15'][['DISCHRG_PSIG_Value (psi)']]
    df_new.reset_index(inplace=True)
    wp = WorkingPredict()
    wp.fit(df_new)
    wp.predict()
    forecast = wp.forecast[['ds', 'y', 'yhat', 'yhat_upper', 'yhat_lower', 'trend',
                           'trend_upper', 'trend_lower', 'daily', 'daily_upper', 'daily_lower']]
    p = plot_predict(forecast)



