# -*- coding: utf-8 -*- 
"""
Project: cold_chain_02
Create time: 2019-10-21
Introduction:
"""
import pandas as pd
import os
from plotnine import ggplot, geom_point, geom_line, scale_x_datetime,theme_bw, theme,\
                     element_text, aes, geom_ribbon, xlab, ylab, geom_vline


def plot_chart(df, category='temperature', time_interval=None):
    if category not in ['temperature', 'pressure']:
        raise TypeError('category: {} not in temperature or pressure'.format(category))
    elif category == 'temperature':
        key_words = 'TEMP_Value'
    else:
        key_words = 'PSIG_Value'

    var_list = [variable for variable in df.columns.values if key_words in variable]
    df_var = df[['Timestamp'] + var_list]
    df_var = df_var.dropna(axis=0)
    df_var = df_var.melt(id_vars=['Timestamp'], var_name='ITEM', value_name='Value')
    df_var['Timestamp'] = pd.to_datetime(df_var['Timestamp'])

    if time_interval is None:
        time_interval = [min(df_var['Timestamp']), max(df_var['Timestamp'])]

    p = (
        ggplot(data=df_var, mapping=aes(x='Timestamp', y='Value')) +
        geom_point(alpha=0.2, mapping=aes(colour='factor(ITEM)'), na_rm=True) +
        geom_line(mapping=aes(colour='factor(ITEM)'), na_rm=True) +
        scale_x_datetime(limits=pd.to_datetime(time_interval), breaks='1 days', date_labels='%y-%m-%d %H:%M') +
        theme_bw() +
        theme(axis_text_x=element_text(angle=45, hjust=0.5, face='bold', color='black'),
              axis_text_y=element_text(face='bold', colour='black'),
              legend_title=element_text(face='bold', colour='black'),
              legend_position='right',
              legend_direction="vertical")
    )
    ggplot.save(
        p,
        filename=category + '_chart' + '.png',
        path=os.path.join(
            os.path.abspath(
                os.path.dirname(__file__)),
            'png'),
        width=8, height=6, units='in', dpi=326, verbose=False)
    return p


def plot_predict(forecast):
    p = (
        ggplot(data=forecast, mapping=aes(x='ds', y='y')) +
        geom_point(colour='blue', alpha=0.3, na_rm=True) +
        geom_line(colour='blue', na_rm=True) +
        geom_line(data=forecast, mapping=aes(x='ds', y='yhat'), colour='red') +
        geom_ribbon(data=forecast, mapping=aes(ymin='yhat_lower', ymax='yhat_upper'),
                    fill='blue', alpha=0.1) +
        scale_x_datetime(breaks='1 days', date_labels='%y-%m-%d %H:%M') +
        xlab('Time') +
        ylab('Pressure') +
        theme_bw() +
        theme(axis_text_x=element_text(angle=45, hjust=1, face='bold', color='black'),
              axis_text_y=element_text(face='bold', colour='black'))
    )

    ggplot.save(
        p,
        filename='predict_pressure_chart.png',
        path=os.path.join(
            os.path.abspath(
                os.path.dirname(__file__)),
            'png'),
        width=8, height=6, units='in', dpi=326, verbose=False)
    return p


def plot_arima(df):
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    p = (
            ggplot(data=df, mapping=aes(x='Timestamp', y=df.columns.values[1])) +
            geom_point(colour='blue', alpha=0.3, na_rm=True) +
            geom_line(colour='blue', na_rm=True) +
            geom_point(mapping=aes(x='Timestamp', y=df.columns.values[2]), colour='red', alpha=0.3, na_rm=True) +
            geom_line(mapping=aes(x='Timestamp', y=df.columns.values[2]), colour='red', na_rm=True) +
            geom_vline(xintercept=max(df[['Timestamp', df.columns.values[1]]].dropna(axis=0)['Timestamp']),
                       color='green', linetype='dashed') +
            # geom_line(mapping=aes(x='Timestamp', y='Lower'), colour='green', na_rm=True, alpha=0.3) +
            # geom_line(mapping=aes(x='Timestamp', y='Upper'), colour='green', na_rm=True, alpha=0.3) +
            geom_ribbon(data=df, mapping=aes(ymin='Lower', ymax='Upper'),
                        fill='red', alpha=0.1) +
            scale_x_datetime(breaks='1 days', date_labels='%y-%m-%d %H:%M') +
            xlab('Time') +
            ylab(df.columns.values[1]) +
            theme_bw() +
            theme(axis_text_x=element_text(angle=45, hjust=1, face='bold', color='black'),
                  axis_text_y=element_text(face='bold', colour='black'))
    )

    ggplot.save(
        p,
        filename=df.columns.values[1] + '_predict.png',
        path=os.path.join(
            os.path.abspath(
                os.path.dirname(__file__)),
            'png'),
        width=8, height=6, units='in', dpi=326, verbose=False)
    return p


if __name__ == '__main__':
    df = pd.read_csv(r'../../preprocessing/raw_data.csv', encoding='gb2312')
    df_temperature = pd.read_csv(r'../../preprocessing/data_arima.csv')
    plot_chart(df, category='temperature', time_interval=['2019-10-10', '2019-10-15'])
    plot_chart(df, category='pressure', time_interval=['2019-10-10', '2019-10-15'])
    plot_arima(df_temperature)
