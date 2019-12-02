# -*- coding: utf-8 -*-
"""
Project: cold_chain_02
Create time: 2019-10-21
Introduction:
"""
import pandas as pd
import os


def read_data(file_name):
    df = None
    file_folder = os.path.dirname(__file__)
    file_path = os.path.join(os.path.dirname(file_folder), 'data', file_name)
    if not os.path.exists(file_path):
        raise FileNotFoundError
    else:
        if os.path.splitext(file_path)[-1] in ['.csv', '.xls', '.xlsx']:
            if file_path.endswith('.csv'):
                try:
                    df = pd.read_csv(file_path, skiprows=3, encoding='gb2312')
                except BaseException:
                    df = pd.read_csv(file_path, skiprows=3, encoding='utf-8')
            elif file_path.endswith('.xls') or file_path.endswith('.xlsx'):
                try:
                    df = pd.read_excel(
                        file_path, skiprows=3, encoding='gb2312')
                except BaseException:
                    df = pd.read_excel(file_path, skiprows=3, encoding='utf-8')
            variable, _ = os.path.splitext(file_name)
            variable = variable.split('-1')[-1].strip()
            df.columns = [df.columns[0]] + \
                list(variable + '_' + df.columns.values[1:])
            df['Timestamp'] = df['Timestamp'].str.replace(
                '十月', '10', regex=False)
            df['Timestamp'] = df['Timestamp'].str.replace(
                r'[A-Z]+ [A-Z]+', '', regex=True)
            df['Timestamp'] = pd.to_datetime(df['Timestamp'])
            df.iloc[:, -1] = df.iloc[:, -1].str.replace(r' [^\d]*$', '', regex=True)
        else:
            print('Cannot read file: {}'.format(file_name))

    return df


def combine_data(file_folder=None):
    if file_folder is not None and os.path.exists(file_folder):
        file_folder = file_folder
    else:
        file_folder = os.path.join(
            os.path.dirname(
                os.path.dirname(__file__)),
            'data')
    file_list = os.listdir(file_folder)
    df_total = pd.DataFrame()
    for file_path in file_list:
        df_tmp = read_data(file_path)
        if df_total is not None and df_tmp is not None:
            df_total = pd.merge(df_total, df_tmp, how='outer', on='Timestamp')
        else:
            df_total = df_tmp

    df_total['Timestamp'] = pd.to_datetime(df_total['Timestamp'])
    df_total.sort_values(by='Timestamp', inplace=True)
    df_total = df_total[['Timestamp'] + sorted(df_total.columns[1:])]
    return df_total


def filter_data(df, category):
    if not set([category]).issubset(set(['temperature', 'pressure'])):
        raise TypeError('category: {} not in temperature or pressure'.format(category))
    elif category == 'temperature':
        key_words = ['TEMP_Value']
    elif category == 'pressure':
        key_words = ['PSIG_Value']
    else:
        key_words = ['TEMP_Value', 'PSIG_Value']

    var_list = [variable for variable in df.columns.values for key_word in key_words if key_word in variable]
    df_var = df[['Timestamp'] + var_list]
    df_var = df_var.dropna(axis=0)
    return df_var


if __name__ == '__main__':
    # df = read_data('LYA-1 DISCHRG_PSIG.csv')
    # print(df.head())
    # df_total = combine_data()
    # df_total.to_csv('raw_data.csv', encoding='gb2312', index=False)
    # print(df_total.head())
    df = pd.read_csv(r'./raw_data.csv', encoding='gb2312')
    df_temp_pres = filter_data(df, category=['temperature', 'pressure'])
    df_temp_pres.to_csv(r'temperature_pressure_data.csv', encoding='gb2312', index=False)