import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler





pd.set_option('display.max_columns', 100)





def deal_historical_transactions(data):
    mode_cols = ['merchant_id', 'category_2', 'category_3']
    for c in mode_cols:
        data[c].fillna(data[c].mode()[0], inplace=True)


def deal_new_merchant_transactions(data):
    mode_cols = ['merchant_id', 'category_2', 'category_3']
    for c in mode_cols:
        data[c].fillna(data[c].mode()[0], inplace=True)

    one_zero_cols = ['authorized_flag', 'category_1']



if __name__ == '__main__':
    nrows = 100000
    data = pd.read_csv('Elo/input/train.csv', nrows=nrows)
    # print(data['feature_1'].value_counts())
    # print(data.head())

    data['outliers'] = 0
    data.loc[data['target'] < -30, 'outliers'] = 1
    for c in ['feature_1', 'feature_2', 'feature_3']:
        order_label = data.groupby([c])['outliers'].mean()
        print(order_label)
        data[c] = data[c].map(order_label)
    print(data['feature_1'])

    # test = pd.read_csv('Elo/input/test.csv', nrows=nrows)
    # # b =test.head()
    #
    # merchants = pd.read_csv('Elo/input/merchants.csv', nrows=nrows)
    # # a = merchants.isnull().sum()
    # # print(a)
    # print(merchants.head())
    # #
    # new_merchant_transactions = pd.read_csv('Elo/input/new_merchant_transactions.csv', nrows=nrows)
    # deal_new_merchant_transactions(new_merchant_transactions)
    # print(new_merchant_transactions.head())
    #
    #
    #
    # # print(new_merchant_transactions.describe())
    #
    #
    # historical_transactions = pd.read_csv('Elo/input/historical_transactions.csv', nrows=nrows)
    # deal_historical_transactions(historical_transactions)
    # print(historical_transactions.head())