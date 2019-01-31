import time
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.svm import SVR
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RidgeCV, LassoCV, LinearRegression
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', 200)



def scale():
    # for c in ['pass_time', 'hist_first_buy', 'new_hist_first_buy', 'card_id_total', 'purchase_amount_total']:
    #     min, max = data[c].min(), data[c].max()
    #     data[c] = data[c].apply(lambda x: (x - min) / (max - min))

    # for c in ['hist_category_2_sum', 'hist_category_3_sum', 'hist_installments_sum', 'hist_month_lag_sum',
    #           'hist_purchase_amount_max', 'hist_purchase_amount_mean', 'hist_authorized_flag_sum']:
    #     mean, std = hist_data[c].mean(), hist_data[c].std()
    #     hist_data[c] = hist_data[c].apply(lambda x: (x - mean) / std)
    return


def deal_transactions(str, data):
    print('start deal with {}...'.format(str))
    mode_cols = ['merchant_id', 'category_2', 'category_3']
    for c in mode_cols:
        data[c].fillna(data[c].mode()[0], inplace=True)

    one_zero_cols = ['authorized_flag', 'category_1']
    for c in one_zero_cols:
        data[c] = data[c].map({'Y': 1, 'N': 0})

    # data['category_3'] = data['category_3'].map({'A': 1, 'B': 2, 'C': 3})
    category_3 = pd.get_dummies(data['category_3'], prefix='category_3')
    data = pd.concat([data.drop('category_3', axis=1), category_3], axis=1)

    data['purchase_date'] = pd.to_datetime(data['purchase_date'])
    data['price'] = data['purchase_amount'] / data['installments']
    data['year'] = (data['purchase_date'].dt.year)
    data['weekofyear'] = data['purchase_date'].dt.weekofyear
    data['month'] = data['purchase_date'].dt.month
    data['dayofweek'] = data['purchase_date'].dt.dayofweek
    data['IsWeekend'] = (data['purchase_date'].dt.dayofweek >= 5).astype(int)
    data['hour'] = data['purchase_date'].dt.hour
    data['month_diff'] = ((datetime.today() - data['purchase_date']).dt.days) // 30
    data['month_diff'] += data['month_lag']
    data['duration'] = data['purchase_amount'] * data['month_diff']
    data['amount_month_ratio'] = data['purchase_amount'] / data['month_diff']

    # Christmas : December 25 2017
    data['Christmas_Day_2017'] = (pd.to_datetime('2017-12-25') - data['purchase_date']).dt.days.apply(
        lambda x: x if x > 0 and x < 100 else 0)
    # Mothers Day: May 14 2017
    data['Mothers_Day_2017'] = (pd.to_datetime('2017-06-04') - data['purchase_date']).dt.days.apply(
        lambda x: x if x > 0 and x < 100 else 0)
    # fathers day: August 13 2017
    data['fathers_day_2017'] = (pd.to_datetime('2017-08-13') - data['purchase_date']).dt.days.apply(
        lambda x: x if x > 0 and x < 100 else 0)
    # Childrens day: October 12 2017
    data['Children_day_2017'] = (pd.to_datetime('2017-10-12') - data['purchase_date']).dt.days.apply(
        lambda x: x if x > 0 and x < 100 else 0)
    # Valentine's Day : 12th June, 2017
    data['Valentine_Day_2017'] = (pd.to_datetime('2017-06-12') - data['purchase_date']).dt.days.apply(
        lambda x: x if x > 0 and x < 100 else 0)
    # Black Friday : 24th November 2017
    data['Black_Friday_2017'] = (pd.to_datetime('2017-11-24') - data['purchase_date']).dt.days.apply(
        lambda x: x if x > 0 and x < 100 else 0)

    # 2018
    # Mothers Day: May 13 2018
    data['Mothers_Day_2018'] = (pd.to_datetime('2018-05-13') - data['purchase_date']).dt.days.apply(
        lambda x: x if x > 0 and x < 100 else 0)

    aggs = {}
    aggs['card_id'] = ['size']
    aggs['purchase_date'] = ['max', 'min']
    for c in ['subsector_id', 'merchant_id', 'merchant_category_id']:
        aggs[c] = ['nunique']
    for c in ['category_1', 'category_2', 'IsWeekend', 'authorized_flag']:
        aggs[c] = ['mean']
    for c in ['Christmas_Day_2017', 'Mothers_Day_2017', 'fathers_day_2017', 'Children_day_2017', 'Valentine_Day_2017',
              'Black_Friday_2017', 'Mothers_Day_2018']:
        aggs[c] = ['mean']
    for c in ['year', 'month', 'weekofyear', 'dayofweek', 'hour']:
        aggs[c] = ['nunique', 'mean', 'max', 'min']
    for c in ['installments', 'purchase_amount', 'month_lag', 'month_diff', 'price', 'duration', 'amount_month_ratio']:
        aggs[c] = ['sum', 'max', 'min', 'mean', 'var', 'skew']

    hist_data = data.groupby('card_id').agg(aggs)
    hist_data.columns = ['hist_' + k + '_' + v for k in aggs.keys() for v in aggs[k]]
    hist_data.reset_index(inplace=True)
    hist_data['hist_purchase_date_diff'] = (hist_data['hist_purchase_date_max'] -
                                            hist_data['hist_purchase_date_min']).dt.days
    hist_data['hist_purchase_date_average'] = hist_data['hist_purchase_date_diff'] / \
                                              hist_data['hist_card_id_size']
    hist_data['hist_purchase_date_utn'] = (datetime.today() - hist_data['hist_purchase_date_max']).dt.days
    hist_data['hist_purchase_date_uto'] = (datetime.today() - hist_data['hist_purchase_date_min']).dt.days

    if str == 'new':
        hist_data.columns = ['new_' + c for c in hist_data.columns]
        hist_data.rename(columns={'new_card_id': 'card_id'}, inplace=True)

    return hist_data


def deal_data(data):
    print('start deal with x_data')
    data['first_active_month'] = pd.to_datetime(data['first_active_month'])
    data['year'] = (data['first_active_month'].dt.year)
    data['month'] = data['first_active_month'].dt.month
    data['weekofyear'] = data['first_active_month'].dt.weekofyear
    data['dayofweek'] = data['first_active_month'].dt.dayofweek

    data['pass_time'] = (datetime.today() - data['first_active_month']).dt.days
    data['augmentation_feature1'] = data['pass_time'] * data['feature_1']
    data['augmentation_feature2'] = data['pass_time'] * data['feature_2']
    data['augmentation_feature3'] = data['pass_time'] * data['feature_3']
    data['feature1_ratio'] = data['feature_1'] / data['pass_time']
    data['feature2_ratio'] = data['feature_2'] / data['pass_time']
    data['feature3_ratio'] = data['feature_3'] / data['pass_time']
    data['feature_sum'] = data['feature_1'] + data['feature_2'] + data['feature_3']
    data['feature_mean'] = data['feature_sum'] / 3
    data['feature_max'] = data[['feature_1', 'feature_2', 'feature_3']].max(axis=1)
    data['feature_min'] = data[['feature_1', 'feature_2', 'feature_3']].min(axis=1)
    data['feature_var'] = data[['feature_1', 'feature_2', 'feature_3']].std(axis=1)

    data['hist_first_buy'] = (data['hist_purchase_date_min'] - data['first_active_month']).dt.days
    data['hist_last_buy'] = (data['hist_purchase_date_max'] - data['first_active_month']).dt.days
    data['new_hist_first_buy'] = (data['new_hist_purchase_date_min'] - data['first_active_month']).dt.days
    data['new_hist_last_buy'] = (data['new_hist_purchase_date_max'] - data['first_active_month']).dt.days

    for c in ['hist_purchase_date_max', 'hist_purchase_date_min',
              'new_hist_purchase_date_max', 'new_hist_purchase_date_min']:
        data[c] = data[c].astype(np.int64) * 1e-9

    data['card_id_total'] = data['new_hist_card_id_size'] + data['hist_card_id_size']
    data['purchase_amount_total'] = data['new_hist_purchase_amount_sum'] + data['hist_purchase_amount_sum']
    data['purchase_amount_mean'] = data['new_hist_purchase_amount_mean'] + data['hist_purchase_amount_mean']
    data['purchase_amount_max'] = data['new_hist_purchase_amount_max'] + data['hist_purchase_amount_max']
    data['purchase_amount_min'] = data['new_hist_purchase_amount_min'] + data['hist_purchase_amount_min']
    data['purchase_amount_ratio'] = data['new_hist_purchase_amount_sum'] / data['hist_purchase_amount_sum']
    data['month_diff_mean'] = data['new_hist_month_diff_mean'] + data['hist_month_diff_mean']
    data['month_diff_ratio'] = data['new_hist_month_diff_mean'] / data['hist_month_diff_mean']
    data['month_lag_mean'] = data['new_hist_month_lag_mean'] + data['hist_month_lag_mean']
    data['month_lag_max'] = data['new_hist_month_lag_max'] + data['hist_month_lag_max']
    data['month_lag_min'] = data['new_hist_month_lag_min'] + data['hist_month_lag_min']
    data['category_1_mean'] = data['new_hist_category_1_mean'] + data['hist_category_1_mean']
    data['installments_total'] = data['new_hist_installments_sum'] + data['hist_installments_sum']
    data['installments_mean'] = data['new_hist_installments_mean'] + data['hist_installments_mean']
    data['installments_max'] = data['new_hist_installments_max'] + data['hist_installments_max']
    data['installments_ratio'] = data['new_hist_installments_sum'] / data['hist_installments_sum']

    data['price_total'] = data['purchase_amount_total'] / data['installments_total']
    data['price_mean'] = data['purchase_amount_mean'] / data['installments_mean']
    data['price_max'] = data['purchase_amount_max'] / data['installments_max']
    data['duration_mean'] = data['new_hist_duration_mean'] + data['hist_duration_mean']
    data['duration_min'] = data['new_hist_duration_min'] + data['hist_duration_min']
    data['duration_max'] = data['new_hist_duration_max'] + data['hist_duration_max']
    data['amount_month_ratio_mean'] = data['new_hist_amount_month_ratio_mean'] + data['hist_amount_month_ratio_mean']
    data['amount_month_ratio_min'] = data['new_hist_amount_month_ratio_min'] + data['hist_amount_month_ratio_min']
    data['amount_month_ratio_max'] = data['new_hist_amount_month_ratio_max'] + data['hist_amount_month_ratio_max']

    # for f in ['feature_1', 'feature_2', 'feature_3']:
    #     label = data.groupby([f])['outliers'].mean()
    #     data[f] = data[f].map(label)
    for c in ['feature_1', 'feature_2', 'feature_3']:
        one_hot_data = pd.get_dummies(data[c], prefix=c)
        data = pd.concat([data.drop(c, axis=1), one_hot_data], axis=1)

    return data


def get_oof(model, x, y, test, classifier):
    n_splits = 12
    oob_hat = np.zeros(x.shape[0])
    skfolds = StratifiedKFold(n_splits=n_splits, shuffle=False, random_state=2019)
    # skfolds = KFold(n_splits=n_splits, shuffle=False, random_state=1)
    test_preds = np.zeros(test.shape[0])

    for k, (trn_idx, val_idx) in enumerate(skfolds.split(x, classifier)):
        if k in [0, 4, 9]:
            print('fold${}'.format(k+1))
        x_train, y_train = x.iloc[trn_idx], y.iloc[trn_idx]
        x_valid, y_valid = x.iloc[val_idx], y.iloc[val_idx]

        model.fit(x_train, y_train)

        oob_hat[val_idx] = model.predict(x_valid)

        test_preds += model.predict(test)
    print('r_mean_squared_error: {}'.format(np.sqrt(mean_squared_error(y, oob_hat))))

    test_preds /= n_splits

    return oob_hat, test_preds


def Dnn(x, y, t):
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    exportCUDA_VISIBLE_DEVICES = 1
    from keras.models import Model
    from keras.layers import Dense, BatchNormalization, Activation, Input, add
    from keras.optimizers import Adam, SGD
    from keras.callbacks import LearningRateScheduler

    x1, x2, y1, y2 = train_test_split(x, y, test_size=0.2, random_state=2019)
    scaler = StandardScaler()
    x1, x2 = scaler.fit_transform(x1), scaler.fit_transform(x2)
    inpt = Input(shape=(x.shape[1],))
    output = Dense(units=512, activation='selu', kernel_initializer='lecun_normal')(inpt)
    output = Dense(units=1024, activation='selu', kernel_initializer='lecun_normal')(output)
    output = Dense(units=1, activation='selu', kernel_initializer='lecun_normal')(output)

    d_model = Model(inputs=inpt, outputs=output)

    d_model.compile(optimizer=Adam(epsilon=1e-08), loss='mse')  #decay=lr/epoch
    d_model.summary()
    update_lr = LearningRateScheduler(lambda x: 1e-3 * 0.95 ** x)
    d_model.fit(x1, y1, batch_size=128, epochs=40, verbose=2, validation_data=(x2, y2), callbacks=[update_lr])

    Dnn_train = d_model.predict(x)
    Dnn_test = d_model.predict(t)
    return Dnn_train, Dnn_test


if __name__ == '__main__':
    train = pd.read_csv('inputs/train.csv')
    test = pd.read_csv('inputs/test.csv')

    new_merchant_transactions = pd.read_csv('inputs/new_merchant_transactions.csv')
    new_hist_transactions = deal_transactions('new', new_merchant_transactions)
    train = train.merge(new_hist_transactions, on='card_id', how='left')
    test = test.merge(new_hist_transactions, on='card_id', how='left')

    historical_transactions = pd.read_csv('inputs/historical_transactions.csv')
    hist_transactions = deal_transactions('hist', historical_transactions)
    train = train.merge(hist_transactions, on='card_id', how='left')
    test = test.merge(hist_transactions, on='card_id', how='left')

    train['outliers'] = 0
    train.loc[train['target'] < -30, 'outliers'] = 1

    train = deal_data(train)
    test = deal_data(test)

    train.fillna(0, inplace=True)
    test.fillna(0, inplace=True)

    test_id = test['card_id']
    test = test.drop(['card_id', 'first_active_month'], axis=1)
    Y = train['target']
    X = train.drop(['card_id', 'first_active_month', 'target', 'outliers'], axis=1)

    # Dnn_train, Dnn_test = Dnn(X, Y, test)

    models = [
    #     # Pipeline([
    #     #     ('poly', PolynomialFeatures()),
    #     #     ('clf', LinearRegression(n_jobs=-1))]),
    #     # Pipeline([
    #     #     ('clf', RandomForestRegressor(n_estimators=100, criterion='mse', max_depth=10, random_state=0, n_jobs=5, verbose=0))]),
    #     # Pipeline([
    #     #     ('clf', RidgeCV(alphas=[0.01, 0.1, 0.3, 0.5, 1, 5, 10]))]),
    #     # Pipeline([
    #         # ('clf', LassoCV(n_jobs=-1))]),
        Pipeline([
            ('clf', XGBRegressor(max_depth=4, learning_rate=0.02, n_estimators=750, objective='reg:linear',  #alpha数据维度高时使用，可以使算法运行加快
                                 booster='gbtree', subsample=0.8, colsample_bytree=0.8, reg_lambda=1,         #lambda一般不使用，但是可以降低过拟合
                                 gamma=0.2, random_state=2019, n_jobs=-1))]),
        Pipeline([
            ('clf', LGBMRegressor(boosting_type='gbdt', n_estimators=4000, num_leaves=63, max_depth=7,
                                  min_child_weight=41.9612869171337, min_split_gain=9.820197773625843,
                                  learning_rate=0.005, objective='regression', subsample=0.9855232997390695,
                                  subsample_freq=1, colsample_bytree=0.5665320670155495,
                                  reg_alpha=9.677537745007898, reg_lambda=8.2532317400459,
                                  metric='rmse', random_state=2019, n_jobs=-1))])
    ]

    sec_train = np.zeros((X.shape[0], len(models)))
    sec_test = np.zeros((test.shape[0], len(models)))
    # titles = ['LR', 'RF', 'Lasso', 'xgb', 'Lgbm']
    titles = ['xgb', 'Lgbm']

    for i, model in enumerate(models):
        print(titles[i], end=':  ')
        model_oof_train, model_oof_test = get_oof(model, X, Y, test, classifier=train['outliers'])
        sec_train[:, i] = model_oof_train
        sec_test[:, i] = model_oof_test
    # sec_train[:, len(models)] = [i for i in Dnn_train]
    # sec_test[:, len(models)] = [i for i in Dnn_test]

    # poly = PolynomialFeatures(degree=3)
    # sec_train = poly.fit_transform(sec_train)
    # sec_test = poly.fit_transform(sec_test)
    final_model = LinearRegression()
    final_model.fit(sec_train, Y)
    predictions = final_model.predict(sec_test)
    print(np.sqrt(mean_squared_error(Y, final_model.predict(sec_train))))

    if np.sqrt(mean_squared_error(Y, final_model.predict(sec_train))) < 3.66:
        filename = 'submission_{}_{}.csv'.format('stacking', datetime.now().strftime('%Y-%m-%d-%H-%M'))
        submission = pd.DataFrame({'card_id': test_id,
                                   'target': predictions})
        submission.to_csv('submissions/{}'.format(filename), index=False)






























































    # print(X.describe())



    # n_splits = 12
    # skfolds = StratifiedKFold(n_splits=n_splits, shuffle=False, random_state=11)
    # oof = np.zeros(len(X))
    # predictions = np.zeros(len(test))
    # feature_importance = pd.DataFrame()
    # for fold_, (trn_idx, val_idx) in enumerate(skfolds.split(X, train['outliers'])):
    #     print('fold%{}'.format(fold_))
    #     x_train, y_train = X.iloc[trn_idx], Y.iloc[trn_idx]
    #     x_valid, y_valid = X.iloc[val_idx], Y.iloc[val_idx]
    #
    #     model = LGBMRegressor(boosting_type='gbdt',
    #                           n_estimators=10000,
    #                           num_leaves=63,
    #                           max_depth=7,
    #                           min_child_weight=41.9612869171337,
    #                           min_split_gain= 9.820197773625843,
    #                           learning_rate=0.005,
    #                           objective='regression',
    #                           subsample=0.9855232997390695,
    #                           subsample_freq=1,
    #                           colsample_bytree=0.5665320670155495,
    #                           reg_alpha=9.677537745007898,
    #                           reg_lambda=8.2532317400459,
    #                           metric='rmse',
    #                           random_state=200*(fold_+1))
    #     model.fit(x_train, y_train, early_stopping_rounds=300, eval_set=[(x_train, y_train), (x_valid, y_valid)],
    #               eval_metric='rmse', verbose=200)
    #     oof[val_idx] = model.predict(x_valid)
    #
    #     fold_importance = pd.DataFrame({'feature': [c for c in X.columns],
    #                                     'importance': model.feature_importances_})
    #     feature_importance = pd.concat([feature_importance, fold_importance], axis=0)
    #     predictions += model.predict(test)
    # final_feature_importance = feature_importance.groupby('feature').mean()\
    #                            .sort_values(by='importance', ascending=False)
    # print(final_feature_importance)
    # print(np.sqrt(mean_squared_error(Y, oof)))
    #
    # if np.sqrt(mean_squared_error(Y, oof)) < 3.66:
    #     filename = 'submission_{}_{}.csv'.format('LGBM', datetime.now().strftime('%Y-%m-%d-%H-%M'))
    #     submission = pd.DataFrame({'card_id': test_id,
    #                                'target': predictions / n_splits})
    #     submission.to_csv('submissions/{}'.format(filename), index=False)
