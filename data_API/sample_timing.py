# -*-coding:utf-8-*-
# @Time:   2021/7/14 9:43
# @Author: FC
# @Email:  18817289038@163.com

import random
import numpy as np
import pandas as pd
import bottleneck as bk
import matplotlib.pyplot as plt
from scipy.stats import dirichlet

from typing import Tuple, Any
from dataclasses import dataclass

from utility.utility import readData, stratified_sampling
from utility.object import Classification, Regression

"""
随机样本
"""


def func1_X(seed: int = 0, size: int = 1000) -> np.array:
    np.random.seed(seed)
    return np.random.randn(size, 10)  # 标准正态分布


def func2_X(seed: int = 0, size: int = 1000) -> np.array:
    np.random.seed(seed)
    return np.random.f(4, 4, size * 10).reshape(size, 10)  # F 分布


def func3_X(seed: int = 0, size: int = 1000) -> np.array:
    np.random.seed(seed)
    X1 = np.random.chisquare(10, size * 10).reshape(size, 10)  # 卡方分布
    X2 = np.random.uniform(100, 130, size * 10).reshape(size, 10)  # 均匀分布
    X3 = np.random.f(4, 4, size * 10).reshape(size, 10)  # F 分布
    return np.hstack((X1[:, 0:3], X2[:, 4:6], X3[:, 3:7]))


def stock_MC_X(seed: int = 0, size: int = 1000) -> pd.DataFrame:
    """
    股价服从几何布朗运动，收益率服从均值为μ-1/2*σ^2，方差为σ^2/T的正态分布
    """
    np.random.seed(seed)
    index5min = readData(r'', "000905SH5minBar", 'csv')
    close5min = pd.pivot(data=index5min, index='tradeDate', columns='tradeTime', values='close')
    close5min.insert(0, 'last', close5min['15:00:00'].shift(1))
    ret5min = close5min.pct_change(axis=1).drop(columns='last')
    param = pd.DataFrame([ret5min.mean(), ret5min.std()], index=['miu', 'sigma']).T

    # sampling Price
    SimulatedRet = {}
    for ind, row in param.iterrows():
        SimulatedRet[ind] = np.random.normal(0, row['sigma'], size + 2)
    samplePrice = (pd.DataFrame(SimulatedRet).stack() + 1).cumprod().unstack()

    # sampling volume
    volD = pd.pivot(data=index5min, index='tradeDate', columns='tradeTime', values='volume').sum(axis=1)
    alpha, scale = volD.mean() ** 2 / volD.var(), volD.mean() / volD.var()
    sampleVol = np.random.gamma(alpha, 1 / scale, size + 2)  # scale 是beita的倒数

    res = pd.DataFrame({
        "close": samplePrice.iloc[:, -1],
        "open": samplePrice.iloc[:, 0],
        "high": samplePrice.max(axis=1),
        "low": samplePrice.min(axis=1),
        "volume": sampleVol})
    res['volume'] = round(res['volume'], -2)
    res['amount'] = round(res['volume'] * res['close'] * 100)

    return res


def index300_X(seed: int = 0, size: int = 1000) -> pd.DataFrame:
    indexData = readData(r'..\data_API\data', "Index", 'csv')
    # indexData = readData(r'..\database', "index_905", 'csv')
    return indexData


def func1(X, seed: int = 1, noise: bool = False, size: int = 1000, **kwargs):
    Y = X[:, 0] + X[:, 4] + X[:, 8] ** 2 + X[:, 6] / (abs(X[:, 3]) + 1)
    if noise:
        noise = np.random.normal(0, np.std(Y), size)
        Y_n = X[:, 0] + X[:, 4] + X[:, 8] ** 2 + X[:, 6] / (abs(X[:, 3]) + 1) + noise
        return X, Y_n, 1 - np.var(noise) / np.var(Y_n)
    else:
        return X, Y, 1


def func2(X, seed: int = 1, noise: bool = False, size: int = 1000, **kwargs):
    Y = bk.move_mean(X[:, 0], 5, min_count=1) + np.where(X[:, 4] > 0.5, 1, -1) - np.where(X[:, 2] > X[:, 1], 1, -1)
    if noise:
        noise = np.random.normal(0, np.std(Y), size)
        Y_n = bk.move_mean(X[:, 0], 5, min_count=1) + np.where(X[:, 4] > 0.5, 1, -1) - np.where(X[:, 2] > X[:, 1], 1,
                                                                                                -1) + noise * 0.9
        return X, Y_n, 1 - np.var(noise) / np.var(Y_n)
    else:
        return X, Y, 1


def func3(X, seed: int = 1, noise: bool = False, size: int = 1000, **kwargs):
    Y = np.sin(X[:, 1]) ** 2 + np.cos(X[:, 2]) * 4 + - abs(X[:, 5]) + X[:, 8] * X[:, 3] * X[:, 2]
    if noise:
        noise = np.random.normal(0, np.std(Y), size)
        Y_n = np.sin(X[:, 1]) ** 2 + np.cos(X[:, 2]) * 4 + - abs(X[:, 5]) + X[:, 8] * X[:, 3] * X[:, 2] + noise
        return X, Y_n, 1 - np.var(noise) / np.var(Y_n)
    else:
        return X, Y, 1


def stock_MC(X, seed: int = 1, noise: Any = False, size: int = 1000, **kwargs):
    random.seed(seed)
    np.random.seed(seed)
    COL = ['close', 'open', 'high', 'low', 'volume', 'amount']
    # 原始标签
    X['Y'] = X['open'].pct_change().shift(-2)
    # 去除空值
    X.dropna(axis=0, how='any', inplace=True)

    # 生成信号
    if kwargs['strategy'] == 'BOLL':
        # 1.布林通道策略
        X['closeMid'] = bk.move_mean(X['close'], 5, min_count=1)
        X['closeStd'] = bk.move_std(X['close'], 5, min_count=1)
        X['Down'] = X['closeMid'] - X['closeStd']  # 布林带下轨
        X['Up'] = X['closeMid'] + X['closeStd']  # 布林带上轨
        X['signal'] = np.where(X['close'] > X['Up'], 1, 0)  # 0.99 *
        # X['signal2'] = np.where(X['close'] > X['closeMid'], 1, 0)
        # X['signal_shift1'] = X['signal1'].shift(1)
        # X['signal3'] = np.where(X[['signal2', 'signal_shift1']].sum(axis=1) == 2, 1, 0)
        # X['signal'] = np.sign(X[['signal1', 'signal3']].sum(axis=1))
    elif kwargs['strategy'] == 'Double MA':
        # 2.双均线策略
        X['MA5'] = bk.move_mean(X['close'], 5, min_count=1)
        X['MA20'] = bk.move_mean(X['close'], 20, min_count=1)
        X['signal'] = np.where(X['MA5'] > 0.999 * X['MA20'], 1, 0)
    elif kwargs['strategy'] == 'ATR':
        X['TR1'] = X['high'] - X['low']
        X['TR2'] = abs(X['high'] - X['close'].shift(1).bfill())
        X['TR3'] = abs(X['low'] - X['close'].shift(1).bfill())
        X['TR'] = X[['TR1', 'TR2', 'TR3']].max(axis=1)
        X['ATR'] = bk.move_mean(X['TR'], 5, min_count=1)
        X['closeMid'] = bk.move_mean(X['close'], 5, min_count=1)
        X['Down'] = X['closeMid'] - 0.5 * X['ATR']
        X['Up'] = X['closeMid'] + 0.5 * X['ATR']
        X['signal'] = np.where(X['close'] > 0.99 * X['Up'], 1, 0)
    elif kwargs['strategy'] == 'Dual Thrust':
        X['HH-LC'] = bk.move_max(X['high'], 5, min_count=1) - bk.move_min(X['close'], 5, min_count=1)
        X['HC-LL'] = bk.move_max(X['close'], 5, min_count=1) - bk.move_min(X['low'], 5, min_count=1)
        X['range'] = X[['HH-LC', 'HC-LL']].max(axis=1)
        X['Up'] = X['open'] + 0.2 * X['range']
        X['Down'] = X['open'] - 0.2 * X['range']
        X['signal'] = np.where(X['close'] > 0.99 * X['Up'], 1, 0)
    elif kwargs['strategy'] == 'Fairy Four Price':
        X['Up'] = X['high'].shift(1).bfill()
        X['Down'] = X['low'].shift(1).bfill()
        X['signal'] = np.where(X['close'] > 0.99 * X['Up'], 1, 0)
    elif kwargs['strategy'] == 'Momentum':
        X['ret'] = X['close'].pct_change(1).fillna(0)
        X['signal'] = np.where(X['ret'] > 0.002, 1, 0)
    elif kwargs['strategy'] == 'Creation1':
        X['corr'] = X['close'].rolling(5, min_periods=1).corr(X['amount']).fillna(0)
        X['signal'] = np.where(X['corr'] > 0.06, 1, 0)
    elif kwargs['strategy'] == 'Creation2':
        X['openR'] = bk.move_rank(X['open'], 5, min_count=1)
        X['amountR'] = bk.move_rank(X['amount'], 5, min_count=1)
        X['highR'] = bk.move_argmin(X['high'], 5, min_count=1)
        X['lowR'] = bk.move_argmax(X['low'], 5, min_count=1)
        X['cor1'] = X['lowR'].rolling(5, min_periods=1).corr(X['highR']).fillna(0)
        X['cor2'] = X['amountR'].rolling(5, min_periods=1).corr(X['openR']).fillna(0)
        X['signal'] = np.where(((X['openR'] + X['highR']) > 1 * (X['amountR'] + X['lowR'])), 1, 0)
    elif kwargs['strategy'] == 'Creation3':
        X['sign1'] = X['close'] + X['high'] + X['low'] + X['open']
        X['sign2'] = bk.move_rank(X['volume'], 5, min_count=1)
        X['corr'] = X['sign1'] * X['sign2']
        X['signal'] = np.where(X['corr'] > 0, 1, 0)
    elif kwargs['strategy'] == 'True Boll':
        X = index300_X()
        # 标签
        X['Y'] = X['open'].pct_change(fill_method=None).shift(-2)
        X.dropna(how='any', axis=0, inplace=True)
        X['closeMid'] = bk.move_mean(X['close'], 5, min_count=1)
        X['closeStd'] = bk.move_std(X['close'], 5, min_count=1)
        X['Down'] = X['closeMid'] - X['closeStd']  # 布林带下轨
        X['Up'] = X['closeMid'] + X['closeStd']  # 布林带上轨
        X['signal'] = np.where(X['close'] > 1.004 * X['Up'], 1, 0)
        size = X.shape[0]
    elif kwargs['strategy'] == 'True Label':
        X = index300_X()
        # 标签
        X['Y'] = X['open'].pct_change(fill_method=None).shift(-2)
        X.dropna(how='any', axis=0, inplace=True)
        X['signal'] = np.where(X['Y'] > X['Y'].mean(), 1, 0)
        size = X.shape[0]
    else:
        X['signal'] = 0

    # 加噪音
    X['signalN'] = X['signal'] = np.sign(X['signal'])
    X['noise'] = np.random.randint(0, 2, size)
    noise_index = np.random.choice(X['noise'].index, int(size * noise), replace=False)
    X.loc[noise_index, 'signalN'] = X.loc[noise_index, 'noise']

    # 样本划分
    trainNum = int(size * (1 - kwargs['verify'] * 2))
    verifyNum = int(size * kwargs['verify'])
    testNum = size - trainNum - verifyNum

    sampleLabel = np.concatenate((np.array([0] * trainNum), np.array([1] * verifyNum), np.array([2] * testNum)))

    random.shuffle(sampleLabel)

    res = Classification(
        X=X[COL].values,
        Y=X['Y'].values,

        YSign=X['signalN'].values,
        sign=X['signal'].values,

        noiseR=noise,

        signNum=sum(X['signalN']),
        signTrainNum=sum(np.where(sampleLabel == 0, X['signalN'], 0)),
        signVerifyNum=sum(np.where(sampleLabel == 1, X['signalN'], 0)),
        signTestNum=sum(np.where(sampleLabel == 2, X['signalN'], 0)),
        signTestR=sum(np.where(sampleLabel == 2, X['signal'], 0)),

        Label=sampleLabel
    )
    return res


def index300(X, seed: int = 1, noise: Any = False, size: int = 1000, ocu: float = 0.1, **kwargs):
    random.seed(seed)
    np.random.seed(seed)

    # COL = ['close', 'preClose', 'ret', 'open', 'high', 'low', 'turnover', 'volume', 'amount']
    COL = ['close', 'open', 'high', 'low', 'volume', 'amount', 'open_1']
    # 标签
    X['open_1'] = X['open'].shift(-1)
    X['Y'] = X['open'].pct_change(fill_method=None).shift(-2)
    X.dropna(how='any', axis=0, inplace=True)
    size = X.shape[0]

    X['TR1'] = X['high'] - X['low']
    X['TR2'] = abs(X['high'] - X['close'].shift(1).bfill())
    X['TR3'] = abs(X['low'] - X['close'].shift(1).bfill())
    X['TR'] = X[['TR1', 'TR2', 'TR3']].max(axis=1)
    X['ATR'] = bk.move_mean(X['TR'], 5, min_count=1)
    # X['signal1'] = np.sign(X['ATR'].rolling(3).apply(lambda x: dirichlet.logpdf(np.array([0.40, 0.32, 0.28]), x))).fillna(0)  # [0.41, 0.33, 0.26] 20%
    # X['signal'] = np.where(X['signal1'] < 0, kwargs['ratio'], 0)
    X['closeMid'] = bk.move_mean(X['close'], 5, min_count=1)
    # X['Down'] = X['closeMid'] - X['ATR']
    # if ocu == 0.1:
    #     rr = 1.06
    # else:
    #     rr = 0.78
    X['Up'] = X['closeMid'] + 0.76 * X['ATR']
    X['signal'] = np.where(X['close'] > X['Up'], 0, 0)
    signNeg = sum(X['signal']) / (size - np.count_nonzero(X['signal']))
    X['signal'] = np.where(X['signal'] == 0, - signNeg, X['signal'])
    X['Y'] = X['Y'] + 0

    # 样本划分:为避免样本不均匀采用间隔抽样的方式
    trainNum = 800
    verifyNum = 140
    verify2Num = 140
    testNum = size - trainNum - verifyNum - verify2Num

    sampleLabel = np.concatenate((np.array([0] * trainNum),
                                  np.array([1] * verifyNum),
                                  np.array([2] * verify2Num),
                                  np.array([3] * testNum)))

    # sampleLabel = stratified_sampling(size, kwargs['sampleOdds'], seed)

    random.shuffle(sampleLabel)

    res = Classification(
        X=X[COL].values,
        Y=X['Y'].values,

        YSign=X['signal'].values,
        sign=np.where(X['signal'] > 0, 1, 0),

        signNum=sum(np.where(X['signal'] > 0, 1, 0)),
        signTrainNum=sum(np.where((sampleLabel == 0) & (X['signal'] > 0), 1, 0)),
        signVerifyNum=sum(np.where((sampleLabel == 1) & (X['signal'] > 0), 1, 0)),
        signVerify2Num=sum(np.where((sampleLabel == 2) & (X['signal'] > 0), 1, 0)),
        signTestNum=sum(np.where((sampleLabel == 3) & (X['signal'] > 0), 1, 0)),

        Label=sampleLabel
    )
    return res


def select_sample(funcName: str,
                  noise: bool = False,
                  seed: int = 1,
                  size: int = 1000,
                  **kwargs) -> dataclass:
    """
    返回随机生成的样本对
    :param funcName: 函数名
    :type funcName: str
    :param noise: 是否包含噪音
    :type noise: bool
    :param seed: 种子数
    :type seed: int
    :param size: 样本规模
    :type size:  int
    :return: (X, Y)
    :rtype: np.array
    """
    X = eval(funcName + '_X')(seed, size)
    sample = eval(funcName)(X, seed, noise, size, **kwargs)
    return sample


if __name__ == '__main__':
    x, y, error = select_sample('stock_MC')
    print('s')
