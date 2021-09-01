# -*-coding:utf-8-*-
# @Time:   2021/8/27 14:28
# @Author: FC
# @Email:  18817289038@163.com

import sys
import random
import numpy as np
import pandas as pd
import bottleneck as bk
import matplotlib.pyplot as plt

from typing import Tuple, Any
from dataclasses import dataclass

from utility.utility import readData, stratified_sampling
from utility.object import Regression, DataInfo

COL = ['openAdj', 'closeAdj', 'highAdj', 'lowAdj', 'volume', 'amount', 'liqMv']

PATH = {
    "stock_feature": {"path": r'../data_API/data', "file": 'AStockData', 'fileType': 'pkl'},
    "stock_pool": {"path": r'../data_API/data', "file": 'StockPoolZD', 'fileType': 'pkl'},
    "label_pool": {"path": r'../data_API/data', "file": 'strategyLabel', 'fileType': 'pkl'},
    "stock_benchmark": {"path": r'../data_API/data', "file": 'benchmarkZD', 'fileType': 'pkl'},
    "stock_factor": {"path": r'../data_API/data', "file": 'strategyLabel', 'fileType': 'pkl'}}


class DataAPI(object):

    @staticmethod
    def read_file(fileName: str) -> Any:
        fileInfo = PATH.get(fileName, None)
        if fileInfo is None:
            print('fileName error!')
        else:
            return readData(fileInfo['path'], fileInfo['file'], fileInfo['fileType'])

    def stock_feature(self, **kwargs) -> pd.DataFrame:
        stockData = self.read_file(sys._getframe().f_code.co_name)
        stockDataSub = stockData[(stockData['date'] >= '2017') & (stockData['date'] < '2019')]
        stockDataSub.set_index(['date', 'code'], inplace=True)
        return stockDataSub

    def stock_pool(self, **kwargs) -> pd.DataFrame:
        stockPool = self.read_file(sys._getframe().f_code.co_name)
        stockPoolSub = stockPool[(stockPool['date'] >= '2017') & (stockPool['date'] < '2019')]
        stockPoolSub.set_index(['date', 'code'], inplace=True)
        return stockPoolSub

    def label_pool(self, **kwargs) -> pd.DataFrame:
        labelPool = self.read_file(sys._getframe().f_code.co_name)
        labelPoolSub = labelPool[(labelPool['date'] >= '2017') & (labelPool['date'] < '2019')]
        labelPoolSub.set_index(['date', 'code'], inplace=True)
        return labelPoolSub

    def stock_benchmark(self, **kwargs) -> pd.DataFrame:
        benchmark = self.read_file(sys._getframe().f_code.co_name)
        benchmarkSub = benchmark[(benchmark['date'] >= '2017') & (benchmark['date'] < '2019')]
        benchmarkSub.set_index('date', inplace=True)
        return benchmarkSub

    def stock_factor(self, **kwargs) -> pd.DataFrame:
        factor = self.read_file(sys._getframe().f_code.co_name)
        return factor


class StockGateWay(DataAPI):

    def get_data(self, funcName: str, **kwargs) -> dataclass:
        dataInput = getattr(self, funcName, None)(**kwargs)
        res = getattr(self, funcName + '_data', None)(dataInput, **kwargs)
        return res

    # 特征数据
    def stock_feature_data(self, data: pd.DataFrame, **kwargs) -> dataclass:
        np.random.seed(kwargs.get('seed', 1))

        data['ret'] = data['openAdj'].groupby('code').pct_change(periods=1, fill_method=None)
        data['ret'] = data['ret'].groupby('code').shift(-2)
        data.dropna(how='any', inplace=True)

        trading_date = data.index.get_level_values('date').drop_duplicates()

        # 交叉样本
        trainN = int(len(trading_date) * 0.7)
        verifyN = int(len(trading_date) * 0.1)
        verify2N = int(len(trading_date) * 0.1)
        testN = len(trading_date) - trainN - verifyN - verify2N

        sampleLabel = np.concatenate((np.array([0] * trainN),
                                      np.array([1] * verifyN),
                                      np.array([2] * verify2N),
                                      np.array([3] * testN)))
        np.random.shuffle(sampleLabel)

        res = Regression(
            dataRaw=data,

            X=data[COL].copy(deep=True),  # 特征
            Y=data['ret'].copy(deep=True),  # 收益率标签

            Label=pd.Series(sampleLabel, index=trading_date)
        )
        return res

    # 股票池
    def stock_pool_data(self, data: pd.DataFrame, **kwargs) -> dataclass:
        return DataInfo(data=data, dataType='stockPool')

    # 标签池（风格因子）
    def label_pool_data(self, data: pd.DataFrame, **kwargs) -> dataclass:
        return DataInfo(data=data, dataType='labelPool')

    # 基准
    def stock_benchmark_data(self, data: pd.DataFrame, **kwargs) -> dataclass:
        return DataInfo(data=data, dataType='benchmark')

    # 风格因子：行业指数和市值也放在一起
    def stock_factor_data(self, data: pd.DataFrame, **kwargs) -> dataclass:
        return DataInfo(data=data, dataType='styleFactor')


if __name__ == '__main__':
    A = StockGateWay()
    A.get_data('stock_feature')
