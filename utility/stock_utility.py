# -*-coding:utf-8-*-
# @Time:   2021/8/27 14:57
# @Author: FC
# @Email:  18817289038@163.com


import time
import numpy as np
import pandas as pd

from copy import deepcopy
from typing import Any, Union, List, Tuple, Dict


# 因子处理方法合集
class MethodSets(object):
    # 方法参数必须被继承复写

    methodProcess = {
        "RO": {"method": "", "p": {}},  # 异常值处理
        "Neu": {"method": "", "p": {}},  # 中性化处理
        "Sta": {"method": "", "p": {}},  # 标准化处理
    }

    def __init__(self):
        self.RO = RemoveOutlier()
        self.Neu = Neutralization()
        self.Sta = Standardization()

    # 更新参数
    def set_params(self, **kwargs):
        """

        Parameters
        ----------
        Returns
        -------
        对于因子处理方法设置因子参数
        """
        for paramName, paramValue in kwargs.items():
            setattr(self, paramName, paramValue)

    def processSingle(self,
                      data: Union[pd.DataFrame, pd.Series],
                      methodN: str,
                      **kwargs) -> Any:
        """
        单一处理方法
        Parameters
        ----------
        data :
        methodN : 方法名
        kwargs :

        Returns
        -------

        """
        value = getattr(self, methodN).process(data=data,
                                               method=self.methodProcess[methodN]['method'],
                                               **self.methodProcess[methodN]['p'],
                                               **kwargs)
        return value

    def processSeq(self,
                   data: pd.DataFrame,
                   methodN: List[str],
                   dataName: str,
                   **kwargs):
        """
        连续处理
        Parameters
        ----------
        data :
        methodN : 方法名list，按顺序执行
        dataName :
        kwargs :

        Returns
        -------

        """
        for M in methodN:
            if self.methodProcess[M]['method'] != "":
                value = getattr(self, M).process(data=data,
                                                 method=self.methodProcess[M]['method'],
                                                 dataName=dataName,
                                                 **self.methodProcess[M]['p'],
                                                 **kwargs)
                data[dataName] = value


# 去极值
class RemoveOutlier(object):
    dataName = ''

    def process(self,
                data: pd.DataFrame,
                dataName: str,
                method='before_after_3%',
                **kwargs) -> pd.Series(float):

        self.dataName = dataName

        method_dict = {
            "before_after_3%": self.before_after_n,
            "before_after_3sigma": self.before_after_3sigma,
            "mad": self.mad
        }
        if method is None:
            return data
        else:
            res = method_dict[method](data, **kwargs)
            return res

    """去极值"""

    # 前后3%
    def before_after_n(self,
                       data: pd.DataFrame,
                       n: int = 3,
                       **kwargs) -> pd.Series(float):
        data_df = data[self.dataName].unstack()
        threshold_down, threshold_up = data_df.quantile(n / 100, axis=1), data_df.quantile(1 - n / 100, axis=1)
        res = data_df.clip(threshold_down, threshold_up, axis=0).stack()
        return res

    # 3倍标准差外
    def before_after_3sigma(self,
                            data: pd.DataFrame,
                            **kwargs) -> pd.Series(float):
        data_df = data[self.dataName].unstack()
        miu, sigma = data_df.mean(axis=1), data_df.std(axis=1)
        threshold_down, threshold_up = miu - 3 * sigma, miu + 3 * sigma
        res = data_df.clip(threshold_down, threshold_up, axis=0).stack()
        return res

    # 绝对中位偏差法
    def mad(self,
            data: pd.DataFrame,
            **kwargs) -> pd.Series(float):
        data_df = data[self.dataName].unstack()
        median = data_df.median(axis=1)
        MAD = data_df.sub(median, axis=0).abs().median(axis=1)
        threshold_up, threshold_down = median + 3 * 1.483 * MAD, median - 3 * 1.483 * MAD
        res = data_df.clip(threshold_down, threshold_up, axis=0).stack()
        return res


# 中性化
class Neutralization(object):
    dataName = ''

    def process(self,
                data: pd.Series,
                dataName: str,
                mvName: str = 'liqMv',
                indName: str = 'indexCode',
                method: str = 'industry+mv',
                **kwargs) -> pd.Series(float):
        """
        若同时纳入行业因子和市值因子需要加上截距项，若仅纳入行业因子则回归方程不带截距项！
        :param data: 因子数据
        :param dataName: 因子数据
        :param mvName: 市值名称
        :param indName: 行业指数名称
        :param method: 中心化方法
        :return: 剔除行业因素和市值因素后的因子

        Args:
            indName ():
            mvName ():

        Parameters
        ----------
        factName :
        """
        self.dataName = dataName

        colName = [self.dataName]
        # read mv and industry data
        if 'mv' in method:
            colName.append(mvName)

        if 'industry' in method:
            colName.append(indName)

        # remove Nan
        dataNew = data[colName].dropna(how='any').copy()
        # neutralization
        res = dataNew.groupby('date', group_keys=False).apply(self.reg)
        return res

    # regression
    def reg(self, data: pd.DataFrame) -> pd.Series(float):
        """！！！不排序回归结果会不一样！！！"""
        dataSub = data.sort_index()
        X = pd.get_dummies(dataSub.loc[:, dataSub.columns != self.dataName], columns=['indexCode'])
        Y = dataSub[self.dataName]
        reg = np.linalg.lstsq(X, Y, rcond=None)
        factNeu = Y - (reg[0] * X).sum(axis=1)
        return factNeu


# 标准化
class Standardization(object):
    dataName = ''

    def process(self,
                data: pd.DataFrame,
                dataName: str,
                method='z_score',
                **kwargs) -> pd.Series(float):
        self.dataName = dataName

        method_dict = {"range01": self.range01,
                       "z_score": self.z_score,
                       "mv": self.market_value_weighting
                       }

        if method is None:
            return data

        res = method_dict[method](data, **kwargs)

        return res

    """标准化"""

    # 标准分数法
    def z_score(self,
                data: pd.DataFrame,
                **kwargs):
        """
        :param data:
        :return:
        """
        data_df = data[self.dataName].unstack()
        miu, sigma = data_df.mean(axis=1), data_df.std(axis=1)
        stand = data_df.sub(miu, axis=0).div(sigma, axis=0).stack()
        return stand

    def range01(self,
                data: pd.DataFrame,
                **kwargs):
        data_df = data[self.dataName]
        denominator, numerator = data_df.max(axis=1) - data_df.min(axis=1), data_df.sub(data_df.min(axis=1), axis=0)
        result = numerator.div(denominator, axis=0).stack()
        return result

    # 市值加权标准化
    def market_value_weighting(self,
                               data: pd.DataFrame,
                               mvName: str = 'liqMv',
                               **kwargs) -> pd.Series(float):
        data_df = data[[self.dataName, mvName]].dropna(how='any')
        dataFact = data_df[self.dataName].unstack()
        dataMv = data_df[mvName].unstack()

        miu, std = (dataFact * dataMv).div(dataMv.sum(axis=1), axis=0).sum(axis=1), dataFact.std(axis=1)
        res = dataFact.sub(miu, axis=0).div(std, axis=0).stack()

        return res


# 持有期收益率计算
def holding_ret(ret: pd.Series, hp: int) -> pd.Series:
    """
    计算持有不同周期的股票收益率
    :param ret: 股票收益率序列
    :param hp:
    :return:
    """

    # Holding period return
    ret = ret.add(1)

    ret_label = 1
    for shift_ in range(hp):
        ret_label *= ret.groupby('code').shift(- shift_)

    ret_label = ret_label.sub(1)

    return ret_label


def weight_cor(data: pd.DataFrame,
               weight: Union[List, pd.Series, np.arange]) -> float:
    """
    加权相关系数计算：加权协方差和加权方差
    """
    data_array, weight_array = np.array(data.T), np.array(weight)
    # calculate the weighted variance
    cov_weight = np.cov(data_array, aweights=weight_array)
    # calculate the weighted covariance
    var_weight_A = np.cov(data_array[0], aweights=weight_array)
    var_weight_B = np.cov(data_array[-1], aweights=weight_array)
    # calculate the weighted correlation
    corr_weight = cov_weight / pow((var_weight_A * var_weight_B), 0.5)

    return corr_weight[0][1]
