# -*-coding:utf-8-*-
# @Time:   2021/7/9 9:40
# @Author: FC
# @Email:  18817289038@163.com

import time
import scipy
import inspect
import numpy as np
import pandas as pd
import bottleneck as bk
from typing import Dict, Any

from utility.object import FuncAtt
from utility.utility import (
    checkDim,
    substitute,
    rolling_window,
    norm_to_uniform,
    uniform_to_uniform,
    SMALL_VALUE
)


# 算子基类
class EvolutionOperator(object):
    """
    基类提供基础算子
    """
    funcAtt: Dict[str, FuncAtt] = {}

    def __init__(self):
        self.func()

    def func(self):
        self.funcAtt.clear()
        for func_name in dir(self):
            if func_name.endswith('EVO'):
                _func_ = getattr(self, func_name)

                arity = getattr(_func_, "__wrapped__", _func_).__code__.co_argcount
                arity = arity - 1 if inspect.ismethod(_func_) else arity

                self.funcAtt[func_name] = FuncAtt(func_name,
                                                  _func_,
                                                  arity
                                                  )


class GEPFunctionTiming(EvolutionOperator):
    __doc__ = """
    基因表达式规划算子类: 择时信号挖掘相关算子
    """
    dim: int = 10

    def __init__(self):
        super().__init__()
        self.funcP = lambda x1, x2: [i for i in range(x1, x2)]
        self.paramL = 400

    """基础函数1"""

    @substitute
    def add_EVO(self, data1: Any, data2: Any):
        data1, data2 = checkDim(data1, data2, dim=self.dim)
        return data1 + data2

    @substitute
    def sub_EVO(self, data1: Any, data2: Any):
        data1, data2 = checkDim(data1, data2, dim=self.dim)
        return data1 - data2

    @substitute
    def mul_EVO(self, data1: Any, data2: Any):
        data1, data2 = checkDim(data1, data2, dim=self.dim)
        return data1 * data2

    @substitute
    def div_EVO(self, data1: Any, data2: Any):
        data1, data2 = checkDim(data1, data2, dim=self.dim)
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.where(np.abs(data2) > SMALL_VALUE, data1 / data2, 0.)

    # @substitute
    # def mean_EVO(self, data1: Any, data2: Any):
    #     data1, data2 = checkDim(data1, data2, dim=self.dim)
    #     return (data1 + data2) / 2

    # @substitute
    # def max_EVO(self, data1: Any, data2: Any):
    #     data1, data2 = checkDim(data1, data2, dim=self.dim)
    #     return np.maximum(data1, data2)
    #
    # @substitute
    # def min_EVO(self, data1: Any, data2: Any):
    #     data1, data2 = checkDim(data1, data2, dim=self.dim)
    #     return np.minimum(data1, data2)

    @substitute
    def abs_EVO(self, x: Any):
        x, = checkDim(x, dim=self.dim)
        return np.abs(x)

    # @substitute
    # def neg_EVO(self, x: Any):
    #     x, = checkDim(x, dim=self.dim)
    #     return - x

    """基础函数2"""

    @substitute
    def sqrt_EVO(self, x: Any):
        x, = checkDim(x, dim=self.dim)
        return np.sign(x) * np.sqrt(abs(x))

    @substitute
    def log_EVO(self, x: Any):
        x, = checkDim(x, dim=self.dim)
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.sign(x) * np.where(np.abs(x) > SMALL_VALUE, np.log(np.abs(x)), 0.)

    # @substitute
    # def square_EVO(self, x: Any):
    #     x, = checkDim(x, dim=self.dim)
    #     return pow(x, 2)

    # @substitute
    # def sin_EVO(self, x: Any):
    #     x, = checkDim(x, dim=self.dim)
    #     return np.sin(x)
    #
    # @substitute
    # def cos_EVO(self, x: Any):
    #     x, = checkDim(x, dim=self.dim)
    #     return np.cos(x)

    # @substitute
    # def inv_EVO(self, x: Any):
    #     x, = checkDim(x, dim=self.dim)
    #     with np.errstate(divide='ignore', invalid='ignore'):
    #         return np.where(np.abs(x) > SMALL_VALUE, 1 / x, 0.)

    @substitute
    def sign_EVO(self, x: Any):
        x, = checkDim(x, dim=self.dim)
        return np.sign(x)

    # @substitute
    # def signal2_EVO(self, x: Any):
    #     x, = checkDim(x, dim=self.dim)
    #     return np.where(x < 0, -1, 0)

    @substitute
    def compare_EVO(self, data1: Any, data2: Any):
        data1, data2 = checkDim(data1, data2, dim=self.dim)
        return np.sign(data1 - data2)

    # @substitute
    # def compare2_EVO(self, data1: Any, data2: Any):
    #     data1, data2 = checkDim(data1, data2, dim=self.dim)
    #     return np.where(data1 > data2, -1, 0)

    """规则构建函数"""

    @substitute
    def ts_mean_EVO(self, x: Any, y: Any):
        res = np.array([0] * self.dim)
        x, y, flag = checkDim(x, y, para=-1, dim=self.dim)
        if flag:
            para = round(abs(y))
            if para >= 5:
                res = bk.move_mean(x, para, min_count=1)
        return res

    """其余函数"""

    @substitute
    def ts_delay_EVO(self, x: Any, y: Any):
        res = np.array([0] * self.dim)
        x, y, flag = checkDim(x, y, para=-1, dim=self.dim)
        if flag:
            para = round(abs(y))
            if para != 0:
                res = np.concatenate((x[:para], x[:-para]))
        return res

    # @substitute
    # def ts_delta_EVO(self, x: Any, y: Any):
    #     res = np.array([0] * self.dim)
    #     x, y, flag = checkDim(x, y, para=-1, dim=self.dim)
    #     if flag:
    #         para = round(abs(y))
    #         if para != 0:
    #             res = x - np.concatenate((x[:para], x[:-para]))
    #     return res

    @substitute
    def ts_max_EVO(self, x: Any, y: Any):
        res = np.array([0] * self.dim)
        x, y, flag = checkDim(x, y, para=-1, dim=self.dim)
        if flag:
            para = round(abs(y))
            if para >= 5:
                res = bk.move_max(x, para, min_count=1)

        return res

    @substitute
    def ts_min_EVO(self, x: Any, y: Any):
        res = np.array([0] * self.dim)
        x, y, flag = checkDim(x, y, para=-1, dim=self.dim)
        if flag:
            para = round(abs(y))
            if para >= 5:
                res = bk.move_min(x, para, min_count=1)
        return res

    @substitute
    def ts_std_EVO(self, x: Any, y: Any):
        res = np.array([0] * self.dim)
        x, y, flag = checkDim(x, y, para=-1, dim=self.dim)
        if flag:
            para = round(abs(y))
            if para >= 5:
                res = bk.move_std(x, para, min_count=1)
        return res

    # @substitute
    # def ts_skew_EVO(self, x: Any):
    #     x = checkDim(x, dim=self.dim)[0]
    #     return pd.Series(x).rolling(5, min_periods=1).skew().fillna(0).values
    #
    # @substitute
    # def ts_kurt_EVO(self, x: Any):
    #     x = checkDim(x, dim=self.dim)[0]
    #     return pd.Series(x).rolling(5, min_periods=1).kurt().fillna(0).values

    # @substitute
    # def ts_product_EVO(self, x: Any):
    #     """
    #     长度不足则补1
    #     """
    #     x = checkDim(x, dim=self.dim)[0]
    #     sub = np.product(rolling_window(x, 5), axis=1)
    #     return np.concatenate((np.cumprod(x[:4]), sub))

    @substitute
    def ts_rank_EVO(self, x: Any, y: Any):
        res = np.array([0] * self.dim)
        x, y, flag = checkDim(x, y, para=-1, dim=self.dim)
        if flag:
            para = round(abs(y))
            if para >= 5:
                res = bk.move_rank(x, para, min_count=5)
                res = np.where(np.isnan(res), 0, res)
        return res

    # @substitute
    # def ts_argmax_EVO(self, x: Any, y: Any):
    #     res = np.array([0] * self.dim)
    #     x, y, flag = checkDim(x, y, para=-1, dim=self.dim)
    #     if flag:
    #         para = round(abs(y))
    #         if para >= 5:
    #             res = para - bk.move_argmax(x, para, min_count=5)
    #             res = np.where(np.isnan(res), 0, res)
    #     return res
    #
    # @substitute
    # def ts_argmin_EVO(self, x: Any, y: Any):
    #     res = np.array([0] * self.dim)
    #     x, y, flag = checkDim(x, y, para=-1, dim=self.dim)
    #     if flag:
    #         para = round(abs(y))
    #         if para >= 5:
    #             res = para - bk.move_argmin(x, para, min_count=5)
    #             res = np.where(np.isnan(res), 0, res)
    #     return res
    #
    # @substitute
    # def ts_argmaxmin_EVO(self, x: Any, y: Any):
    #     res = np.array([0] * self.dim)
    #     x, y, flag = checkDim(x, y, para=-1, dim=self.dim)
    #     if flag:
    #         para = round(abs(y))
    #         if para >= 5:
    #             res = bk.move_argmax(x, para, min_count=5) - bk.move_argmin(x, para, min_count=5)
    #             res = np.where(np.isnan(res), 0, res)
    #
    #     return res

    @substitute
    def ts_corrP_EVO(self, data1: Any, data2: Any, data3: Any):
        """
        pearson相关系数
        """
        res = np.array([0] * self.dim)
        data1, data2, data3, flag = checkDim(data1, data2, data3, para=-1, dim=self.dim)
        if flag:
            para = round(abs(data3))
            if para >= 5:
                res = pd.Series(data1).rolling(para, min_periods=1).corr(pd.Series(data2)).fillna(0).values
                res[np.isinf(res)] = 0
        return res

    # @substitute
    # def ts_corrS_EVO(self, data1: Any, data2: Any, data3: Any):
    #     """
    #     秩相关系数
    #     """
    #     res = np.array([0] * self.dim)
    #     data1, data2, data3, flag = checkDim(data1, data2, data3, para=-1, dim=self.dim)
    #     if flag:
    #         para = round(abs(data3))
    #         if para >= 5:
    #             sub1, sub2 = rolling_window(data1, para), rolling_window(data2, para)
    #             sub1, sub2 = sub1.argsort(), sub2.argsort()
    #             corSub = np.diag(np.corrcoef(sub1, sub2)[:sub1.shape[0], -sub1.shape[0]:])
    #             res = np.concatenate((np.array([0] * (para - 1)), np.where(np.isnan(corSub), 0, corSub)))
    #     return res

    @substitute
    def ts_stand01_EVO(self, x: Any, y: Any):
        """
        常数标准化为0
        """
        res = np.array([0] * self.dim)
        x, y, flag = checkDim(x, y, para=-1, dim=self.dim)
        if flag:
            para = round(abs(y))
            if para >= 5:
                xMax = bk.move_max(x, para, min_count=1)
                xMin = bk.move_min(x, para, min_count=1)
                denominator = xMax - xMin
                with np.errstate(divide='ignore', invalid='ignore'):
                    res = np.where(denominator > SMALL_VALUE, (x - xMin) / denominator, 0)
        return res

    @substitute
    def ts_standNorm_EVO(self, x: Any, y: Any):
        """
        常数标准化为0
        """
        res = np.array([0] * self.dim)
        x, y, flag = checkDim(x, y, para=-1, dim=self.dim)
        if flag:
            para = round(abs(y))
            if para >= 5:
                xMean = bk.move_mean(x, para, min_count=1)
                xStd = bk.move_std(x, para, min_count=1)
                with np.errstate(divide='ignore', invalid='ignore'):
                    res = np.where(xStd > SMALL_VALUE, (x - xMean) / xStd, 0)
        return res

    @substitute
    def ts_reg_EVO(self, data1: Any, data2: Any, data3: Any):
        res = np.array([0] * self.dim)
        data1, data2, data3, flag = checkDim(data1, data2, data3, para=-1, dim=self.dim)

        if flag:
            para = round(abs(data3))
            if para >= 5:
                data = data1 * data2
                data1Mean = bk.move_mean(data1, para, min_count=1)
                data2Mean = bk.move_mean(data2, para, min_count=1)
                dataMean = bk.move_mean(data, para, min_count=1)
                denominator = bk.move_var(data1, para, min_count=1)
                with np.errstate(divide='ignore', invalid='ignore'):
                    res = np.where(denominator > SMALL_VALUE, (dataMean - data1Mean * data2Mean) / denominator, 0)
        return res


class GEPFunctionSelecting(EvolutionOperator):
    __doc__ = """
    基因表达式规划算子类: 选股信号挖掘相关算子
    """
    index: pd.MultiIndex = None

    def __init__(self):
        super().__init__()

    """基础函数"""
    @substitute
    def add_EVO(self, data1: Any, data2: Any) -> pd.Series:
        data1, data2 = checkDim(data1, data2, dim=len(self.index))
        return pd.Series(data1 + data2, index=self.index)

    @substitute
    def sub_EVO(self, data1: Any, data2: Any) -> pd.Series:
        data1, data2 = checkDim(data1, data2, dim=len(self.index))
        return pd.Series(data1 - data2, index=self.index)

    @substitute
    def mul_EVO(self, data1: Any, data2: Any) -> pd.Series:
        data1, data2 = checkDim(data1, data2, dim=len(self.index))
        return pd.Series(data1 * data2, index=self.index)

    @substitute
    def div_EVO(self, data1: Any, data2: Any) -> pd.Series:
        data1, data2 = checkDim(data1, data2, dim=len(self.index))
        with np.errstate(divide='ignore', invalid='ignore'):
            return pd.Series(np.where(np.abs(data2) > SMALL_VALUE, data1 / data2, 0.), index=self.index)

    # @substitute
    # def max_EVO(self, data1: Any, data2: Any):
    #     data1, data2 = checkDim(data1, data2, dim=len(self.index))
    #     return pd.Series(np.maximum(data1, data2), index=self.index)
    #
    # @substitute
    # def min_EVO(self, data1: Any, data2: Any):
    #     data1, data2 = checkDim(data1, data2, dim=len(self.index))
    #     return pd.Series(np.minimum(data1, data2), index=self.index)

    @substitute
    def abs_EVO(self, x: Any) -> pd.Series:
        x, = checkDim(x, dim=len(self.index))
        return pd.Series(np.abs(x), index=self.index)

    # @substitute
    # def neg_EVO(self, x: Any):
    #     x, = checkDim(x, dim=len(self.index))
    #     return - x

    @substitute
    def sqrt_EVO(self, x: Any) -> pd.Series:
        x, = checkDim(x, dim=len(self.index))
        return pd.Series(np.sign(x) * np.sqrt(abs(x)), index=self.index)

    @substitute
    def log_EVO(self, x: Any) -> pd.Series:
        x, = checkDim(x, dim=len(self.index))
        with np.errstate(divide='ignore', invalid='ignore'):
            return pd.Series(np.sign(x) * np.where(np.abs(x) > SMALL_VALUE, np.log(np.abs(x)), 0.), index=self.index)

    # # @substitute
    # # def square_EVO(self, x: Any):
    # #     x, = checkDim(x, dim=len(self.index))
    # #     return pow(x, 2)
    #
    # # @substitute
    # # def sin_EVO(self, x: Any):
    # #     x, = checkDim(x, dim=len(self.index))
    # #     return np.sin(x)
    # #
    # # @substitute
    # # def cos_EVO(self, x: Any):
    # #     x, = checkDim(x, dim=len(self.index))
    # #     return np.cos(x)
    #
    # # @substitute
    # # def inv_EVO(self, x: Any):
    # #     x, = checkDim(x, dim=len(self.index))
    # #     with np.errstate(divide='ignore', invalid='ignore'):
    # #         return np.where(np.abs(x) > SMALL_VALUE, 1 / x, 0.)

    @substitute
    def sign_EVO(self, x: Any) -> pd.Series:
        x, = checkDim(x, dim=len(self.index))
        return pd.Series(np.sign(x), index=self.index)

    @substitute
    def compare_EVO(self, data1: Any, data2: Any) -> pd.Series:
        data1, data2 = checkDim(data1, data2, dim=len(self.index))
        return pd.Series(np.sign(data1 - data2), index=self.index)

    """时间序列函数"""
    @substitute
    def ts_mean_EVO(self, x: Any, y: Any) -> pd.Series(float):
        res = pd.Series(0, index=self.index)
        x, y, flag = checkDim(x, y, para=-1, dim=len(self.index))

        if flag:
            para = round(abs(y))
            if para >= 5:
                x = pd.Series(x, index=self.index)
                res = pd.Series(bk.move_mean(x, para, min_count=1), index=self.index)
                x2 = x.groupby('code').head(para - 1)
                res.loc[x2.index] = x2
        return res

    @substitute
    def ts_delay_EVO(self, x: Any, y: Any) -> pd.Series(float):
        res = pd.Series(0, index=self.index)
        x, y, flag = checkDim(x, y, para=-1, dim=len(self.index))
        if flag:
            para = round(abs(y))
            if para != 0:
                x = pd.Series(x, index=self.index)
                res = x.shift(para)
                x2 = x.groupby('code').head(para)
                res.loc[x2.index] = x2
        return res

    # @substitute
    # def ts_delta_EVO(self, x: Any, y: Any) -> pd.Series(float):
    #     res = pd.Series(0, index=self.index)
    #     x, y, flag = checkDim(x, y, para=-1, dim=len(self.index))
    #     if flag:
    #         para = round(abs(y))
    #         if para != 0:
    #             x = pd.Series(x, index=self.index)
    #             x1 = x.shift(para)
    #             x2 = x.groupby('code').head(para)
    #             x1.loc[x2.index] = x2
    #             res = x - x1
    #     return res

    @substitute
    def ts_weight_mean_EVO(self, x: Any, y: Any) -> pd.Series(float):
        res = pd.Series(0, index=self.index)
        x, y, flag = checkDim(x, y, para=-1, dim=len(self.index))

        if flag:
            para = round(abs(y))
            if para != 0:
                weight = [i + 1 for i in range(para)]
                x1 = rolling_window(x, para)
                res = pd.Series((x1 * weight / sum(weight)).sum(axis=1), index=self.index)
                x2 = pd.Series(x, index=self.index).groupby('code').head(para)
                res.loc[x2.index] = x2
        return res

    @substitute
    def ts_max_EVO(self, x: Any, y: Any) -> pd.Series(float):
        res = pd.Series(0, index=self.index)
        x, y, flag = checkDim(x, y, para=-1, dim=len(self.index))
        if flag:
            para = round(abs(y))
            if para >= 5:
                x = pd.Series(x, index=self.index)
                res = pd.Series(bk.move_max(x, para, min_count=5), index=self.index)
                x2 = x.groupby('code').head(para - 1)
                res.loc[x2.index] = x2
        return res

    @substitute
    def ts_min_EVO(self, x: Any, y: Any) -> pd.Series(float):
        res = pd.Series(0, index=self.index)
        x, y, flag = checkDim(x, y, para=-1, dim=len(self.index))
        if flag:
            para = round(abs(y))
            if para >= 5:
                x = pd.Series(x, index=self.index)
                res = pd.Series(bk.move_min(x, para, min_count=5), index=self.index)
                x2 = x.groupby('code').head(para - 1)
                res.loc[x2.index] = x2
        return res

    @substitute
    def ts_std_EVO(self, x: Any, y: Any) -> pd.Series(float):
        res = pd.Series(0, index=self.index)
        x, y, flag = checkDim(x, y, para=-1, dim=len(self.index))
        if flag:
            para = round(abs(y))
            if para >= 5:
                res = pd.Series(bk.move_std(x, para, 5), index=self.index)
                x2 = res.groupby('code').head(para - 1)
                res.loc[x2.index] = 0
        return res

    # # @substitute
    # # def ts_skew_EVO(self, x: Any):
    # #     x = checkDim(x, dim=self.dim)[0]
    # #     return pd.Series(x).rolling(5, min_periods=1).skew().fillna(0).values
    # #
    # # @substitute
    # # def ts_kurt_EVO(self, x: Any):
    # #     x = checkDim(x, dim=self.dim)[0]
    # #     return pd.Series(x).rolling(5, min_periods=1).kurt().fillna(0).values
    #
    # # @substitute
    # # def ts_product_EVO(self, x: Any):
    # #     """
    # #     长度不足则补1
    # #     """
    # #     x = checkDim(x, dim=self.dim)[0]
    # #     sub = np.product(rolling_window(x, 5), axis=1)
    # #     return np.concatenate((np.cumprod(x[:4]), sub))

    @substitute
    def ts_rank_EVO(self, x: Any, y: Any) -> pd.Series(float):
        res = pd.Series(0, index=self.index)
        x, y, flag = checkDim(x, y, para=-1, dim=len(self.index))
        if flag:
            para = round(abs(y))
            if para >= 5:
                res = pd.Series(bk.move_rank(x, para, min_count=5), index=self.index)
                x2 = res.groupby('code').head(para - 1)
                res.loc[x2.index] = 0
        return res

    @substitute
    def ts_argmax_EVO(self, x: Any, y: Any) -> pd.Series(float):
        res = pd.Series(0, index=self.index)
        x, y, flag = checkDim(x, y, para=-1, dim=len(self.index))
        if flag:
            para = round(abs(y))
            if para >= 5:
                res = pd.Series(para - bk.move_argmax(x, para, min_count=5), index=self.index)
                x2 = res.groupby('code').head(para - 1)
                res.loc[x2.index] = 0
        return res

    @substitute
    def ts_argmin_EVO(self, x: Any, y: Any) -> pd.Series(float):
        res = pd.Series(0, index=self.index)
        x, y, flag = checkDim(x, y, para=-1, dim=len(self.index))
        if flag:
            para = round(abs(y))
            if para >= 5:
                res = pd.Series(para - bk.move_argmin(x, para, min_count=5), index=self.index)
                x2 = res.groupby('code').head(para - 1)
                res.loc[x2.index] = 0
        return res

    # @substitute
    # def ts_argmaxmin_EVO(self, x: Any, y: Any) -> pd.Series(float):
    #     res = pd.Series(0, index=self.index)
    #     x, y, flag = checkDim(x, y, para=-1, dim=len(self.index))
    #     if flag:
    #         para = round(abs(y))
    #         if para >= 5:
    #             res = pd.Series(bk.move_argmax(x, para, min_count=5) - bk.move_argmin(x, para, min_count=5),
    #                             index=self.index)
    #             x2 = res.groupby('code').head(para - 1)
    #             res.loc[x2.index] = 0
    #     return res

    @substitute
    def ts_corrP_EVO(self, data1: Any, data2: Any, data3: Any) -> pd.Series(float):
        """
        pearson相关系数
        """
        res = pd.Series(0, index=self.index)
        data1, data2, data3, flag = checkDim(data1, data2, data3, para=-1, dim=len(self.index))
        if flag:
            para = round(abs(data3))
            if para >= 5:
                res = pd.Series(data1).rolling(para, min_periods=1).corr(pd.Series(data2)).fillna(0)
                x2 = res.groupby('code').head(para - 1)
                res.loc[x2.index] = 0
                res[np.isinf(res)] = 0
        return res

    # # @substitute
    # # def ts_corrS_EVO(self, data1: Any, data2: Any, data3: Any):
    # #     """
    # #     秩相关系数
    # #     """
    # #     res = np.array([0] * self.dim)
    # #     data1, data2, data3, flag = checkDim(data1, data2, data3, para=-1, dim=self.dim)
    # #     if flag:
    # #         para = round(abs(data3))
    # #         if para >= 5:
    # #             sub1, sub2 = rolling_window(data1, para), rolling_window(data2, para)
    # #             sub1, sub2 = sub1.argsort(), sub2.argsort()
    # #             corSub = np.diag(np.corrcoef(sub1, sub2)[:sub1.shape[0], -sub1.shape[0]:])
    # #             res = np.concatenate((np.array([0] * (para - 1)), np.where(np.isnan(corSub), 0, corSub)))
    # #     return res

    @substitute
    def ts_stand01_EVO(self, x: Any, y: Any) -> pd.Series(float):
        """
        常数标准化为0
        """
        res = pd.Series(0, index=self.index)
        x, y, flag = checkDim(x, y, para=-1, dim=len(self.index))
        if flag:
            para = round(abs(y))
            if para >= 5:
                xMax = bk.move_max(x, para, min_count=1)
                xMin = bk.move_min(x, para, min_count=1)
                denominator = xMax - xMin
                with np.errstate(divide='ignore', invalid='ignore'):
                    res = pd.Series(np.where(denominator > SMALL_VALUE, (x - xMin) / denominator, 0), index=self.index)

                x2 = res.groupby('code').head(para - 1)
                res.loc[x2.index] = 0
        return res

    @substitute
    def ts_standNorm_EVO(self, x: Any, y: Any) -> pd.Series(float):
        """
        常数标准化为0
        """
        res = pd.Series(0, index=self.index)
        x, y, flag = checkDim(x, y, para=-1, dim=len(self.index))
        if flag:
            para = round(abs(y))
            if para >= 5:
                xMean = bk.move_mean(x, para, min_count=1)
                xStd = bk.move_std(x, para, min_count=1)
                with np.errstate(divide='ignore', invalid='ignore'):
                    res = pd.Series(np.where(xStd > SMALL_VALUE, (x - xMean) / xStd, 0), index=self.index)

                x2 = res.groupby('code').head(para - 1)
                res.loc[x2.index] = 0

        return res

    @substitute
    def ts_reg_EVO(self, data1: Any, data2: Any, data3: Any) -> pd.Series(float):
        res = pd.Series(0, index=self.index)
        data1, data2, data3, flag = checkDim(data1, data2, data3, para=-1, dim=len(self.index))

        if flag:
            para = round(abs(data3))
            if para >= 5:
                data = data1 * data2
                data1Mean = bk.move_mean(data1, para, min_count=1)
                data2Mean = bk.move_mean(data2, para, min_count=1)
                dataMean = bk.move_mean(data, para, min_count=1)
                denominator = bk.move_var(data1, para, min_count=1)
                with np.errstate(divide='ignore', invalid='ignore'):
                    res = pd.Series(np.where(denominator > SMALL_VALUE, (dataMean - data1Mean * data2Mean) / denominator, 0),
                                    index=self.index)
                x2 = res.groupby('code').head(para - 1)
                res.loc[x2.index] = 0
        return res

    """截面函数"""
    @substitute
    def cs_rank_EVO(self, data: Any) -> pd.Series(float):
        data, = checkDim(data, dim=len(self.index))
        return pd.Series(data, index=self.index).groupby('date').rank()  # 升序排列

    @substitute
    def cs_scale_EVO(self, data: Any) -> pd.Series(float):
        data, = checkDim(data, dim=len(self.index))
        data = pd.Series(data, index=self.index)
        return data.div(abs(data).groupby('date').sum())

    @substitute
    def cs_demean_EVO(self, data: Any) -> pd.Series(float):
        data, = checkDim(data, dim=len(self.index))
        data = pd.Series(data, index=self.index)
        return data.sub(data.groupby('date').mean())


class GPFunction(EvolutionOperator):
    pass


class GAFunction(EvolutionOperator):
    pass


if __name__ == '__main__':
    GEP()
