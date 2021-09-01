# -*-coding:utf-8-*-
# @Time:   2021/7/9 10:54
# @Author: FC
# @Email:  18817289038@163.com

import os
import time
import scipy
import bisect
import numpy as np
import numba as nb
import pandas as pd
import seaborn as sns
import datetime as dt
import pickle5 as pickle
import matplotlib.pyplot as plt

from copy import deepcopy
from functools import wraps
from typing import Any, Union, List, Tuple
from numpy.lib.stride_tricks import sliding_window_view

SMALL_VALUE = 1e-12  # 亿分之一以下定义为0


# 维度和参数的检查
def checkDim(*args, para: int = 0, dim: int) -> List[Any]:
    if para != 0:
        flag = 1
        args1 = tuple(np.repeat(i, dim) if isinstance(i, (float, int)) or not i.shape else i for i in args[:para])
        for arg in args[para:]:
            if not isinstance(arg, (float, int)):
                flag = 0
                break
        args = list(args1) + list(args[para:])
        return args + [flag]
    else:
        args = tuple(np.repeat(i, dim) if isinstance(i, (float, int)) or not i.shape else i for i in args)

        return list(args)


def norm_to_uniform(x: Union[np.array, pd.Series], mappingRange: List[Any]) -> Any:
    rangeUniform = [i / len(mappingRange) for i in range(len(mappingRange) + 1)]
    if np.diff(x).std() < SMALL_VALUE:
        y = 1
    else:
        y = scipy.stats.norm.cdf(np.diff(x).mean() / np.diff(x).std())  # 差分标准化
    rangeIndex = bisect.bisect_right(rangeUniform, y, hi=len(mappingRange))
    return mappingRange[rangeIndex - 1]


def uniform_to_uniform(x: Union[float, int],
                       x_range: Tuple[Union[float, int], Union[float, int]],
                       mappingRange: List[Any]) -> Any:
    rangeUniform = [i / len(mappingRange) for i in range(len(mappingRange) + 1)]
    y = x / (x_range[-1] - x_range[0]) + (0.5 - np.mean(x_range))  # 放缩再做中心移动
    rangeIndex = bisect.bisect_right(rangeUniform, y, hi=len(mappingRange))
    return mappingRange[rangeIndex - 1]


# 深度复制数据
def substitute(func):
    @wraps(func)  # 不改变函数属性
    def wrapper(*args, **kwargs):
        res = func(*deepcopy(args), **deepcopy(kwargs))
        return res

    return wrapper


def timer(func):
    def wrapper(*args, **kwargs):
        func_name = func.__name__

        sta = time.time()

        res = func(*args, **kwargs)

        rang_time = round((time.time() - sta) / 60, 4)

        print(f"\033[1;31m{dt.datetime.now().strftime('%X')}: It takes\033[0m "
              f"\033[1;33m{rang_time:<6}Min\033[0m "
              f"\033[1;31mto run func\033[0m "
              f"\033[1;33m\'{func_name}\'\033[0m")
        return res

    return wrapper


# 滚动窗口构造函数:只支持一维
def rolling_window(data: pd.Series, window: int) -> np.array:
    """
    缺失部分填充nan
    :param data:
    :param window:
    :return:
    """
    resSub = sliding_window_view(data, window)
    res = np.concatenate((np.full((window - 1, window), np.nan), resSub))
    return res


# 数据读取
def readData(path: str, name: str, fileType: str) -> pd.DataFrame:
    filePath = os.path.join(path, name + '.' + fileType)
    if os.path.exists(filePath):
        if fileType == 'csv':
            data = pd.read_csv(filePath)
            return data
        elif fileType == 'pkl':
            with open(filePath, 'rb') as f:
                data = pickle.load(f)
            return data


# 分层抽样
def stratified_sampling(length: int, odds: Tuple, seed: int) -> np.array:
    """
    分层抽样
    :param length:
    :param odds:
    :param seed:
    :return:
    """
    np.random.seed(seed)
    if len(odds) == 2:
        labels = (np.resize(0, odds[0]), np.resize(1, odds[1]))
    elif len(odds) == 3:
        label1 = np.random.randint(1, odds[0])
        labels = (np.resize(0, label1), np.resize(1, odds[1]), np.resize(0, odds[0] - label1), np.resize(2, odds[2]))
    else:
        labels = ([0],)
        print("only one sub-sample: len(odds)==1")
    return np.resize(np.concatenate(labels), length)


if __name__ == '__main__':
    p = rolling_window(pd.Series(np.array([1, 2, 3, 4,5,6,7,8])), 5)
