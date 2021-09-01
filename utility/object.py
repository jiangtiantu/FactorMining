# -*-coding:utf-8-*-
# @Time:   2021/7/9 10:53
# @Author: FC
# @Email:  18817289038@163.com

import numpy as np
import pandas as pd
from typing import Callable, Union
from dataclasses import dataclass


# 算子属性类
class FuncAtt:
    def __init__(self, funcName: str, funcMethod: Callable, arity: int):
        self.funcName = funcName
        self.funcMethod = funcMethod
        self.arity = arity


#  样本属性类
@dataclass
class Classification(object):
    X: np.array  # 总样本特征
    Y: np.array  # 总样本标签(原始标签)

    YSign: np.array = None  # 总样本标签(加入信号后的标签)
    sign: np.array = None  # 信号(真实信号)

    noiseR: float = None  # 噪音占比
    noiseMean: float = None  # 噪音均值
    noiseVar: float = None  # 噪音方差

    signNum: int = None  # 总信号个数
    signTrainNum: int = None  # 训练集信号个数
    signVerifyNum: int = None  # 验证集信号个数
    signVerify2Num: int = None  # 验证集信号个数
    signTestNum: int = None  # 测试集信号个数

    signTestR: int = None

    Label: np.array = None  # 样本划分标记

    def __post_init__(self):
        self.sampleL = self.X.shape[0]
        self.trainL = list(self.Label).count(0)
        self.verifyL = list(self.Label).count(1)
        self.verify2L = list(self.Label).count(2)
        self.testL = list(self.Label).count(3)

        self.ratio = 1 if self.signTrainNum == 0 else self.trainL / self.signTrainNum - 1


@dataclass
class Regression(object):
    """
    默认训练集标记为0，验证集标记为1，测试集标记为2
    """
    dataRaw: Union[pd.DataFrame, np.array]  # 原始数据

    X: Union[pd.DataFrame, np.array]  # 总样本特征
    Y: Union[pd.DataFrame, np.array]  # 总样本标签(原始标签)

    sign: Union[pd.DataFrame, np.array] = None

    Label: Union[pd.DataFrame, np.array] = None  # 样本划分标记


@dataclass
class DataInfo(object):
    data: Union[pd.DataFrame, np.array]  # 原始数据
    dataType: str = None
