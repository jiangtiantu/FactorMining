# -*-coding:utf-8-*-
# @Time:   2021/8/27 11:29
# @Author: FC
# @Email:  18817289038@163.com


import os
import time
import numpy as np
import geppy as gep
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.stats import ttest_ind, hmean
from typing import Any, Tuple, Dict
from utility.utility import checkDim, timer
from collections import defaultdict

from utility.object import DataInfo
from utility.operator_func import GEPFunctionSelecting
from base_class.template import GEP
from utility.stock_utility import MethodSets, holding_ret, weight_cor
from data_API.sample_stock import StockGateWay

sns.set(font='SimHei', palette="muted", color_codes=True)
sns.set_style("darkgrid", {"font.sans-serif": ['simhei', 'Droid Sans Fallback']})

API = StockGateWay()


class GEPSelecting(GEP, MethodSets):
    Factor = 'GEP'

    def __init__(self, opt):
        super().__init__(opt)
        super(GEP, self).__init__()

        self.Func = GEPFunctionSelecting()
        self.the = 0
        self.rf_a = 0.04  # 年化无风险收益率
        self.hp = 1  # 持有期
        self.res = defaultdict(list)

    def primitive(self):
        super().primitive()

    def constant(self):
        super().constant()

    @staticmethod
    def sum_gen(*args):
        return sum(args)

    # @timer
    def integration(self, factor: pd.Series) -> pd.Series:

        SP, LP = self.dataSet["stockPool"].data, self.dataSet["labelPool"].data
        ExpCleanNeu = pd.merge(factor.reindex(SP.index),
                               LP,
                               left_index=True,
                               right_index=True,
                               how='left')
        # 中性化
        self.processSeq(ExpCleanNeu, methodN=['RO', 'Neu', 'Sta'], dataName=factor.name)
        ExpCleanNeu.dropna(subset=[factor.name], inplace=True)
        return ExpCleanNeu

    @timer
    def evolution(self,
                  individual: gep.Chromosome,
                  **kwargs) -> Tuple[Any, Any]:
        """
        将信号权重向非信号权重调整
        :param individual:
        :param kwargs:
        :return:
        """

        # 因子计算
        funcL = self.TB.compile(individual)
        factor = funcL(*self.dataSet["X"].X.to_dict('series').values())  # series
        factor = pd.Series(checkDim(factor, dim=len(self.dataSet['X'].X.index))[0],
                           index=self.dataSet['X'].X.index,
                           name=self.Factor)

        if len(factor.unique()) == 1:
            fitness = 0,
            individual.verify.values = 0,
            individual.verify2.values = 0,
            individual.test.values = 0,

            individual.trainInfo = 0
            individual.verifyInfo = 0
            individual.verify2Info = 0
            individual.testInfo = 0
        else:
            # 数据整合:去极值，中性化，标准化
            factorNew = self.integration(factor)

            # 评估
            res = self.effectiveness(factorNew)

            # 样本切分
            dataTrain = res[self.dataSet['X'].Label == 0]
            dataVerify = res[self.dataSet['X'].Label == 1]
            dataVerify2 = res[self.dataSet['X'].Label == 2]
            dataTest = res[self.dataSet['X'].Label == 3]

            individual.verify.values = abs(dataVerify.mean() / dataVerify.std()),
            individual.verify2.values = abs(dataVerify2.mean() / dataVerify2.std()),
            individual.test.values = abs(dataTest.mean() / dataTest.std()),

            individual.trainInfo = dataTrain.mean()
            individual.verifyInfo = dataVerify.mean()
            individual.verify2Info = dataVerify2.mean()
            individual.testInfo = dataTest.mean()

            # 适应度计算
            fitness = abs(dataTrain.mean() / dataTrain.std()),
        return fitness

    # @timer
    def effectiveness(self, factor: pd.DataFrame) -> pd.Series(float):
        IC_rank = factor.groupby('date').apply(lambda x: weight_cor(x[[self.Factor, 'retOpen']].rank(), x['stockWeight']))
        return IC_rank

    def penalty_func(self, value: float) -> float:
        """
        罚函数
        :param value:
        :return:
        """
        return self.penal if value < self.penal else value

    def genetic(self):
        # 遗传
        super().genetic()

    def statistic(self):
        super().statistic()

        # self.stats.register(name="min_verify", function=np.min, dataType='oth1')
        self.stats.register(name="max_verify", function=np.max, dataType='oth1')
        # self.stats.register(name="med_verify", function=np.median, dataType='oth1')
        self.stats.register(name="avg_verify", function=np.mean, dataType='oth1')

        # self.stats.register(name="min_verify2", function=np.min, dataType='oth2')
        self.stats.register(name="max_verify2", function=np.max, dataType='oth2')
        # self.stats.register(name="med_verify2", function=np.median, dataType='oth2')
        self.stats.register(name="avg_verify2", function=np.mean, dataType='oth2')

    def benchmark(self):
        trainBM = self.dataSet["dataIn"].Y[np.argwhere(self.dataSet["dataIn"].Label == 0).flatten()]
        verifyBM = self.dataSet["dataIn"].Y[np.argwhere(self.dataSet["dataIn"].Label == 1).flatten()]
        testBM = self.dataSet["dataIn"].Y[np.argwhere(self.dataSet["dataIn"].Label == 2).flatten()]

        trainSR = ((trainBM + 1).prod() ** (240 / self.dataSet["dataIn"].trainL) - 1 - self.rf_a) / np.std(
            trainBM) / np.sqrt(240)
        verifySR = ((verifyBM + 1).prod() ** (240 / self.dataSet["dataIn"].verifyL) - 1 - self.rf_a) / np.std(
            verifyBM) / np.sqrt(
            240)
        testSR = ((testBM + 1).prod() ** (240 / self.dataSet["dataIn"].testL) - 1 - self.rf_a) / np.std(
            testBM) / np.sqrt(240)

        self.res['BM'] = pd.Series([trainSR, verifySR, testSR], index=['trainSR', 'verifySR', 'testSR'])

    def run(self):
        # 初始化
        self.primitive()
        self.constant()
        self.genetic()
        self.TB.register('evaluate', self.evolution)
        self.statistic()

        # func index
        self.Func.index = self.dataSet['X'].X.index
        # 进化
        self.competition(self.sum_gen)

        # 模型性能记录
        self.model_performance()

    # 模型性能和公式记录
    def model_performance(self):

        formula = []
        for gen, forms in self.bestInd.inds.items():
            for form in forms:
                formula.append(pd.Series([gen,
                                          form.fitness.values[0],
                                          form.verify.values[0],
                                          form.verify2.values[0],
                                          form.test.values[0],

                                          form.trainInfo,
                                          form.verifyInfo,
                                          form.verify2Info,
                                          form.testInfo,
                                          form.__str__()],

                                         index=['gen', 'train', 'verify', 'verify2', 'test',
                                                'trainInfo', 'verifyInfo', 'verify2Info', 'testInfo', 'formula']))
        formula = pd.concat(formula, axis=1).T
        formula['verifyF1'], formula['verifyF2'] = 0, 0

        formulaSub1 = formula[(formula['verify'] > 0) & (formula['verify2'] > 0)]
        formulaSub2 = formula[(formula['verify'] > 0) & (formula['verify2'] > 0) & (formula['train'] > 0)]
        formula.loc[formulaSub1.index, 'verifyF1'] = hmean(formulaSub1[['verify', 'verify2']], axis=1)
        formula.loc[formulaSub2.index, 'verifyF2'] = hmean(formulaSub2[['train', 'verify', 'verify2']], axis=1)

        self.res['formula'] = formula


def data_input(seed: int = 1, hp: int = 1) -> Dict[str, Any]:
    dataDict = {
        "X": API.get_data('stock_feature', seed=seed),
        "stockPool": API.get_data('stock_pool', seed=seed),
        "labelPool": API.get_data('label_pool', seed=seed),
        "benchmark": API.get_data('stock_benchmark', seed=seed),
        # "styleFactor": API.get_data('stock_factor', seed=seed),
    }
    dataDict['Y'] = DataInfo(data=holding_ret(dataDict['labelPool'].data['retOpen'], hp), dataType='retOpen')

    return dataDict


def main(iterN):
    hp = 1
    path = os.path.join(r'A:\Work\Working\13.基于机器学习的因子挖掘\测试')
    if not os.path.exists(path):
        os.mkdir(path)
    # model parameter
    model_params = {
        "path": path,
        "dataSet": data_input(iterN, hp),

        "methodProcess": {
            "RO": {"method": "mad", "p": {}},  #
            "Neu": {"method": "industry+mv", "p": {"mvName": "liqMv", "indName": "indexCode"}},  #
            "Sta": {"method": "z_score", "p": {}}  #
        },

        "rf_a": 0.04,
        "hp": hp,  # 持有周期
        "fee": 0.001,  # 手续费
        "h": 6,  # 基因头部长度
        "n_genes": 2,  # 染色体基因个数
        "n_elites": 50,
        "bestNum": 10,  # 最优个体数量
        "const": "ENC",
        "penal": -1,
        "n_pop": 500,  # 种群规模
        "n_gen": 40,  # 进化代数
    }

    print(f"iter: {iterN}, Head:{model_params['h']}, Num Gen:{model_params['n_genes']}")
    A = GEPSelecting(opt=1)
    A.set_params(**model_params)
    A.run()

    A.res['formula'].to_csv(os.path.join(model_params['path'], f'formula{iterN}.csv'))
    pd.DataFrame(A.logs).to_csv(os.path.join(model_params['path'], f'iteration{iterN}.csv'))


if __name__ == '__main__':
    seed = int(input())
    main(seed)
