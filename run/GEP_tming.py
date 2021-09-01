# -*-coding:utf-8-*-
# @Time:   2021/7/16 14:52
# @Author: FC
# @Email:  18817289038@163.com

import os
import time
import numpy as np
import geppy as gep
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from typing import Any, Tuple
from collections import defaultdict
from utility.utility import checkDim
from scipy.stats import ttest_ind, hmean

from base_class.template import GEP
from data_API.sample_timing import select_sample


sns.set(font='SimHei', palette="muted", color_codes=True)
sns.set_style("darkgrid", {"font.sans-serif": ['simhei', 'Droid Sans Fallback']})


class GEPTiming(GEP):

    def __init__(self, opt):
        super().__init__(opt)
        self.the = 0
        self.rf_a = 0.04  # 年化无风险收益率
        self.fee = 0.002  # 日均手续费
        self.penal = 0.3  # 罚阈值
        self.res = defaultdict(list)
        self.paramL = 400  # 用来计算参数的数据长度

    def primitive(self):
        super().primitive()

    def constant(self):
        super().constant()

    @staticmethod
    def sum_gen(*args):
        return sum(args)

    def evolution(self,
                  individual: gep.Chromosome,
                  **kwargs) -> Tuple[Any, Any]:
        """
        将信号权重向非信号权重调整
        :param individual:
        :param kwargs:
        :return:
        """

        # 信号
        self.Func.dim = self.dataSet['X'].X.shape[0]
        funcL = self.TB.compile(individual)
        yAllPre = np.array(funcL(*tuple(self.self.dataSet['X'].X.T)))
        yAllPre = checkDim(yAllPre, dim=self.self.dataSet['X'].sampleL)[0]

        # 收益率的合成
        ySignAll = np.where(yAllPre > self.the, 1, 0)
        # yRetAll = self.ret_synthesis(self.self.dataSet['X'].Y, ySignAll)

        # 收益率标签
        yRetTrain1 = self.dataSet['X'].Y[np.argwhere((self.dataSet['X'].Label == 0) & (ySignAll == 1)).flatten()]
        yRetTrain0 = self.dataSet['X'].Y[np.argwhere((self.dataSet['X'].Label == 0) & (ySignAll == 0)).flatten()]

        yRetVerify1 = self.dataSet['X'].Y[np.argwhere((self.dataSet['X'].Label == 1) & (ySignAll == 1)).flatten()]
        yRetVerify0 = self.dataSet['X'].Y[np.argwhere((self.dataSet['X'].Label == 1) & (ySignAll == 0)).flatten()]

        yRetVerify21 = self.dataSet['X'].Y[np.argwhere((self.dataSet['X'].Label == 2) & (ySignAll == 1)).flatten()]
        yRetVerify20 = self.dataSet['X'].Y[np.argwhere((self.dataSet['X'].Label == 2) & (ySignAll == 0)).flatten()]

        yRetTest1 = self.dataSet['X'].Y[np.argwhere((self.dataSet['X'].Label == 3) & (ySignAll == 1)).flatten()]
        yRetTest0 = self.dataSet['X'].Y[np.argwhere((self.dataSet['X'].Label == 3) & (ySignAll == 0)).flatten()]

        # 01标签
        flag0Train = ySignAll[np.argwhere((self.dataSet['X'].Label == 0) & (self.dataSet['X'].sign == 0)).flatten()]
        flag1Train = ySignAll[np.argwhere((self.dataSet['X'].Label == 0) & (self.dataSet['X'].sign == 1)).flatten()]

        flag0Verify = ySignAll[np.argwhere((self.dataSet['X'].Label == 1) & (self.dataSet['X'].sign == 0)).flatten()]
        flag1Verify = ySignAll[np.argwhere((self.dataSet['X'].Label == 1) & (self.dataSet['X'].sign == 1)).flatten()]

        flag0Verify2 = ySignAll[np.argwhere((self.dataSet['X'].Label == 2) & (self.dataSet['X'].sign == 0)).flatten()]
        flag1Verify2 = ySignAll[np.argwhere((self.dataSet['X'].Label == 2) & (self.dataSet['X'].sign == 1)).flatten()]

        flag0Test = ySignAll[np.argwhere((self.dataSet['X'].Label == 3) & (self.dataSet['X'].sign == 0)).flatten()]
        flag1Test = ySignAll[np.argwhere((self.dataSet['X'].Label == 3) & (self.dataSet['X'].sign == 1)).flatten()]

        # mean, std
        meanTrain1 = 0 if len(yRetTrain1) < 3 else np.mean(yRetTrain1)
        stdTrain1 = 0 if len(yRetTrain1) < 3 else np.std(yRetTrain1)
        meanTrain0 = 0 if len(yRetTrain0) < 3 else np.mean(yRetTrain0)
        stdTrain0 = 0 if len(yRetTrain0) < 3 else np.std(yRetTrain0)

        meanVerify11 = 0 if len(yRetVerify1) < 3 else np.mean(yRetVerify1)
        stdVerify11 = 0 if len(yRetVerify1) < 3 else np.std(yRetVerify1)
        meanVerify10 = 0 if len(yRetVerify0) < 3 else np.mean(yRetVerify0)
        stdVerify10 = 0 if len(yRetVerify0) < 3 else np.std(yRetVerify0)

        meanVerify21 = 0 if len(yRetVerify21) < 3 else np.mean(yRetVerify21)
        stdVerify21 = 0 if len(yRetVerify21) < 3 else np.std(yRetVerify21)
        meanVerify20 = 0 if len(yRetVerify20) < 3 else np.mean(yRetVerify20)
        stdVerify20 = 0 if len(yRetVerify20) < 3 else np.std(yRetVerify20)

        meanTest1 = 0 if len(yRetTest1) < 3 else np.mean(yRetTest1)
        stdTest1 = 0 if len(yRetTest1) < 3 else np.std(yRetTest1)
        meanTest0 = 0 if len(yRetTest0) < 3 else np.mean(yRetTest0)
        stdTest0 = 0 if len(yRetTest0) < 3 else np.std(yRetTest0)

        # 适应度计算
        fitnessTrain = self.tTest(yRetTrain1, yRetTrain0, 0.05 * self.dataSet['X'].trainL)
        # sampleTrain = (self.dataSet['X'].trainL - self.dataSet['X'].signTrainNum) * 2
        # correctTrain = sum(flag1Train) * self.dataSet['X'].ratio + len(flag0Train) - sum(flag0Train)
        # individual.train = correctTrain / sampleTrain
        individual.trainInfo = len(yRetTrain1), len(yRetTrain0), meanTrain1, stdTrain1, meanTrain0, stdTrain0

        # 验证集
        individual.verify.values = self.tTest(yRetVerify1, yRetVerify0, 0.05 * self.dataSet['X'].verifyL),
        individual.verify2.values = self.tTest(yRetVerify21, yRetVerify20, 0.05 * self.dataSet['X'].verify2L),
        # correctVerify = sum(flag1Verify) * self.dataSet['X'].ratio + len(flag0Verify) - sum(flag0Verify)
        # individual.verify.values = correctVerify / sampleVerify,

        individual.verifyInfo = len(yRetVerify1), len(yRetVerify0), meanVerify11, stdVerify11, meanVerify10, stdVerify10
        individual.verify2Info = len(yRetVerify21), len(yRetVerify20), meanVerify21, stdVerify21, meanVerify20, stdVerify20

        # 测试集
        sampleTest = self.dataSet['X'].signTestNum * self.dataSet['X'].ratio + self.dataSet['X'].testL - self.dataSet['X'].signTestNum
        correctTest = sum(flag1Test) * self.dataSet['X'].ratio + len(flag0Test) - sum(flag0Test)
        individual.test.values = correctTest / sampleTest,
        individual.test2.values = self.tTest(yRetTest1, yRetTest0, 3),
        individual.testInfo = len(yRetTest1), len(yRetTest0), meanTest1, stdTest1, meanTest0, stdTest0

        fitnessTrain = 0 if np.isnan(fitnessTrain) else fitnessTrain

        return fitnessTrain,

    @staticmethod
    def tTest(x1, x2, the):
        if len(x1) < the or len(x2) < the * 4:
            return 0
        # return np.mean(x1) / np.std(x1)
        return ttest_ind(x1, x2, equal_var=True).statistic
        # return np.mean(x1) / np.std(x1)
        # sampleMean = np.mean(x1) - np.mean(x2)
        # sampleStd = np.sqrt((np.var(x1) * (len(x1) - 1) + np.var(x2) * (len(x2) - 1)) / (len(x1) + len(x2) - 2))
        # return sampleMean / sampleStd

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
        trainBM = self.dataSet['X'].Y[np.argwhere(self.dataSet['X'].Label == 0).flatten()]
        verifyBM = self.dataSet['X'].Y[np.argwhere(self.dataSet['X'].Label == 1).flatten()]
        testBM = self.dataSet['X'].Y[np.argwhere(self.dataSet['X'].Label == 2).flatten()]

        trainSR = ((trainBM + 1).prod() ** (240 / self.dataSet['X'].trainL) - 1 - self.rf_a) / np.std(trainBM) / np.sqrt(240)
        verifySR = ((verifyBM + 1).prod() ** (240 / self.dataSet['X'].verifyL) - 1 - self.rf_a) / np.std(verifyBM) / np.sqrt(
            240)
        testSR = ((testBM + 1).prod() ** (240 / self.dataSet['X'].testL) - 1 - self.rf_a) / np.std(testBM) / np.sqrt(240)

        self.res['BM'] = pd.Series([trainSR, verifySR, testSR], index=['trainSR', 'verifySR', 'testSR'])
        #
        # trainBM1 = self.dataSet['X'].Y[np.argwhere((self.dataSet['X'].Label == 0) & (self.dataSet['X'].sign != 0)).flatten()]
        # trainBMAll = self.dataSet['X'].Y[np.argwhere(self.dataSet['X'].Label == 0).flatten()]
        #
        # verifyBM1 = self.dataSet['X'].Y[np.argwhere((self.dataSet['X'].Label == 1) & (self.dataSet['X'].sign != 0)).flatten()]
        # verifyBMAll = self.dataSet['X'].Y[np.argwhere(self.dataSet['X'].Label == 1).flatten()]
        #
        # testBM1 = self.dataSet['X'].Y[np.argwhere((self.dataSet['X'].Label == 2) & (self.dataSet['X'].sign != 0)).flatten()]
        # testBMAll = self.dataSet['X'].Y[np.argwhere(self.dataSet['X'].Label == 2).flatten()]
        #
        # testRBM1 = self.dataSet['X'].Y[np.argwhere((self.dataSet['X'].Label == 3) & (self.dataSet['X'].sign != 0)).flatten()]
        # testRBMAll = self.dataSet['X'].Y[np.argwhere(self.dataSet['X'].Label == 3).flatten()]
        #
        # trainSR = (np.mean(trainBM1) - np.mean(trainBMAll)) / np.std(trainBM1)
        # verifySR = (np.mean(verifyBM1) - np.mean(verifyBMAll)) / np.std(verifyBM1)
        # testSR = (np.mean(testBM1) - np.mean(testBMAll)) / np.std(testBM1)
        # testRSR = (np.mean(testRBM1) - np.mean(testRBMAll)) / np.std(testRBM1)
        #
        # self.res['BM'] = pd.Series([trainSR, verifySR, testSR, testRSR],
        #                            index=['trainSR', 'verifySR', 'testSR', 'testRSR'])

    def run(self):
        # 初始化
        self.primitive()
        self.constant()
        self.genetic()
        self.TB.register('evaluate', self.evolution)
        self.statistic()

        # benchmark: 根据benchmark设定惩罚系数，注意惩罚系数的符号问题！
        # self.benchmark()
        # self.penal = min(self.penal, self.res['BM']['trainSR'], self.res['BM']['verifySR']) + self.penal

        # 进化
        self.competition(self.sum_gen)

        # 模型性能记录
        self.model_performance()

    # 模型性能和公式记录
    def model_performance(self):
        self.Func.dim = self.dataSet['X'].X.shape[0]

        formula = []
        for gen, forms in self.bestInd.inds.items():
            for form in forms:
                formula.append(pd.Series([gen,
                                          form.fitness.values[0],
                                          form.verify.values[0],
                                          form.verify2.values[0],

                                          form.test.values[0],
                                          form.test2.values[0],

                                          form.trainInfo,
                                          form.verifyInfo,
                                          form.verify2Info,
                                          form.testInfo,
                                          form.__str__()],

                                         index=['gen', 'train', 'verify', 'verify2', 'test', 'test2',
                                                'trainInfo', 'verifyInfo', 'verify2Info', 'testInfo', 'formula']))
        formula = pd.concat(formula, axis=1).T
        formula['verifyF1'], formula['verifyF2'] = 0, 0

        formulaSub1 = formula[(formula['verify'] > 0) & (formula['verify2'] > 0)]
        formulaSub2 = formula[(formula['verify'] > 0) & (formula['verify2'] > 0) & (formula['train'] > 0)]
        formula.loc[formulaSub1.index, 'verifyF1'] = hmean(formulaSub1[['verify', 'verify2']], axis=1)
        formula.loc[formulaSub2.index, 'verifyF2'] = hmean(formulaSub2[['train', 'verify', 'verify2']], axis=1)

        self.res['formula'] = formula

    def ret_synthesis(self, ret: np.array, label: np.array) -> np.array:
        """
        资金无限：当天持仓情况下开仓，每个仓位资金都相同
        """
        labelZero = np.array([0] * len(label))
        rets = []
        for i in range(self.hp):
            labelSub = labelZero.copy()
            fee = self.fee if i == 0 or i == self.hp - 1 else 0
            labelSub[i:] = label[:len(label) - i]
            rets.append(np.where(labelSub == 1, (1 - fee) * ret - fee, 0))

        retSyn = np.mean(np.array(rets), axis=0)
        return retSyn


def main():
    for ratio in [0]:  # [0.01, 0.008, 0.006, 0.004, 0.002, 0.001]:
        funcName = 'index300'
        # sampleOdds = (3, 1, 1)
        path = os.path.join(r'A:\Work\Working\13.基于机器学习的因子挖掘\相关实验\21.自适应概率')
        if not os.path.exists(path):
            os.mkdir(path)
        for iterN in range(200, 230):
            sampleClass = select_sample(funcName, seed=iterN, ratio=ratio)
            # model parameter
            model_params = {
                "path": path,
                "dataSet": {"X": sampleClass},
                "funcName": funcName,

                "rf_a": 0.04,
                "hp": 1,  # 持有周期
                "fee": 0.001,  # 手续费
                "h": 6,  # 基因头部长度
                "n_genes": 2,  # 染色体基因个数
                "n_elites": 50,
                "bestNum": 4,  # 最优个体数量
                "const": "ENC",
                "penal": -1,
                "n_pop": 500,  # 种群规模
                "n_gen": 50,  # 进化代数
            }
            print(f"{funcName}, iter: {iterN}, Head:{model_params['h']}, Num Gen:{model_params['n_genes']}")
            A = GEPTiming(opt=1)
            A.set_params(**model_params)
            A.run()

            # Write = pd.ExcelWriter(os.path.join(model_params['path'], f'GEP_iter{iterN}.xlsx'))
            A.res['formula'].to_csv(os.path.join(model_params['path'], f'formula{iterN}.csv'))
            pd.DataFrame(A.logs).to_csv(os.path.join(model_params['path'], f'iteration{iterN}.csv'))
            # Write.save()
    # # 策略总净值曲线
    # indexData = sampleClass.dataRaw.copy(deep=True)
    # indexData['indexRet'] = indexData['close'].pct_change(fill_method=None).fillna(0)
    # ret = pd.concat([indexData[['date', 'indexRet']],
    #                  A.res['retT'],
    #                  pd.Series(sampleClass.Label, name='label')],
    #                 axis=1).set_index('date')
    # ret_new = pd.concat([ret[ret['label'] == 0], ret[ret['label'] == 1], ret[ret['label'] == 2]])
    # nav = (ret_new + 1).cumprod()
    # ex_nav = nav.div(nav['indexRet'], axis=0)
    #
    # #  作图
    # fig = plt.figure(figsize=(20, 12))
    # flag = 1
    # for col_ in A.res['retT'].columns:
    #     ax = fig.add_subplot(4, 4, flag)
    #     nav[['indexRet', col_]].plot(kind='line', ax=ax, legend=False)
    #     ax.axvline(x=sampleClass.trainL, ls="--", c="red")
    #     ax.axvline(x=sampleClass.trainL + sampleClass.verifyL, ls="--", c="red")
    #     ex_nav[col_].plot(kind='line', ax=ax, secondary_y=True, rot=30)
    #     ax.legend().set_visible(False)
    #     plt.title(f"{col_}")
    #     flag += 1
    # title = f"Head{model_params['h']}_N{model_params['n_genes']}_Train"
    # plt.suptitle(title)
    # fig.tight_layout(pad=1, w_pad=1, h_pad=1)
    # plt.savefig(os.path.join(model_params['path'], f"nav_{title}.png"), dpi=100)
    # plt.show()
    #
    # # 策略总净值曲线
    # indexData = sampleClass.dataRaw.copy(deep=True)
    # indexData['indexRet'] = indexData['close'].pct_change(fill_method=None).fillna(0)
    # ret = pd.concat([indexData[['date', 'indexRet']],
    #                  A.res['retV'],
    #                  pd.Series(sampleClass.Label, name='label')],
    #                 axis=1).set_index('date')
    # ret_new = pd.concat([ret[ret['label'] == 0], ret[ret['label'] == 1], ret[ret['label'] == 2]])
    # nav = (ret_new + 1).cumprod()
    # ex_nav = nav.div(nav['indexRet'], axis=0)
    #
    # fig = plt.figure(figsize=(20, 12))
    # flag = 1
    # for col_ in A.res['retV'].columns:
    #     ax = fig.add_subplot(4, 4, flag)
    #     nav[['indexRet', col_]].plot(kind='line', ax=ax, legend=False)
    #     ax.axvline(x=sampleClass.trainL, ls="--", c="red")
    #     ax.axvline(x=sampleClass.trainL + sampleClass.verifyL, ls="--", c="red")
    #     ex_nav[col_].plot(kind='line', ax=ax, secondary_y=True, rot=30)
    #     ax.legend().set_visible(False)
    #     plt.title(f"{col_}")
    #     flag += 1
    # title = f"Head{model_params['h']}_N{model_params['n_genes']}_Verify"
    # plt.suptitle(title)
    # fig.tight_layout(pad=1, w_pad=1, h_pad=1)
    # plt.savefig(os.path.join(model_params['path'], f"nav_{title}.png"), dpi=100)
    # plt.show()


if __name__ == '__main__':
    main()
