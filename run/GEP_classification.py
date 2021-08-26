# -*-coding:utf-8-*-
# @Time:   2021/7/16 14:52
# @Author: FC
# @Email:  18817289038@163.com

import os
import numpy as np
import geppy as gep
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn.model_selection as ms

from copy import deepcopy
from typing import Any, Tuple
from utility.utility import checkDim
from collections import defaultdict
from sklearn.metrics import r2_score, f1_score

from base_class.template import GEP
from database.sample import select_sample


class GEPClassification(GEP):

    def __init__(self, opt):
        super().__init__(opt)

        self.the = 0  # 分类阈值
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
        self.Func.dim = self.data.X.shape[0]
        funcL = self.TB.compile(individual)
        yAllPre = np.array(funcL(*tuple(self.data.X.T)))
        yAllPre = checkDim(yAllPre, dim=self.data.sampleL)[0]
        ySignAll = np.where(yAllPre > self.the, 1, 0)

        # 样本划分
        ySignTrain = np.where(self.data.Label == 0, ySignAll, -1)
        ySignVerify = np.where(self.data.Label == 1, ySignAll, -1)
        ySignTest = np.where(self.data.Label == 2, ySignAll, -1)

        # 适应度计算
        sampleTrain = (self.data.trainL - self.data.signTrainNum) * 2
        flag0Train = ySignTrain[np.argwhere((self.data.Label == 0) & (self.data.YSign == 0)).flatten()]
        flag1Train = ySignTrain[np.argwhere((self.data.Label == 0) & (self.data.YSign == 1)).flatten()]
        correctTrain = sum(flag1Train) * self.data.ratio + len(flag0Train) - sum(flag0Train)
        fitnessTrain = self.penalty_func(correctTrain / sampleTrain)

        sampleVerify = self.data.signVerifyNum * self.data.ratio + self.data.verifyL - self.data.signVerifyNum
        flag0Verify = ySignVerify[np.argwhere((self.data.Label == 1) & (self.data.YSign == 0)).flatten()]
        flag1Verify = ySignVerify[np.argwhere((self.data.Label == 1) & (self.data.YSign == 1)).flatten()]
        correctVerify = sum(flag1Verify) * self.data.ratio + len(flag0Verify) - sum(flag0Verify)
        individual.verify.values = correctVerify / sampleVerify,

        sampleTest = self.data.signTestNum * self.data.ratio + self.data.testL - self.data.signTestNum
        flag0Test = ySignTest[np.argwhere((self.data.Label == 2) & (self.data.YSign == 0)).flatten()]
        flag1Test = ySignTest[np.argwhere((self.data.Label == 2) & (self.data.YSign == 1)).flatten()]
        correctTest = sum(flag1Test) * self.data.ratio + len(flag0Test) - sum(flag0Test)
        individual.test.values = correctTest / sampleTest,

        sampleTestR = self.data.signTestR * self.data.ratio + self.data.testL - self.data.signTestR
        flag0TestR = ySignTest[np.argwhere((self.data.Label == 2) & (self.data.sign == 0)).flatten()]
        flag1TestR = ySignTest[np.argwhere((self.data.Label == 2) & (self.data.sign == 1)).flatten()]
        correctTestR = sum(flag1TestR) * self.data.ratio + len(flag0TestR) - sum(flag0TestR)
        individual.testR.values = correctTestR / sampleTestR,

        return fitnessTrain,

    def penalty_func(self, value: float) -> float:
        """
        罚函数
        :param value:
        :return:
        """
        return 0. if value < self.penal else value

    def genetic(self):
        # 遗传
        super().genetic()

    def statistic(self):
        super().statistic()

        # self.stats.register(name="min_verify", function=np.min, dataType='oth1')
        self.stats.register(name="max_verify", function=np.max, dataType='oth1')
        # self.stats.register(name="med_verify", function=np.median, dataType='oth1')
        self.stats.register(name="avg_verify", function=np.mean, dataType='oth1')

    def run(self):
        # 初始化
        self.primitive()
        self.constant()
        self.genetic()
        self.TB.register('evaluate', self.evolution, 1)
        self.statistic()

        # 进化
        self.competition(self.sum_gen)

        # 模型性能记录
        self.model_performance()
        # self.MC()

    # 模型性能和公式记录
    def model_performance(self):

        # 记录所有个体
        formula = []
        for gen, forms in self.bestInd.inds.items():
            for form in forms:
                formula.append(pd.Series([gen,
                                          form.fitness.values[0],
                                          form.verify.values[0],
                                          form.test.values[0],
                                          form.testR.values[0],
                                          form.trainNum,
                                          form.__str__()],
                                         index=['gen', 'train', 'verify', 'test', 'testR', 'sign', 'formula']))
        self.res['formula'] = pd.concat(formula, axis=1).T

    def MC(self):
        """
        蒙特卡罗模拟
        """
        MC1, MC2 = defaultdict(list), defaultdict(list)
        for i in range(50):
            sampleSim = select_sample(self.funcName, self.funcNoise, self.seed + 10 + i, self.size,
                                      strategy=self.strategy, verify=self.verify)
            for j in range(self.bestNum):
                ySimPred = gep.compile_(self.bestInd.items2[j], self.gep_p)(*tuple(sampleSim.X.T))
                ySignSim = np.where(ySimPred > self.the, 1, 0)

                sampleN = len(sampleSim.YSign) - sum(sampleSim.YSign) + sum(sampleSim.YSign) * self.data.ratio

                flag0 = ySignSim[np.argwhere(sampleSim.YSign == 0).flatten()]
                flag1 = ySignSim[np.argwhere(sampleSim.YSign == 1).flatten()]
                correctN = sum(flag1) * self.data.ratio + len(flag0) - sum(flag0)  # 训练集参数
                MC1[j].append(round(correctN / sampleN, 4))

                # 剔除噪音
                sample = len(sampleSim.sign) - sum(sampleSim.sign) + sum(sampleSim.sign) * self.data.ratio
                flag0 = ySignSim[np.argwhere(sampleSim.sign == 0).flatten()]
                flag1 = ySignSim[np.argwhere(sampleSim.sign == 1).flatten()]
                correct = sum(flag1) * self.data.ratio + len(flag0) - sum(flag0)  # 训练集参数
                MC2[j].append(round(correct / sample, 4))

        self.res['MC_Noise'] = pd.DataFrame(MC1)
        self.res['MC'] = pd.DataFrame(MC2)


def main():
    res = defaultdict(dict)
    seed = 22
    funcName = 'stock_MC'
    funcNoise = 0.0
    size = 1000
    verify = 0.2
    for strategy in ['BOLL', 'Double MA', 'ATR',
                     'Dual Thrust', 'Fairy Four Price', 'Momentum',
                     'Creation1', 'Creation2', 'Creation3']:
        strategy = 'True Boll'
        sampleClass = select_sample(funcName, funcNoise, seed, size, strategy=strategy, verify=verify)

        print(f"{strategy}: {sum(sampleClass.sign)}")
        # model parameter
        params = {
            "path": r'A:\Work\Working\13.基于机器学习的因子挖掘\算法测试\方程参数自定义',
            "strategy": strategy,
            "data": sampleClass,
            "funcName": funcName,
            "funcNoise": funcNoise,
            "size": sampleClass.sampleL,
            "seed": seed,
            "verify": verify,
            "paramL": 400,
            "h": 5,  # 基因头部长度
            "n_genes": 2,  # 染色体基因个数
            "n_elites": 100,
            "const": "ENC",
            "penal": 0.0,
            "n_pop": 1000,  # 种群规模
            "n_gen": 100,  # 进化代数
        }
        A = GEPClassification(opt=1)
        A.set_params(**params)
        A.run()

        res["fit"][strategy] = pd.DataFrame(A.res['fit'])
        pd.DataFrame(A.logs).to_csv(os.path.join(params['path'], f'GEP_iteration_{strategy}.csv'), index=False)
        A.res['formula'].to_csv(os.path.join(params['path'], f'GEP_formula_{strategy}.csv'), index=False)
        res["MC_Noise"][strategy] = pd.DataFrame(A.res['MC_Noise'])
        res["MC"][strategy] = pd.DataFrame(A.res['MC'])
        res['ratio'][strategy] = A.data.ratio

    pd.Series(res['ratio']).to_csv(os.path.join(params['path'], 'ratio.csv'))
    #  画图
    fit = pd.concat(res['fit']).stack().reset_index()
    fit.columns = ['strategy', 'sample', 'formula', 'value']
    MC_Noise = pd.concat(res['MC_Noise']).stack().reset_index()
    MC_Noise.columns = ['strategy', 'sample', 'formula', 'value']
    MC = pd.concat(res['MC']).stack().reset_index()
    MC.columns = ['strategy', 'sample', 'formula', 'value']

    sns.catplot(x='sample', y='value', hue='formula', col='strategy', kind="bar", col_wrap=3, aspect=1.5, data=fit)
    plt.savefig(os.path.join(params['path'], f"GEP_noise_{int(funcNoise * 100)}%.png"), dpi=100, bbox_inches='tight')

    fig = plt.figure(figsize=(15, 10))
    ax1 = fig.add_subplot(2, 1, 1)
    sns.barplot(x='strategy', y='value', hue='formula', data=MC_Noise, ax=ax1)
    ax1.legend().set_visible(False)
    plt.title(f"MC_Noise_{int(funcNoise * 100)}%")
    ax2 = fig.add_subplot(2, 1, 2)
    sns.barplot(x='strategy', y='value', hue='formula', data=MC, ax=ax2)
    plt.title("MC")
    plt.savefig(os.path.join(params['path'], f"GEP_MC_noise_{int(funcNoise * 100)}%.png"), dpi=100, bbox_inches='tight')
    plt.show()


def main2():
    res = []
    funcName = 'stock_MC'
    funcNoise = 0.0
    verify = 0.2
    trainI, verifyI, testI = {}, {}, {}
    for i in range(50):
        seed = np.random.randint(0, 100)
        print(i)
        sampleClass = select_sample(funcName, funcNoise, seed, 1000, strategy='True Boll', verify=0.2)

        # model parameter
        params = {
            "path": r'A:\Work\Working\13.基于机器学习的因子挖掘\真实样本测试\test',
            "strategy": 'True Boll',
            "data": sampleClass,
            "funcName": funcName,
            "funcNoise": funcNoise,
            "size": sampleClass.sampleL,
            "seed": seed,
            "verify": verify,
            "paramL": 400,
            "h": 5,  # 基因头部长度
            "n_genes": 3,  # 染色体基因个数
            "n_elites": 50,
            "const": 'ENC',
            "penal": 0.5,
            "n_pop": 500,  # 种群规模
            "n_gen": 10,  # 进化代数
        }
        print(f"test: {sum(sampleClass.sign)}, n_elites: {params['n_elites']}")
        A = GEPClassification(opt=1)
        A.set_params(**params)
        A.run()

        iters = A.res['formula'][(A.res['formula']['gen'] == 10)].sort_values(['verify']).iloc[-5:, :]
        iters['iters'] = i
        res.append(iters)
        Write = pd.ExcelWriter(os.path.join(params['path'], f'GEP_seed{seed}.xlsx'))
        pd.DataFrame(A.logs).to_excel(Write, sheet_name='iteration')
        A.res['formula'].to_excel(Write, sheet_name='formula')
        Write.close()
    pd.concat(res).to_csv(os.path.join(params['path'], f'GEP_noise{funcNoise}.csv'))

        # trainI[i] = pd.DataFrame(A.logs)['max_train']
        # verifyI[i] = pd.DataFrame(A.logs)['max_verify']
        # testI[i] = pd.DataFrame(A.res['fit'])['testFitR']
    # pd.DataFrame(trainI).to_csv(os.path.join(params['path'], f'GEP_iteration_train.csv'), index=False)
    # pd.DataFrame(verifyI).to_csv(os.path.join(params['path'], f'GEP_iteration_verify.csv'), index=False)
    # pd.DataFrame(testI).to_csv(os.path.join(params['path'], f'GEP_iteration_testR.csv'), index=False)


if __name__ == '__main__':
    main2()
