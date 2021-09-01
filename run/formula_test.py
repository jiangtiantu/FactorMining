# -*-coding:utf-8-*-
# @Time:   2021/7/26 11:22
# @Author: FC
# @Email:  18817289038@163.com

import os
import numpy as np
import pandas as pd
import geppy as gep
import seaborn as sns
import bottleneck as bk
import matplotlib.pyplot as plt

from scipy.stats import t, hmean, ttest_ind
from functools import reduce
from typing import Any, Tuple
from utility.utility import checkDim
from collections import defaultdict

from base_class.template import GEP
from data_API.sample_timing import select_sample
from utility.operator_func import GEPFunctionTiming
from run.GEP_tming import GEPTiming

sns.set(font='SimHei', palette="muted", color_codes=True)
sns.set_style("darkgrid", {"font.sans-serif": ['simhei', 'Droid Sans Fallback']})

FUNC_TIMING = GEPFunctionTiming()

replace = {
    "信号占比0.1": "信号占比0.5",
    "信号占比0.3": "信号占比0.4",
    "信号占比0.5": "信号占比0.3",
    "信号占比0.8": "信号占比0.195",
    "信号占比1": "信号占比0.12",
    "信号占比1.1": "信号占比0.09",
    "信号占比1.2": "信号占比0.07",
    "信号占比1.3": "信号占比0.053",
}


def get_data():
    sampleClass = select_sample('index300', seed=2, noise=0, verify=0.2)
    return sampleClass


def FUNCS(data: np.array):
    FUNC_TIMING.dim = data.X.shape[0]

    # signal = data.X[:, 3] - FUNC_TIMING.min_EVO(data.X[:, 2], data.X[:, 0]) + FUNC_TIMING.rolling_argmin_EVO(data.X[:, 3]) * 2
    # signal = FUNC_TIMING.rolling_argmin_EVO(data.X[:, 3]) + data.X[:, 3] + FUNC_TIMING.log_EVO(data.X[:, 2]) - data.X[:, 0]
    # signal = data.X[:, 3] - data.X[:, 0] + 0.8252 + FUNC_TIMING.rolling_argmax_EVO(FUNC_TIMING.mean_EVO(data.X[:, 5], FUNC_TIMING.rolling_std_EVO(data.X[:, 4])))
    # signal = data.X[:, 3] - data.X[:, 0] + FUNC_TIMING.log_EVO(data.X[:, 3])  # data.X[:, 3] - data.X[:, 0]
    signal = FUNC_TIMING.signal1_EVO(data.X[:, 4]) + FUNC_TIMING.log_EVO(FUNC_TIMING.sqrt_EVO(data.X[:, 4])) - data.X[:, 0] + data.X[:, 3]
    ret = np.where(signal >= 0, data.Y, 0)
    rets = pd.DataFrame([ret, data.Y], index=['strategy', 'hs300']).T
    navs = (rets + 1).cumprod()
    navs['relative'] = navs['strategy'] / navs['hs300']
    navs[['strategy', 'hs300']].plot()
    navs['relative'].plot(kind='line', secondary_y=True, rot=30)
    return sign


def test(yPreN, ratio, data):
    return test_noise, test


def main():
    data = get_data()
    nav = FUNCS(data)
    return res_df


def plot1():
    path = r'A:\Work\Working\13.基于机器学习的因子挖掘\真实样本测试\3.真实样本挖掘\9.常数_信号_函数个数问题_无标签'
    folders = os.listdir(path)
    resList = []
    for folder in folders:
        folderPath = os.path.join(path, folder)
        best = pd.read_csv(os.path.join(folderPath, "GEP_res.csv"))
        res_df = best[['train', 'verify', 'test', 'trainS']]
        res_df[['funcNum', 'const', 'signal']] = folder.split('_')
        resList.append(res_df)

    res = pd.concat(resList)
    res['signal'].replace(replace, inplace=True)

    print('plot')
    plot
    for const in ['无常数', '有常数']:
        for dataName in ['train', 'verify', 'test']:
            res['train'] = np.where(res['train'] > 10, 10, res['train'])
            res['train'] = np.where(res['train'] < -10, -10, res['train'])
            if dataName == 'train':
                bins = [i for i in range(-10, 11)]
            else:
                bins = [i / 40 for i in range(20, 41)]
            dataSub = res[res['const'] == const]
            sns.displot(
                dataSub, x=dataName, col="signal", row="funcNum",
                bins=bins, height=2, facet_kws=dict(margin_titles=True),
            )
            plt.savefig(os.path.join(r'A:\Work\Working\13.基于机器学习的因子挖掘\真实样本测试\3.真实样本挖掘\结果',
                                     f"{dataName}_有标签_{const}.png"),
                        dpi=100,
                        bbox_inches='tight')

    # res_describe = res.groupby(['funcNum', 'const', 'signal']).describe().reset_index()
    # dataSub = res_describe[res_describe['const'] == '有常数']
    # fig = plt.figure(figsize=(14, 12))
    # flag = 1
    # for data1 in ['train', 'verify', 'test']:
    #     for stat in ['mean', '50%', 'std']:
    #         ax = fig.add_subplot(3, 3, flag)
    #         data_heat = pd.pivot(columns='funcNum', values=(data1, stat), index='signal', data=dataSub)
    #         sns.heatmap(data_heat, annot=True, cmap="YlGnBu", annot_kws={'size': 9}, ax=ax)
    #         ax.set_title((data1, stat))
    #         flag += 1
    # plt.show()
    res_describe.to_csv(r'A:\Work\Working\13.基于机器学习的因子挖掘\真实样本测试\3.真实样本挖掘\结果\有标签.csv',
                        encoding='GBK')


def plot2():
    path = r'A:\Work\Working\13.基于机器学习的因子挖掘\真实样本测试\3.真实样本挖掘\10.信号强度_信号个数问题_无标签'
    folders = os.listdir(path)
    resList = []
    for folder in folders:
        folderPath = os.path.join(path, folder)
        best = pd.read_csv(os.path.join(folderPath, "GEP_res.csv"))
        res_df = best[['train', 'verify', 'test', 'trainS']]
        res_df[['signalNum', 'signalRSI']] = folder.split('_')
        resList.append(res_df)

    res = pd.concat(resList)

    res_describe = res.groupby(['signalNum', 'signalRSI']).describe().reset_index()
    fig = plt.figure(figsize=(16, 14))
    flag = 1
    for data1 in ['train', 'verify', 'test']:
        for stat in ['mean', '50%', 'std']:
            ax = fig.add_subplot(3, 3, flag)
            data_heat = pd.pivot(columns='signalNum', values=(data1, stat), index='signalRSI', data=res_describe)
            sns.heatmap(data_heat, annot=True, cmap="YlGnBu", annot_kws={'size': 9}, ax=ax)
            ax.set_title((data1, stat))
            flag += 1
    plt.show()

    # plot
    for dataName in ['train', 'verify', 'test']:
        res['train'] = np.where(res['train'] > 10, 10, res['train'])
        res['train'] = np.where(res['train'] < -10, -10, res['train'])
        if dataName == 'train':
            bins = [i for i in range(-10, 11)]
        else:
            bins = [i / 40 for i in range(20, 41)]
        sns.displot(
            res, x=dataName, col="signalRSI", row="signalNum",
            bins=bins, height=3, facet_kws=dict(margin_titles=True),
        )
        plt.savefig(os.path.join(r'A:\Work\Working\13.基于机器学习的因子挖掘\真实样本测试\3.真实样本挖掘\结果',
                                 f"{dataName}_无标签_信号强弱.png"),
                    dpi=100,
                    bbox_inches='tight')
    res_describe = res.groupby(['signalNum', 'signalRSI']).describe()
    res_describe.to_csv(r'A:\Work\Working\13.基于机器学习的因子挖掘\真实样本测试\3.真实样本挖掘\结果\信号强度测试.csv',
                        encoding='GBK')


def plot3():
    path = r'A:\Work\Working\13.基于机器学习的因子挖掘\真实样本测试\3.真实样本挖掘\10.信号强度_信号个数问题_无标签'
    folders = os.listdir(path)
    train, verify = {}, {}

    for folder in folders:
        folderPath = os.path.join(path, folder)
        trainList, verifyList = [], []
        for i in range(30):
            iterSub = pd.read_csv(os.path.join(folderPath, f"iteration{i}.csv"))
            trainList.append(iterSub['avg_train'])
            verifyList.append(iterSub['max_verify'])
        train[folder] = pd.concat(trainList, axis=1).mean(axis=1)
        verify[folder] = pd.concat(verifyList, axis=1).mean(axis=1)

    # pd.DataFrame(train).to_csv(r'A:\Work\Working\13.基于机器学习的因子挖掘\真实样本测试\3.真实样本挖掘\结果\训练集迭代.csv',
    #                            encoding='GBK')
    pd.DataFrame(verify).to_csv(r'A:\Work\Working\13.基于机器学习的因子挖掘\真实样本测试\3.真实样本挖掘\结果\验证集迭代.csv',
                                encoding='GBK')


def test1():
    path = r'A:\Work\Working\13.基于机器学习的因子挖掘\相关实验\19.真实样本_独立样本t检验_有无构造函数_分类问题_Timing\有构造函数'
    folders = os.listdir(path)
    res, best = {}, defaultdict(list)
    for folder in folders:
        print(folder)
        if folder in ['对比', '结果整理.xlsx', '结果', 'F1.csv']:
            continue
        folderPath = os.path.join(path, folder)
        files = os.listdir(folderPath)
        m = []
        for file in files:
            data = pd.read_csv(os.path.join(folderPath, file), encoding='GBK')
            if file.startswith('formula'):
                formulaSub2 = data[(data['verify'] > 0) & (data['verify2'] > 0) & (data['train'] > 0)]
                data.loc[formulaSub2.index, 'verifyF2'] = hmean(formulaSub2[['train', 'verify', 'verify2']], axis=1)
                eff10 = data.drop_duplicates(subset=['train', 'verify', 'verify2', 'test', 'test2']).sort_values(
                    'verifyF2', ascending=False).iloc[:1, ]
                eff10['p'] = eff10['test2'].map(lambda x: t.sf(x, 136))
                eff10['iter'] = file[7:-4]
                best[folder].append(eff10)  # [eff10['verifyF1'] >= 1.98]
            else:
                m.append(data)
        # res[folder] = reduce(lambda x, y: x + y, m) / 50
    # iterRes = pd.DataFrame(res).to_csv(os.path.join(path, 'testF1.csv'))
    resList = []
    for i in best.keys():
        sub = pd.concat(best[i])
        sub['Name'] = i
        resList.append(sub)
    res = pd.concat(resList)
    res_df = pd.pivot(data=res, index='iter', columns='Name', value='p')
    bestFormula = pd.concat(best).to_csv(os.path.join(path, 'testF2.csv'))
    # plot


def P():
    p = pd.read_csv(r'A:\Work\Working\13.基于机器学习的因子挖掘\真实样本测试\3.真实样本挖掘\16.无构造函数\有构造函数\结果\1.3有构造函数_最优公式.csv', encoding='GBK')
    g10 = p.sort_values(['verifyF1'], ascending=False)
    m10 = {}
    for i in g10['Name1'].drop_duplicates():
        if i == '信号占比0.2_信号强度0.0':
            continue
        m10[i] = g10[g10['Name1'] == i].sort_values('p').reset_index()['test']
    pd.DataFrame(m10).to_csv(r'A:\Work\Working\13.基于机器学习的因子挖掘\真实样本测试\3.真实样本挖掘\16.无构造函数\有构造函数\结果\前10_test.csv', encoding='GBK')


def MCS():
    path = r'A:\Work\Working\13.基于机器学习的因子挖掘\真实样本测试\3.真实样本挖掘\结果'
    data = pd.read_csv(os.path.join(path, 'formula5.csv'), encoding='GBK')
    dataInput = data[['test2', 'p', 'formula', 'iter']]
    res = []
    for ind_, row_ in dataInput.iterrows():
        # ocupy, stre = row_['Name1'].split('_')
        # ocu, ratio = float(ocupy[4:]), float(stre[4:])
        # seed = int(row_['Name2'][8:])
        sampleClass = select_sample('index300', seed=int(row_['iter']), ratio=0, ocu=1)
        # print(ocupy, stre, ocu)
        model_params = {
            "path": path,
            "data": sampleClass,
            "funcName": 'index300',

            "paramL": 400,
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
        A = GEPTiming(opt=1)
        A.set_params(**model_params)
        A.run()
        A.gep_p.add_function(A.sum_gen, 2, 'sum_gen')
        A.Func.dim = 1216
        yAllPre = _compile_gene(row_['formula'], A.gep_p)(*tuple(sampleClass.X.T))

        yAllPre = checkDim(yAllPre, dim=A.data.sampleL)[0]
        ySignAll = np.where(yAllPre > 0, 1, 0)
        yRetTest1 = A.data.Y[np.argwhere((A.data.Label == 3) & (ySignAll == 1)).flatten()]
        yRetTest0 = A.data.Y[np.argwhere((A.data.Label == 3) & (ySignAll == 0)).flatten()]
        t_O = A.ttest(yRetTest1, yRetTest0, 3)  # 原始t,用来检验样本是否正确

        # 测试集样本
        yRetTest = A.data.Y[(A.data.Label == 3).flatten()]
        Test1 = np.mean(yRetTest1)  # 测试集标记为1的收益率均值
        Mean = []
        for i in range(1000):
            mean1 = np.mean(np.random.choice(yRetTest, len(yRetTest1), replace=False))
            if mean1 > Test1:
                Mean.append(1)
            else:
                Mean.append(0)
        p_value = sum(Mean) / 1000
        res.append(pd.Series([row_['iter'], row_['test2'], row_['p'], t_O, p_value, row_['formula']],
                             index=['iter', 'test_O', 'p_O', 't_New', 'p_new', 'formula']))
    pd.concat(res, axis=1)


def _compile_gene(g, pset):
    """
    Compile one gene *g* with the primitive set *pset*.
    :return: a function or an evaluated result
    """
    code = str(g)
    if len(pset.input_names) > 0:  # form a Lambda function
        args = ', '.join(pset.input_names)
        code = 'lambda {}: {}'.format(args, code)
    # evaluate the code
    try:
        return eval(code, pset.globals, {})
    except MemoryError:
        print('s')


# p = pd.read_csv(r'A:\Work\Working\13.基于机器学习的因子挖掘\真实样本测试\3.真实样本挖掘\12.函数参数优化\test.csv')


def test4():
    qq = 0
    path = r'A:\Work\Working\13.基于机器学习的因子挖掘\相关实验\21.真实样本_独立样本t检验_自适应概率_分类问题_Timing\自适应概率'
    files = os.listdir(path)
    m, bestF1, bestF2 = [], [], []
    for file in files:
        if file in ['结果整理.xlsx', '对比.xlsx', '迭代结果.xlsx', '固定概率', '历史']:
            continue
        print(file)
        data = pd.read_csv(os.path.join(path, file))
        if file.startswith('formula'):
            effF1 = data.drop_duplicates(subset=['train', 'verify', 'verify2', 'test', 'test2']).sort_values(
                'verifyF1', ascending=False).iloc[:1, ]
            effF1['p'] = effF1['test2'].map(lambda x: t.sf(x, 136))
            effF1['iter'] = file[7:-4]
            bestF1.append(effF1)

            effF2 = data.drop_duplicates(subset=['train', 'verify', 'verify2', 'test', 'test2']).sort_values(
                'verifyF2', ascending=False).iloc[:1, ]
            effF2['p'] = effF2['test2'].map(lambda x: t.sf(x, 136))
            effF2['iter'] = file[7:-4]
            bestF2.append(effF2)
        else:
            m.append(data)
    # iters = reduce(lambda x, y: x + y, m) / 30
    pd.concat(bestF1).to_csv(os.path.join(path, 'F1.csv'), encoding='GBK')
    pd.concat(bestF2).to_csv(os.path.join(path, 'F2.csv'), encoding='GBK')
    # iters.to_csv(os.path.join(path, '迭代.csv'), encoding='GBK')


if __name__ == '__main__':
    op = pd.DataFrame({"A": [1, 2, 3], "R": [4, 5, 6]})
    p = lambda *x: sum(*x)
    res = p(*dict(op.iteritems()).values())
    test4()
