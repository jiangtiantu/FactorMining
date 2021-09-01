# -*-coding:utf-8-*-
# @Time:   2021/7/14 13:43
# @Author: FC
# @Email:  18817289038@163.com

import random
import numpy as np
import geppy as gep
from copy import deepcopy

from operator import eq
from functools import partial
from bisect import bisect_right
from dataclasses import dataclass
from collections import defaultdict
from abc import ABC, abstractmethod
from deap import creator, base, tools
from typing import List, Callable, Dict

from base_class.basic import gep_simple as run1  # 自适应概率
from base_class.basic2 import gep_simple as run2
from utility.operator_func import GEPFunctionTiming, GEPFunctionSelecting

"""
基因表达式相关模块
"""


def verifyValue(obj):
    """Returns directly the argument *obj*.
    """
    return obj


class Chromosome(gep.Chromosome):

    @staticmethod
    def compare(ind):
        return {'fitness': ind.fitness, 'verify': ind.verify}

    def __hash__(self):
        return hash(self.__str__())

    def __eq__(self, other):
        return hash(self.__str__()) == hash(other.__str__())  # if self.compare(self) == self.compare(other) else False

    def __str__(self):
        """
        Return the expressions in a human readable string.
        """
        if self.linker is not None:
            return '{0}({1})'.format(self.linker.__name__, ','.join(str(g) for g in self))
        else:
            assert len(self) == 1
            return str(self[0])


class StatisticGEP(tools.Statistics):
    """
    训练集和验证集统计指标
    """

    def __init__(self, key, **kwargs):
        super().__init__(key)
        self.functions = defaultdict(dict)

        for ind, func in kwargs.items():
            setattr(self, ind, func)

    def register(self, name: str, function: Callable, dataType: str = '', *args, **kwargs):
        """
        :param name: 函数名称
        :param function: 函数
        :param dataType: 数据类型
        :param args:
        :param kwargs:
        :return:
        """
        self.functions[dataType][name] = partial(function, *args, **kwargs)
        self.fields.append(name)

    def compile(self, data):
        """
        Apply to the input sequence *data* each registered function and return the results as a dictionary.
        :param data: Sequence of objects on which the statistics are computed.
        """
        entry = dict()
        for dataName, funcs in self.functions.items():
            indFunc = getattr(self, dataName)
            values = tuple(indFunc(elem) for elem in data)

            for key, func in funcs.items():
                entry[key] = round(func(values), 4)

        return entry


class HallOfFameGEP(tools.HallOfFame):
    """
    保存最优训练集公式和验证集公式
    """

    def __init__(self, maxsize):
        super().__init__(maxsize, similar=eq)
        self.keys2 = list()
        self.items2 = list()
        self.inds = defaultdict(list)

    def update(self, population):
        self.inds[population[0].gen] = set(population)  # 保留每代个体

        for ind in population:
            if len(self) == 0 and self.maxsize != 0:
                self.insert(population[0])
                continue
            else:
                if ind.fitness > self[-1].fitness or len(self) < self.maxsize:
                    for hofer in self:
                        if self.similar(ind, hofer):
                            break
                    else:
                        if len(self) >= self.maxsize:
                            self.remove(-1)
                        self.insert(ind)
        for ind in population:
            if len(self.items2) == 0 and self.maxsize != 0:
                self.insert2(population[0])
                continue
            else:
                if ind.verify > self.items2[-1].verify or len(self.items2) < self.maxsize:
                    for hofer in self.items2:
                        if self.similar(ind, hofer):
                            break
                    else:
                        if len(self.items2) >= self.maxsize:
                            self.remove2(-1)
                        self.insert2(ind)

    def insert2(self, item):
        item = deepcopy(item)
        i = bisect_right(self.keys2, item.verify)
        self.items2.insert(len(self.items2) - i, item)
        self.keys2.insert(i, item.verify)

    def remove2(self, index):
        del self.keys2[len(self.items2) - (index % len(self.items2) + 1)]
        del self.items2[index]

    def clear(self):
        super().clear()
        del self.items2[:]
        del self.keys2[:]


class GEP(ABC):
    n_pop = 500  # 种群规模
    n_gen = 50  # 进化代数
    h = 6  # 基因头部长度
    n_genes = 3  # 染色体基因个数
    n_elites = 10  # 保留精英个体数
    r = 5  # RNC length
    penal = 1  # fitness惩罚阈值
    const = 'ENC'  # 常量的类型
    bestNum = 10  # 存储最优个体数量

    def __init__(self, opt: int = 1):
        self.check_class(opt)
        self.dataSet: Dict[str, dataclass]  # 数据类
        self.TB = gep.Toolbox()
        self.Func = GEPFunctionTiming()

        self.gep_p = None  # PrimitiveSet
        self.stats = None  # Statistic
        self.pops = None  # Population
        self.logs = None  # Evolutionary record
        self.bestInd = None  # Excellent individual

    # 最值定义
    @staticmethod
    def check_class(opt: int = 1):
        if 'FitnessMin' in creator.__dict__.keys():
            del creator.__dict__['FitnessMin']
        if 'Individual' in creator.__dict__.keys():
            del creator.__dict__['Individual']

        creator.create("FitnessMin", base.Fitness, weights=(opt,))
        creator.create("Individual", Chromosome,
                       fitness=creator.FitnessMin,
                       verify=creator.FitnessMin,
                       verify2=creator.FitnessMin,
                       test=creator.FitnessMin,
                       test2=creator.FitnessMin,
                       testR=creator.FitnessMin,  # 实际准确率

                       flag=0,  # 是否发生过变异或选择等进化操作
                       trainInfo=None,
                       verifyInfo=None,
                       verify2Info=None,
                       testInfo=None,
                       gen=None)

    def set_params(self, **kwargs):
        for paramName, paramValue in kwargs.items():
            setattr(self, paramName, paramValue)

    @abstractmethod
    def primitive(self):
        """
        终端集
        """
        self.gep_p = gep.PrimitiveSet('Main', input_names=[f"X{pos}" for pos in range(self.dataSet['X'].X.shape[1])])
        for func_, funcClass in self.Func.funcAtt.items():
            self.gep_p.add_function(funcClass.funcMethod, funcClass.arity, func_)

    @abstractmethod
    def constant(self):
        """
        常量定义
        """
        if self.const == 'ENC':
            self._ENC()
        elif self.const == 'RNC':
            self._RNC()
        else:
            self.TB.register('gene_gen', gep.Gene, pset=self.gep_p, head_length=self.h)

    def _ENC(self):
        self.gep_p.add_ephemeral_terminal(name='enc', gen=self._randomR)
        self.TB.register('gene_gen', gep.Gene, pset=self.gep_p, head_length=self.h)
        self.TB.register('mut_ephemeral', gep.mutate_uniform_ephemeral, ind_pb=0.1, pb=1)

    def _RNC(self):
        self.gep_p.add_rnc_terminal()
        self.TB.register('rnc_gen', random.uniform, a=-2, b=2)
        self.TB.register('gene_gen', gep.GeneDc, pset=self.gep_p, head_length=self.h, rnc_gen=self.TB.rnc_gen,
                         rnc_array_length=self.r)
        self.TB.register('mut_dc', gep.mutate_uniform_dc, ind_pb=0.1, pb=0.5)
        self.TB.register('mut_invert_dc', gep.invert_dc, pb=0.1)
        self.TB.register('mut_transpose_dc', gep.transpose_dc, pb=0.1)
        self.TB.register('mut_rnc_array_dc', gep.mutate_rnc_array_dc, rnc_gen=self.TB.rnc_gen, ind_pb=0.1, pb=0.5)

    @staticmethod
    def _randomR():
        return np.random.uniform(-30, 30)

    @abstractmethod
    def genetic(self):
        """
        遗传多样性
        """
        # 生物遗传
        self.TB.register('select', tools.selTournament, tournsize=5)  # 单次抽取5个个体进行竞争
        self.TB.register('mut_uniform', gep.mutate_uniform, pset=self.gep_p, ind_pb=0.1, pb=0.5)  # 点变异
        self.TB.register('mut_is_transpose', gep.is_transpose, pb=0.1)  # IS 转座
        self.TB.register('mut_ris_transpose', gep.ris_transpose, pb=0.1)  # RIS 转座
        self.TB.register('mut_gene_transpose', gep.gene_transpose, pb=0.1)  # 基因转座

        self.TB.register('cx_1p', gep.crossover_one_point, pb=0.5)  # 单点交叉
        self.TB.register('cx_2p', gep.crossover_two_point, pb=0.5)  # 双点交叉
        self.TB.register('cx_gene', gep.crossover_gene, pb=0.5)  # 基因交叉

        self.TB.register('mut_invert', gep.invert, pb=0.1)  # 反转

    @abstractmethod
    def evolution(self,
                  individual: Chromosome,
                  **kwargs) -> float:
        """
        适应度函数
        """
        pass

    def penalty_func(self, **kwargs) -> float:
        """
        罚函数
        """
        pass

    def statistic(self):
        # 结果的统计
        self.stats = StatisticGEP(key=lambda ind: ind.fitness.values[0],
                                  oth1=lambda ind: ind.verify.values[0],
                                  oth2=lambda ind: ind.verify2.values[0])

        # self.stats.register("std_train", np.std)
        self.stats.register(name="min_train", function=np.min, dataType='key')
        self.stats.register(name="max_train", function=np.max, dataType='key')
        # self.stats.register(name="med_train", function=np.median, dataType='key')
        self.stats.register(name="avg_train", function=np.mean, dataType='key')

    def competition(self, linkFunc: Callable):
        self.TB.register('individual', creator.Individual, gene_gen=self.TB.gene_gen, n_genes=self.n_genes,
                         linker=linkFunc)
        self.TB.register("population", tools.initRepeat, list, self.TB.individual)
        # 解码
        self.TB.register('compile', gep.compile_, pset=self.gep_p)

        self.pops = self.TB.population(n=self.n_pop)
        self.bestInd = HallOfFameGEP(self.bestNum)  # 名人堂

        # pool = mp.Pool(processes=3)
        # self.TB.register('map', pool.map)

        self.pops, self.logs = run2(self.pops,
                                    self.TB,
                                    n_generations=self.n_gen,
                                    n_elites=self.n_elites,
                                    stats=self.stats,
                                    hall_of_fame=self.bestInd,
                                    verbose=True)
        # pool.close()
