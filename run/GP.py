# -*-coding:utf-8-*-
# @Time:   2021/7/6 17:16
# @Author: FC
# @Email:  18817289038@163.com

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from gplearn.fitness import make_fitness
from gplearn.functions import make_function
from gplearn.genetic import SymbolicRegressor


def _backtest_return(y, y_pred, w):
    sign_pred = np.sign(y_pred)
    num_change = sum(sign_pred[:-1] != sign_pred[1:])
    res = sum(y * sign_pred) - 0.002 * num_change
    return res


backtest_return = make_fitness(_backtest_return, greater_is_better=True)


def _logicl_1(x):
    return np.where(x > 0, 1, -1)


def _logicl_2(x1, x2):
    return np.where(x1 > x2, 1, -1)


def _square(x):
    return x ** 2


logicl_1 = make_function(_logicl_1, 'logicl_1', arity=1)
logicl_2 = make_function(_logicl_2, 'logicl_2', arity=2)
square = make_function(_square, 'square', arity=1)

function_set = ('add', 'sub', 'mul', 'div', 'abs', 'neg', 'max', 'min', logicl_1, logicl_2, square)

np.random.seed(1)
x = np.random.randn(1000, 13)
y = np.where(x[:, 0] > 0.1, 2, -1) + x[:, 3] ** 2 + (x[:, 2] + 0.2) ** 2 + 0.7 + (x[:, 1] - x[:, 4]) ** 2  #
x_train = x[:-100]
y_train = y[:-100]
x_test = x[-100:]
y_test = y[-100:]
model = SymbolicRegressor(population_size=500,
                          generations=50,
                          tournament_size=80,  # 个体进入下一代的数量
                          stopping_criteria=0.0,
                          const_range=(-1, 1),
                          init_depth=(2, 10),
                          init_method='half and half',
                          function_set=function_set,
                          metric='mse',
                          parsimony_coefficient=0.05,
                          p_crossover=0.7,
                          p_subtree_mutation=0.1,
                          p_hoist_mutation=0.1,
                          p_point_mutation=0.1,
                          p_point_replace=0.1,
                          max_samples=0.8,
                          warm_start=False,
                          low_memory=False,
                          n_jobs=1,
                          verbose=1,
                          random_state=0)

model.fit(x_train, y_train)
# X_gp = model.predict(x_test)
score_gp = model.score(x_test, y_test)
g_y = model.predict(x_test)

diff = pd.Series(g_y - y_test)
res = pd.DataFrame([y_test, g_y], index=['y', 'pred']).T.sort_values(by='y')
res['y_'] = res['y']

fig = plt.figure(figsize=(10, 8))
ax1 = fig.add_subplot(2, 1, 1)
res.plot.scatter(x='y', y='pred', ax=ax1)
res.plot.line(x='y', y='y_', ax=ax1)
ax2 = fig.add_subplot(2, 1, 2)
diff.plot.bar(ax=ax2)
plt.suptitle(f"$R^2:{score_gp:.4f}$")

plt.show()
print(model)
print(score_gp)
print('s')
