# coding=utf-8
"""
Project: TimeSeries_Tutorial
Create time: 2019/12/12
Introduction:
"""
import numpy as np
import random
import matplotlib.pyplot as plt
from arch import arch_model
from pymannkendall import pre_whitening_modification_test
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


random.seed(1)
data_increase = [random.gauss(0, i*0.01) for i in range(0, 100)]
squared_increase = np.array([x**2 for x in data_increase])
# 划分训练集&测试集
n_test = int(len(data_increase) * 0.1)
train = data_increase[:-n_test]
test = data_increase[-n_test:]

plt.plot(data_increase)
plot_acf(squared_increase, lags=50)
plot_pacf(squared_increase, lags=50)
# 定义模型
model = arch_model(train, mean='Zero', vol='GARCH', p=15, q=15, rescale='10*y')  # 均值mean='Zero', 滞后参数p=15

# 拟合模型
model_fit = model.fit()
print(model_fit.summary)
model_fit.plot()

# 预测测试集
yhat = model_fit.forecast(horizon=n_test)

var = [i*0.01 for i in range(0, 100)]
# print(model_fit.conditional_volatility)
result = pre_whitening_modification_test(model_fit.conditional_volatility)
if result.trend == 'increasing':
    print('波动增加')
elif result.trend == 'decreasing':
    print('波动减小')
else:
    print('状态未知')

plt.show()
