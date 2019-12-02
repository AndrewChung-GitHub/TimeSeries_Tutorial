> 用于时间序列分析和预测的ARIMA模型参数可能很难配置，因为有3个参数需要通过迭代试验来评估。我们可以使用网格搜索程序自动化为ARIMA模型评估大量超参数的过程。

本教程主要涵盖：

- 可用于调整ARIMA超参数以进行滚动一步预测的一般过程。
- 如何在标准单变量时间序列数据集上应用ARIMA超参数优化。
- 扩展程序的想法，以更精细和强大的模型。

## 01. 评估ARIMA模型
我们可以通过在训练数据集上准备ARIMA模型并评估测试集的预测来评估ARIMA模型。
- 将数据集拆分为训练(66%)和测试集(34%)
- 遍历测试数据集中的时间步长
    - 训练ARIMA模型
    - 进一步预测
    - 获取并存储实际观察结果
- 计算预测的误差分数与预期值的比较(计算并返回均方误差分数)

```py
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error


# evaluate an ARIMA model for a given order (p,d,q)
def evaluate_arima_model(X, arima_order):
    # prepare training dataset
    train_size = int(len(X) * 0.66)
    train, test = X[0:train_size], X[train_size:]
    history = [x for x in train]
    # make predictions
    predictions = list()
    for t in range(len(test)):
        model = ARIMA(history, order=arima_order)
        model_fit = model.fit(disp=0)
        yhat = model_fit.forecast()[0]
        predictions.append(yhat)
        history.append(test[t])
    # calculate out of sample error
    error = mean_squared_error(test, predictions)
    return error
```
现在我们知道如何评估一组ARIMA超参数，让我们看看我们如何重复调用这个函数来计算要评估的参数网格。 

## 02. 迭代ARIMA参数
用户必须指定要迭代的(p, d, q)参数网格，然后调用`evaluate_arima_model()`函数，为每个参数创建模型并评估其性能。通过`evaluate_models()`函数实现跟踪最小误差数值及其对应的参数配置。

两个注意事项：
- 确保输入数据类型是`float`(而不是`int`,`string`)，因为这可能导致ARIMA过程失败
- `statsmodels ARIMA`程序内部使用数值优化程序来为模型找到一组系数过程中，可能会失败而引发异常。我们必须捕获这些异常并跳过导致问题的配置。
```py
import warnings
warnings.filterwarnings('ignore')


# evaluate combinations of p, d and q values for an ARIMA model
def evaluate_models(dataset, p_values, d_values, q_values):
    dataset = dataset.astype('float32')
    best_score, best_cfg = float("inf"), None
    for p in p_values:
        for d in d_values:
            for q in q_values:
                order = (p, d, q)
                try:
                    mse = evaluate_arima_model(dataset, order)
                    if mse < best_score:
                        best_score, best_cfg = mse, order
                    print('ARIMA%s MSE=%.3f' % (order, mse))
                except:
                    continue
    print('Best ARIMA%s MSE=%.3f' % (best_cfg, best_score))
```

## 03. 案例研究
### 01. 洗发水销售案例
> - Shampoo Sales数据集描述了3年期间每月洗发水的销售数量。单位是销售计数，有36个观察。原始数据集归功于Makridakis，Wheelwright和Hyndman（1998）。
> - 下载地址：https://raw.githubusercontent.com/jbrownlee/Datasets/master/shampoo.csv

```py
import pandas as pd
from datetime import datetime
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')


# load dataset
def parser(x):
    return datetime.strptime('190' + x, '%Y-%m')


series = pd.read_csv(r'../data/shampoo.csv', header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)

# evaluate parameters
p_values = [0, 1, 2, 4, 6, 8, 10]
d_values = range(0, 3)
q_values = range(0, 3)
evaluate_models(series, p_values, d_values, q_values)

# result
#> ARIMA(0, 0, 0) MSE=52425.268
#> ARIMA(0, 0, 1) MSE=38145.166
#> ARIMA(0, 0, 2) MSE=23989.599
#> ARIMA(0, 1, 0) MSE=18003.173
#> ... ...
#> Best ARIMA(6, 1, 1) MSE=4291.323
```

### 02.每日女性出生案例
> - 每日女性出生数据集描述了1959年加利福尼亚州每日女性出生人数。单位是计数，有365个观测值。数据集的来源归功于Newton（1988）。
> - 下载地址: https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-total-female-births.csv

```py
import pandas as pd
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')


# load dataset
series = pd.read_csv(r'../data/daily-total-female-births.csv', header=0)
series['Date'] = pd.to_datetime(series['Date'])
series.set_index('Date', inplace=True)

# evaluate parameters
p_values = [0, 1, 2, 4, 6, 8, 10]
d_values = range(0, 3)
q_values = range(0, 3)
warnings.filterwarnings("ignore")
evaluate_models(series.values, p_values, d_values, q_values)

#> result
#> ARIMA(0, 0, 0) MSE=67.063
#> ARIMA(0, 0, 1) MSE=62.165
#> ARIMA(0, 0, 2) MSE=60.386
#> ARIMA(0, 1, 0) MSE=84.038
#> ARIMA(0, 1, 1) MSE=56.653
#> ... ...
#> Best ARIMA(6, 1, 0) MSE=53.187
```

## 04. 扩展
本节列出了一些扩展可能希望探索的方法的想法。

- **种子网格**。ACF和PACF图的经典诊断工具仍然可以与用于搜索ARIMA参数网格的结果一起使用。
- **替代措施**。搜索旨在优化样本外均方误差。这可以更改为另一个样本外统计数据，样本内统计数据，如AIC或BIC，或两者的某种组合。您可以选择对项目最有意义的指标。
- **残留诊断**。可以自动计算残差预测误差的统计数据，以提供拟合质量的附加指示。例子包括残差分布是否为高斯分布以及残差中是否存在自相关的统计检验。
- **更新模型**。ARIMA模型是从头开始为每个一步预测创建的。仔细检查API后，可以使用新的观察更新模型的内部数据，而不是从头开始重新创建。
- **先决条件**。ARIMA模型可以对时间序列数据集做出假设，例如正态性和平稳性。可以检查这些，并在训练给定模型之前针对给定的数据集引发警告。
