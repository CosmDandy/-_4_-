```python
# Python 3.11.2
# %pip install -r requriments.txt

# Python 3.10.10
# %conda install --file requriments.txt
# %conda create --name PD --file file.txt
```


```python
from IPython.display import Image

# Импорт библиотек для работы с данными
import pandas as pd
import numpy as np
import scipy.stats as stats

# Импорт библиотек для визуализации данных
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px

# Импорт библиотек для пред обработки данных
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
from sklearn import preprocessing

# Импорт библиотек для отбора признаков
from sklearn.feature_selection import RFE, RFECV, SelectFromModel, SelectFpr, chi2

# Импорт библиотек для валидации и кросс-валидации
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import LeavePOut
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate

# Импорт библиотек линейной регрессии
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import Poisson
from statsmodels.regression.recursive_ls import RecursiveLS
from statsmodels.regression.linear_model import OLS, GLS, GLSAR, WLS
from sklearn.linear_model import Ridge, Lasso, ElasticNet, SGDRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, SGDRegressor

# Импорт библиотек для машинного обучения
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

# Импорт библиотек для оценки качества моделей
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
```

# Анализ экономики Москвы через построение и оценку параметров систем независимых, регрессионных и одновременных уравнений.

## Введение
Анализ экономики Москвы является важной задачей, которая требует построения и оценки параметров систем независимых, регрессионных и одновременных уравнений. Это позволяет получить более точное представление о взаимосвязях и влиянии различных факторов на экономические показатели города. Такой анализ может быть полезен для принятия решений в различных сферах, таких как инвестиции, управление городской инфраструктурой, развитие бизнеса и других. В данной работе мы рассмотрим методы построения и оценки параметров систем уравнений, которые могут быть применены для анализа экономики Москвы и помогут получить более глубокое понимание экономической ситуации в городе.

## Цель работы
Целью данной работы является анализ экономики Москвы через построение и оценку параметров систем независимых, регрессионных и одновременных уравнений.

## Задачи работы
Для достижения поставленной цели необходимо решить следующие задачи:
- Провести предварительный анализ данных, визуализировать их, выявить и устранить выбросы, пропуски и аномалии.
- Построить систему независимых уравнений, оценить параметры уравнений системы, проанализировать их значимость и провести отбор признаков.
- Построить систему регрессионных уравнений, оценить параметры уравнений системы, проанализировать их значимость и провести отбор признаков.
- Построить систему одновременных уравнений, оценить параметры уравнений системы, проанализировать их значимость и провести отбор признаков.
- Сравнить результаты оценки параметров систем независимых, регрессионных и одновременных уравнений.
- Сделать выводы по результатам проведенного анализа.

## Методы исследования
Для решения поставленных задач будут использованы следующие методы:
- Методы визуализации данных (гистограммы, диаграммы рассеяния, диаграммы корреляции, диаграммы боксплот, диаграммы виолончели)
- Методы предобработки данных (заполнение пропусков, удаление выбросов, нормализация данных)
- Методы отбора признаков (обратное и прямое исключение признаков)
- Методы валидации и кросс-валидации (hold-out, k-fold, leave-one-out, leave-p-out, repeated k-fold)
- Методы линейной регрессии (OLS, GLS, GLSAR, WLS, RecursiveLS, Poisson)
- Методы машинного обучения (линейная регрессия, регрессия с регуляризацией, логистическая регрессия, метод опорных векторов, метод ближайших соседей, наивный байесовский классификатор, дерево решений, случайный лес, персептрон, градиентный спуск)
- Методы оценки качества моделей (R2, MAE, MSE)

# Предварительный анализ и обработка данных

Прежде чем приступать к построению и оценке параметров систем независимых, регрессионных и одновременных уравнений, необходимо провести предварительный анализ данных, визуализировать их, выявить и устранить выбросы, пропуски и аномалии. Для этого загрузим данные и посмотрим на них. Для визуализации данных будем использовать библиотеку plotly express, а для работы с данными - библиотеку pandas.

## Загрузка данных


```python
# Загрузка данных с помощью pandas из файла csv
data = pd.read_csv("input/data.csv", delimiter=";")

# Заранее объявим столбец с годом так как он понадобится нам позже
year = data['Год']

# Просмотр первых 5 строк данных
data.head()
```

Наши данные содержат следующие признаки:

- Год - год, в котором были собраны данные

Зависимые переменные:
- $y_1$ - инвестиции в основной капитал, млн руб.
- $y_2$ - валовой региональный продукт (ВРП), млн.руб.
- $y_3$ - сумма доходов населения за год, млн руб.

Независимые переменные:
- $x_1$ - финансовый результат деятельности (чистая прибыль)
- $x_2$ - прямые иностранные инвестиции, млн USD
- $x_3$ - среднегодовая численности занятых, тыс чел.
- $x_4$ - стоимость основных фондов, млн. руб
- $x_5$ - степень износа основных фондов, %
- $x_6$ - затраты на научные исследования и разработки, млн руб.
- $x_7$ - объём инновационных товаров работ услуг, млн руб.
- $x_8$ - экспорт, млн USD
- $x_9$ - импорт, млн. USD
- $x_{10}$ - сумма остатков вкладов на счетах в Банке России, млн. руб.
- $x_{11}$ - прожиточный минимум в регионе РФ (г. Москва), тыс.руб.


```python
data.shape
```

На данном этапе мы видим, что в наших данных 15 признаков (включая год) и 25 наблюдение.

## Изучение данных на наличие пропусков, нулевых значений и ошибок измерений

Посмотрим имеются ли в данных пропуски и нулевые значения. Для этого воспользуемся методом isna() и sum().


```python
pd.concat([data.isna().sum(), data.isnull().sum()], axis=1, keys=['isna', 'isnull'])
```

Пропусков и нулевых значений в данных нет, так как нами заранее была проведена очистка и обработка данных на которую мы потратили достаточно много времени.

Подробно на этом этапе мы останавливаться не будем, так как это не является целью данной работы. Если обобщить, то при обработке данных были удалены ненужные признаки, а также были заполнены пропуски, удалены выбросы и аномалии, исправлены ошибки измерения.

Далее опишем функцию, которая поможет нам визуализировать все имеющиеся данные, а так же является мощным инструментом для поиска ошибок измерения (по графикам рассеяния можно увидеть наличие выбросов и аномалий).

Функция ```combine_scatter_plots``` строит графики рассеяния для каждой пары признаков в наборе данных. Это позволяет нам визуально оценить наличие ошибок измерений. Если ошибок нет, то точки на графике должны располагаться вблизи линии тренда. Если же точки расположены вдали от линии тренда, то это может говорить о наличии ошибок измерений. В таком случае необходимо исключить из набора данных некорректные значения.


```python
def combine_scatter_plots(height=1000, width=1100, isyear=True):
    fig = make_subplots(rows=5, cols=3)
    data_copy = data.copy()
    if isyear:
        data_copy.drop('Год', axis=1, inplace=True)
    for i, column in enumerate(data_copy.columns.values):
        if 0 <= i < 3:
            row, col = 1, i + 1
        elif 3 <= i < 6:
            row, col = 2, i + 1 - 3
        elif 6 <= i < 9:
            row, col = 3, i + 1 - 6
        elif 9 <= i < 12:
            row, col = 4, i + 1 - 9
        else:
            row, col = 5, i + 1 - 12

        if 0 <= i < 3:
            fig.add_trace(go.Scatter(y=data_copy[f'y{i + 1}'], x=year, name=column), row=row, col=col)
        else:
            fig.add_trace(go.Scatter(y=data_copy[f'x{i - 2}'], x=year, name=column), row=row, col=col)
    fig.update_layout(height=height, width=width, title_text="Зависимость признаков от времени")
    fig.show()
```


```python
combine_scatter_plots()
```

На графиках рассеяния видно, что в большинстве случаев точки расположены вблизи линии тренда (к примеру $y_1$, $y_2$), что говорит об отсутствии ошибок измерений, но есть несколько признаков с достаточно сильным отклонением (к примеру $x_2$, $x_5$), несмотря на это мы не будем их удалять, так как в этих случаях данные и правда являются корректными.

## Проверка данных на наличие выбросов и изучение распределения данных

Нахождение выбросов - это один из методов анализа данных, который используется для выявления аномальных значений в наборе данных. Выбросы могут возникать из-за ошибок в данных, случайных вариаций или из-за наличия редких, но реальных экстремальных значений.

Существует несколько методов нахождения выбросов, в том числе метод Z-оценки и метод межквартильного размаха (IQR).

**Межквартильный размах (IQR)**

Метод межквартильного размаха (IQR) основан на вычислении интерквартильного размаха - разницы между 75-перцентилью и 25-перцентилем. Затем определяется верхняя и нижняя границы выбросов, которые определяются как 1,5 межквартильных размаха за пределами верхнего и нижнего квартилей.

$$(Q_1 - 1.5 \cdot IQR, Q_3 + 1.5 \cdot IQR)$$


```python
# Находим Первый квартиль (Q1) и Третий квартиль (Q3) и рассчитываем межквартильный размах
Q1 = data.quantile(q=.25)
Q3 = data.quantile(q=.75)
IQR = data.apply(stats.iqr)

# Оставляем только те значения, которые больше нижней границы и меньше верхней границы.
data_clean = data[~((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))).any(axis=1)]
data_clean.shape
```

**Z-оценка**

Метод Z-оценки основан на расчете Z-оценки для каждого значения в наборе данных и проверке, насколько далеко каждое значение от среднего значения. Значения, которые находятся на расстоянии больше чем три стандартных отклонения от среднего, могут быть считаны выбросами.

$$z = \frac{x - \mu}{\sigma}$$


```python
z = np.abs(stats.zscore(data))
data_clean = data[(z < 3).all(axis=1)]
data_clean.shape
```

В результате проверки на наличие выбросов мы получили исходный набор данных. Это говорит нам о том что выбросов в данных найденно не было.


```python
# Заменяем исходный набор данных на очищенный
data = data_clean
```

Еще одним мощными инстурментом для проверки данных на наличие выбросов и аномалий являются ящиковые диаграммы и диаграммы скрипок. Построим их для каждого признака в наборе данных. Для этого реализуем функции ```combine_box_plots``` и ```combine_violin_plots```.

Функция combine_box_plots строит ящиковые диаграммы для каждого признака в наборе данных, где значения признака разбиваются на квартили и представляются в виде ящика. Ящик показывает границы первого и третьего квартилей, а также медиану. Черта внутри ящика показывает медиану, а концы усов - границы первого и третьего квартилей. Точки, выходящие за границы усов, считаются выбросами. Также на графике отображаются все значения признака.


```python
# Построение графиков
def combine_box_plots(height=1000, width=1100):
    fig = make_subplots(rows=5, cols=3)
    data_copy = data.copy()
    data_copy.drop('Год', axis=1, inplace=True)
    for i, column in enumerate(data_copy.columns.values):
        if 0 <= i < 3:
            row, col = 1, i + 1
        elif 3 <= i < 6:
            row, col = 2, i + 1 - 3
        elif 6 <= i < 9:
            row, col = 3, i + 1 - 6
        elif 9 <= i < 12:
            row, col = 4, i + 1 - 9
        else:
            row, col = 5, i + 1 - 12

        if 0 <= i < 3:
            fig.add_trace(go.Box(x=data_copy[f'y{i + 1}'], name=column), row=row, col=col)
        else:
            fig.add_trace(go.Box(x=data_copy[f'x{i - 2}'], name=column), row=row, col=col)
    fig.update_layout(height=height, width=width, title_text="Ящиковые диаграммы распределения признаков",
                      showlegend=False)
    fig.show()
```


```python
combine_box_plots()
```

Функция combine_violin_plots строит аналогичные диаграммы, но вместо ящика используется график, который представляет **распределение** значений признака с помощью ядерной оценки плотности.


```python
def combine_violin_plots(height=1500, width=1100):
    fig = make_subplots(rows=5, cols=3)
    data_copy = data.copy()
    data_copy.drop('Год', axis=1, inplace=True)
    for i, column in enumerate(data_copy.columns.values):
        if 0 <= i < 3:
            row, col = 1, i + 1
        elif 3 <= i < 6:
            row, col = 2, i + 1 - 3
        elif 6 <= i < 9:
            row, col = 3, i + 1 - 6
        elif 9 <= i < 12:
            row, col = 4, i + 1 - 9
        else:
            row, col = 5, i + 1 - 12

        if 0 <= i < 3:
            fig.add_trace(go.Violin(x=data_copy[f'y{i + 1}'], name=column), row=row, col=col)
        else:
            fig.add_trace(go.Violin(x=data_copy[f'x{i - 2}'], name=column), row=row, col=col)
    fig.update_layout(height=height, width=width, title_text="Распределение признаков", showlegend=False)
    fig.show()
```


```python
combine_violin_plots()
```

Как мы можем видеть, в данных нет выбросов и аномалий. Также мы можем заметить, что распределение значений признаков не является нормальным. Имеются как унимодально распределенные признаки, так и многомодально распределенные признаки.

## Интерполяция данных

В предыдущем разделе мы убедились, что в наших данных нет пропусков и нулевых значений. Однако, мы имеем всего лишь 25 наблюдений, что является недостаточным количеством для построения моделей, так как это приводит к большим ошибкам и переобучению модели. Поэтому нам необходимо увеличить количество наблюдений. Для этого мы будем использовать метод интерполяции данных.

Для этого создадим класс ```InterpolateData```, который будет принимать на вход данные и метод интерполяции. По умолчанию метод интерполяции равен 'linear', а частота - 'Q' (квартальная). Для интерполяции данных мы будем использовать метод interpolate() из библиотеки pandas.


```python
class InterpolateData:
    def __init__(self, data, method='linear', freq='Q'):
        self.data = data.copy()
        self.year = pd.to_datetime(self.data['Год'].astype(str), format='%Y')
        self.data_quarterly = data.copy()
        self.data_quarterly['Год'] = pd.to_datetime(self.data_quarterly['Год'].astype(str), format='%Y')
        self.data_quarterly.set_index('Год', inplace=True)
        self.data_quarterly = self.data_quarterly.resample(freq).mean().interpolate(method=method)

    def head(self):
        return self.data_quarterly.head()

    def scatter(self):
        fig = px.scatter(self.data_quarterly)
        fig.update_layout(title_text="Данные после интерполяции", showlegend=True)
        return fig.show()

    def original_scatter(self):
        fig = px.scatter(self.data.drop(columns=['Год']), x=self.year, y=self.data.drop(columns=['Год']).columns)
        fig.update_layout(title_text="Данные до интерполяции", showlegend=True)
        return fig.show()

    def after_interpolate(self):
        return self.data_quarterly
```

В нашей работе мы будем использовать метод интерполяции под наименованием 'akima'. Данный метод является одним из наиболее точных методов интерполяции. Для наглядности построим графики рассеяния до и после интерполяции.

### Данные до интерполяции


```python
InterpolateData(data).original_scatter()
```

### Данные после интерполяции с помощью метода 'akima'


```python
# InterpolateData(data).scatter()
# InterpolateData(data, method='cubic').scatter()
# InterpolateData(data, method='quadratic').scatter()
InterpolateData(data, method='akima').scatter()
```


```python
data = InterpolateData(data, method='akima').after_interpolate()
data.head()
```


```python
data.shape
```

После интерполяции данных мы получили $\sim$100 наблюдений (было 25), что является достаточным количеством для построения моделей. Так же мы удалили столбец 'Год', так как он больше не нужен. Снова посмотри на то как распределены признаки. После этого перейдем к избавлению от мультиколлинеарности.


```python
combine_scatter_plots(isyear=False)
```

## Избавление от мультиколлинеарности между признаками путем нормализации и центрирования данных

Мультиколлинеарность - это явление, при котором два или более предикторов взаимосвязаны между собой. Это может привести к нестабильности оценок и усложнить интерпретацию результатов. Чтобы избежать этого, можно использовать нормализацию и центрирование данных. В данном случае была использована нормализация данных.

В нашем случае мы будем использовать следующие методы нормализации данных:
- Центрирование данных
- Минимаксная нормализация данных
- Z-нормализация данных
- Max-нормализация данных
- Robust-нормализация данных

### Центрирование данных для избавления от мультиколлинеарности
Этот способ приводит все значения признаков к среднему значению 0. Это делается путем вычитания среднего значения из каждого значения признака.
$$
x_{i}^{*}=x_{i}-\bar{x}
$$


```python
# data = data.apply(lambda x: x - x.mean())
```

### Минимаксная нормализация данных
Этот метод приводит все значения признаков к диапазону от 0 до 1. Это делается путем вычитания минимального значения и деления на разницу между максимальным и минимальным значениями. Именно этот метод нормализации данных мы будем использовать в нашей работе, как основным.

$$
x_{i}^{*}=\frac{x_{i}-\min (x)}{\max (x)-\min (x)}
$$


```python
data = data.apply(lambda x: (x - x.min()) / (x.max() - x.min()))
```

### Нормализация средним (Z-нормализация)
Этот метод приводит все значения признаков к диапазону от -1 до 1. Это делается путем вычитания среднего значения и деления на стандартное отклонение.
$$
x_{i}^{*}=\frac{x_{i}-\bar{x}}{\sigma}
$$


```python
# data = data.apply(lambda x: (x - x.mean()) / x.std())
```

### MaxAbsScaler
Этот метод приводит все значения признаков к диапазону от -1 до 1. Это делается путем деления на максимальное абсолютное значение.
$$
x_{i}^{*}=\frac{x_{i}}{\max \left(\left|x_{i}\right|\right)}
$$


```python
# transformer = MaxAbsScaler().fit(data)
# data = pd.DataFrame(transformer.transform(data))
```

### RobustScaler
Этот метод приводит все значения признаков к диапазону от -1 до 1. Это делается путем вычитания медианы и деления на интерквартильный размах.
$$
x_{i}^{*}=\frac{x_{i}-\operatorname{med}(x)}{IQR}
$$


```python
# transformer = RobustScaler().fit(data)
# data = pd.DataFrame(transformer.transform(data))
```


```python
data.columns = ['y1', 'y2', 'y3', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11']
data.head()
```


```python
data.describe()
```

Как мы видим из таблицы, все признаки находятся в диапазоне от 0 до 1. Так же мы видим, что среднее значение признаков близко к 0.5, а стандартное отклонение близко к 0.3. Это говорит о том, что данные нормализованы правильно. Это позволило нам избавиться от мультиколлинеарности между признаками.

## Добавление шума в данные

Для объяснения добавления шума в данные, рассмотрим следующий график. На нем изображены зависимости признаков от времени.


```python
# px.scatter(data, title="Зависимость признаков от времени", height=500)
px.scatter(data['y1'], title="Зависимость признака Y1 от времени до добавления шума", height=500)
```

Как видно из графика, данные имеют очень сильную линейную структуру, из-за этого мы получаем сверх высокие метрики при обучении модели. Добавление шума в данные, позволяет сделать данные более реалистичными, так как в реальных данных всегда присутствует шум. Так же добавление шума в данные позволяет сделать данные более устойчивыми к выбросам.


```python
np.random.seed(101)
noise = np.random.normal(0, .06, data.shape)
noisy_data = data + noise
data = noisy_data
data = data.apply(lambda x: (x - x.min()) / (x.max() - x.min()))
px.scatter(data['y1'], title="Зависимость признака Y1 от времени после добавления шума", height=500)
```


```python
combine_scatter_plots(isyear=False)
```

Теперь наши данные выглядят более реалистично. Так же мы видим, что данные стали более устойчивыми к выбросам.

## Построение матрицы коэффициентов межфакторной корреляции


```python
fig = px.imshow(data.corr(), text_auto='.2f', color_continuous_scale="gnbu",
                labels={'color': 'R', 'x': 'Фактор', 'y': 'Фактор'})
fig.update_layout(height=1000, title_text="Матрица коэффициентов межфакторной корреляции")
fig.show()
```

Можно заметить, что между некоторыми признаками есть сильная линейная зависимость. Но при этом большая часть признаков слабо коррелирует между собой.


```python
un_data = data.copy()
un_data.columns = ['Инвестиции в основной капитал, млн руб. y1', 'Валовой региональный продукт (ВРП), млн.руб. y2',
                   'Сумма доходов населения за год, млн руб. y3',
                   'Финансовый результат деятельности (чистая прибыль), млн.руб. x1',
                   'Прямые иностранные инвестиции, млн USD x2',
                   'Среднегодовая численности занятых, тыс чел. x3', 'Стоимость основных фондов, млн. руб. x4',
                   'Степень износа основных фондов, % x5',
                   'Затраты на научные исследования и разработки, млн руб. x6',
                   'Объём инновационных товаров работ услуг, млн руб. x7', 'Экспорт, млн USD x8',
                   'Импорт, млн. USD x9', 'Сумма остатков вкладов на счетах в Банке России, млн. руб. x10',
                   'Прожиточный минимум в регионе РФ (г. Москва), тыс.руб. x11']

fig = px.imshow(un_data.corr(), text_auto='.2f', color_continuous_scale="gnbu",
                labels={'color': 'R', 'x': 'Фактор', 'y': 'Фактор'})
fig.update_layout(height=1500, width=1500, title_text="Матрица коэффициентов межфакторной корреляции")
fig.show()
```

# Разделение данных на обучающую и тестовую выборки. Валидация и кросс-валидация

Для дальнейшего обучения модели, данные разделяются на обучающую и тестовую выборки. Обучающая выборка - это часть данных, на которых модель будет обучаться.


```python
X = data.drop(['y1', 'y2', 'y3'], axis=1)
y = data[['y1', 'y2', 'y3']]

X.shape, y.shape
```

Для упрощения написания коды заранее разделим данный на зависимые и независимые переменные.

## Валидация на отложенных данных (Hold-Out Validation)
**Валидация на отложенных данных (Hold-Out Validation)** - способ валидации при котором данные разделяются на две выборки: обучающую и тестовую. В этом методе, модель обучается на обучающей выборке данных, а затем проверяется на тестовой выборке, которая содержит данные, не использованные при обучении модели. Обычно, обучающая выборка составляет 70-80% от исходных данных, а тестовая - оставшиеся 20-30%.


```python
Image('https://i0.wp.com/neptune.ai/wp-content/uploads/2022/10/Cross-validation-hold-out.jpg', width=600)
```


```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
```

## Кросс-валидация (Cross-validation)
**Кросс-валидация (Cross-validation)** – способ [[Валидация (Validating)|валидации]] при котором данные разбиваются на несколько равных частей, называемых "складками" (folds). Затем каждая складка используется в качестве тестовой выборки, а оставшиеся складки - в качестве обучающей выборки. Таким образом, модель обучается и тестируется на каждой из складок, что позволяет получить оценку ее производительности на всех данных.

Этот метод обеспечивает более стабильные оценки производительности, чем [[Валидация на отложенных данных (Hold-Out Validation)|метод отложенной выборки]], но может быть затратным с точки зрения вычислительных ресурсов.


```python
Image('https://i0.wp.com/neptune.ai/wp-content/uploads/2022/10/Cross-validation-k-fold.jpg', width=600)
```


```python
# kf = KFold(n_splits=5, shuffle=True, random_state=42)
#
# for train_index, test_index in kf.split(X):
#     print("TRAIN:", train_index, "TEST:", test_index)
#     X_train, X_test = X.iloc[train_index], X.iloc[test_index]
#     y_train, y_test = y.iloc[train_index], y.iloc[test_index]

# write a function to get cross validation scores
```

### Leave-One-Out Cross-validation (LOO/LOOCV)
**Leave-One-Out Cross-validation (LOO/LOOCV)** - это частный случай кросс-валидации, при котором количество складок равно количеству наблюдений в данных. Таким образом, каждое наблюдение используется в качестве тестовой выборки, а оставшиеся наблюдения - в качестве обучающей выборки. Этот метод обеспечивает наиболее точную оценку производительности модели, но может быть очень затратным с точки зрения вычислительных ресурсов.


```python
Image('https://i0.wp.com/neptune.ai/wp-content/uploads/2022/10/Cross-validation-leave-one-out.jpg', width=600)
```


```python
# loo = LeaveOneOut()
#
# for train_index, test_index in loo.split(X):
#     print("TRAIN:", train_index, "TEST:", test_index)
#     X_train, X_test = X.iloc[train_index], X.iloc[test_index]
#     y_train, y_test = y.iloc[train_index], y.iloc[test_index]
```

### Leave-p-out cross-validation (LpOC)
**Leave-p-out cross-validation (LpOC)** - почти полностью повторяет метод Leave-One-Out cross-validation (LOO/LOOCV) и отличается лишь тем что изначально задается размер тестовой выборки в то время как в Leave-One-Out cross-validation он всегда равен 1.


```python
# lpo = LeavePOut(p=5)

# for train_index, test_index in lpo.split(X):
#     print("TRAIN:", train_index, "TEST:", test_index)
#     X_train, X_test = X.iloc[train_index], X.iloc[test_index]
#     y_train, y_test = y.iloc[train_index], y.iloc[test_index]
```

В итоге получаем следующие размеры выборок:


```python
X_train.shape, X_test.shape, y_train.shape, y_test.shape
```

# Построение модели линейной регрессии

Линейная регрессия - это метод анализа данных, который используется для описания и прогнозирования линейной связи между зависимой и одной или несколькими независимыми переменными. Он часто используется в статистическом анализе данных для построения математической модели, которая может быть использована для прогнозирования значений зависимой переменной на основе значений независимых переменных.

Для построения модели линейной регрессии необходимо иметь набор данных, который состоит из пар значений зависимой и независимой переменных. Затем на основе этих данных проводится анализ, который позволяет оценить параметры модели - коэффициенты наклона и пересечения. Эти параметры могут быть использованы для построения уравнения линейной регрессии.

Для оценки качества модели линейной регрессии используются различные статистические показатели, такие как коэффициент детерминации (R-квадрат) или корреляционный коэффициент. Они позволяют оценить, насколько хорошо модель соответствует данным и насколько точно она может использоваться для прогнозирования значений зависимой переменной.

Важно понимать, что линейная регрессия является моделью, которая описывает только линейные отношения между переменными. Если связь между переменными не является линейной, то модель линейной регрессии может оказаться неэффективной и требовать использования других методов анализа данных.


```python
class CustomLinearRegression:
    def __init__(self, y, x, model_type=OLS):
        self.model_type = model_type
        self.y = y
        self.x = x
        self.model = self.model_type(self.y, self.x).fit()

    def get_coefficients(self):
        print(self.model.params)
        answer = []
        names = self.x.columns.values.tolist()
        for i, par in enumerate(self.model.params):
            answer.append(f'+ {par:.4f}{names[i][0]}_{{{names[i][1:]}}}')
        print(f'$$\hat {self.y.name[0]}_{self.y.name[1]} = {" ".join(answer)[2:]}$$')

    def summary(self):
        print(self.model.summary())

    def pre(self, X_test):
        return self.model.predict(X_test)

    def predict(self, X_test, y_test):
        r2 = r2_score(y_test, self.model.predict(X_test))
        mae = mean_absolute_error(y_test, self.model.predict(X_test))
        mse = mean_squared_error(y_test, self.model.predict(X_test))
        print(
            f'Коэффициент детерминации (R^2): {r2 * 100:.1f}%\nCредняя ошибка аппроксимации (MAE): {mae * 100:.1f}%\nCредняя квадратичная ошибка (MSE): {mse * 100:.1f}%')

    def F(self):
        return self.model.fvalue

    def current_p(self, index):
        return self.model.f_pvalue[index]

    def forward_selection(self):
        xi = []
        stop = False
        x_new = []  # список переменных, которые будут включены в модель
        x_len = len(self.x.columns) - 1  # количество столбцов в датафрейме
        x_test = []
        for n in range(x_len):
            F_max = 0  # максимальное значение F
            p_max = 0  # максимальное значение p
            F_max_i = 0
            F_max_x = ''
            if not stop:
                for i in range(x_len):
                    if i not in xi:
                        x_test = x_new.copy()
                        x_test.append(self.x.columns[i])
                        F = CustomLinearRegression(self.y, self.x.loc[:, x_test]).F()
                        print(f"'x{i + 1}',", F)
                        if F > F_max:
                            F_max = F
                            F_max_i = i
                            F_max_x = f"x{i + 1}"
                x_new.append(F_max_x)
                xi.append(F_max_i)
                print(x_new)
                for j in range(len(x_new)):
                    p = CustomLinearRegression(self.y, self.x.loc[:, x_new]).current_p(j)
                    if p > 0.05:
                        stop = True
                        print(f'P-value: {p}')
                        break
        CustomLinearRegression(self.y, self.x.loc[:, x_new[:-1]]).summary()

    def backward_elimination(self, p=0.05, start=0):
        regression = self.model_type(self.y, self.x).fit()
        futures_num = len(self.x.columns)
        for i in range(futures_num):
            regression = self.model_type(self.y, self.x).fit()
            max_p = max(regression.pvalues.values[start:])
            if max_p > p:
                for j in range(0, futures_num - i):
                    if regression.pvalues[j] == max_p:
                        self.x = self.x.drop(self.x.columns[[j]], axis=1)
        print(regression.summary())

    def custom_feature_selection(self, x, y, model=LinearRegression(), selection_method=RFE):
        result_list = list()
        result = selection_method(model).fit(x, y)
        for i, feature in enumerate(result.get_support()):
            if feature:
                result_list.append(f'x{i + 1}')
        CustomLinearRegression(y, x[result_list]).summary()

    def correlation_map(self):
        return px.imshow(self.x.corr(), text_auto='.2f', color_continuous_scale="gnbu")


class CustomLinearRegressionWithConst(CustomLinearRegression):
    def __init__(self, y, x, model_type=OLS):
        super().__init__(y, x, model_type)
        self.x = sm.add_constant(self.x)
        self.model = self.model_type(self.y, self.x).fit()

    def backward_elimination(self, p=0.05, start=1):
        regression = self.model_type(self.y, self.x).fit()
        futures_num = len(self.x.columns)
        for i in range(futures_num):
            self.x = sm.add_constant(self.x)
            regression = self.model_type(self.y, self.x).fit()
            max_p = max(regression.pvalues.values[start:])
            if max_p > p:
                for j in range(0, futures_num - i):
                    if regression.pvalues[j] == max_p:
                        self.x = self.x.drop(self.x.columns[[j]], axis=1)
        print(regression.summary())


class CustomLinearRegressionRegularized(CustomLinearRegression):
    def __init__(self, y, x, model_type=OLS, alpha=0.01, L1_wt=0.05):
        super().__init__(y, x, model_type)
        self.alpha = alpha
        self.L1_wt = L1_wt
        self.model = sm.OLS(self.y, self.x).fit_regularized(alpha=self.alpha, L1_wt=self.L1_wt, refit=True)

    def backward_elimination(self, p=0.05, start=0):
        regression = self.model_type(self.y, self.x).fit_regularized(alpha=self.alpha, L1_wt=self.L1_wt, refit=True)
        features_num = len(self.x.columns)
        for i in range(features_num):
            regression = self.model_type(self.y, self.x).fit_regularized(alpha=self.alpha, L1_wt=self.L1_wt, refit=True)
            max_p = max(regression.pvalues[start:])
            if max_p > p:
                for j in range(features_num - i):
                    if regression.pvalues[j] == max_p:
                        self.x = self.x.drop(self.x.columns[[j]], axis=1)
        print(regression.summary())

# class CustomRidgeLinearRegression(CustomLinearRegression):
#     def __init__(self, y, x, model_type=OLS):
#         super().__init__(y, x, model_type)
#         self.model = Ridge(alpha=0.1).fit(self.x, self.y)
```

# Отбор переменных для модели линейной регрессии

Отбор переменных для модели линейной регрессии - это процесс выбора наиболее важных и значимых независимых переменных, которые будут использоваться в модели для прогнозирования зависимой переменной. Это важный шаг в построении модели, так как использование неподходящих переменных может привести к низкой точности прогнозов и низкому качеству модели.

Существует несколько методов отбора переменных для модели линейной регрессии, включая:

- Метод пошагового отбора переменных - это метод, который включает или исключает независимые переменные в модель пошагово на основе их значимости. На каждом шаге происходит выбор переменной, которая наиболее улучшает качество модели.

- Метод отбора на основе значимости - этот метод основан на анализе значимости каждой независимой переменной с помощью статистических тестов, таких как t-тест или F-тест. Переменные с низким уровнем значимости исключаются из модели.

- Метод регуляризации - это метод, который добавляет штрафы за большие значения коэффициентов в модель. Он позволяет автоматически отбирать переменные, которые наиболее важны для модели, и уменьшает вероятность переобучения.

- Метод отбора на основе информационных критериев - этот метод основан на использовании различных информационных критериев, таких как AIC (критерий Акаике) и BIC (критерий Шварца), чтобы выбрать модель с наименьшей ошибкой.

Выбор метода отбора переменных зависит от конкретной задачи и объема доступных данных. В нашей работе мы будем использовать метод пошагового отбора переменных и метод отбора на основе значимости.

Для наглядности снова посмотрим на данные, которые мы будем использовать для построения модели линейной регрессии.


```python
data.head()
```

Как было ранее сказано, мы имеем дело с набором данных, содержащим 11 независимых переменных и 3 зависимых переменных. Переменные x1-x11 являются независимыми переменными, а переменные y1-y3 являются зависимыми переменными. Разделим данные на независимые и зависимые переменные.

### Разделение данных на зависимые и независимые переменные

Зависимые переменные - это переменные, значения которых мы хотим предсказать. В нашем случае это переменные $y_1$, $y_2$ и $y_3$.


```python
y1_test, y1_train = y_test['y1'], y_train['y1']
y2_test, y2_train = y_test['y2'], y_train['y2']
y3_test, y3_train = y_test['y3'], y_train['y3']
```

Независимые переменные - это переменные, значения которых мы используем для прогнозирования зависимых переменных. В нашем случае это переменные $[x_1, \cdots, x_{11}]$.


```python
x1_test, x1_train = X_test['x1'], X_train['x1']
x2_test, x2_train = X_test['x2'], X_train['x2']
x3_test, x3_train = X_test['x3'], X_train['x3']
x4_test, x4_train = X_test['x4'], X_train['x4']
x5_test, x5_train = X_test['x5'], X_train['x5']
x6_test, x6_train = X_test['x6'], X_train['x6']
x7_test, x7_train = X_test['x7'], X_train['x7']
x8_test, x8_train = X_test['x8'], X_train['x8']
x9_test, x9_train = X_test['x9'], X_train['x9']
x10_test, x10_train = X_test['x10'], X_train['x10']
x11_test, x11_train = X_test['x11'], X_train['x11']
```

## Оценка предварительно составленных систем уравнений

Всего у нас три системы уравнений, которые мы можем использовать для прогнозирования зависимых переменных
- Система независимых уравнений;
- Система рекурсивных уравнений;
- Система одновременных уравнений.

### Система независимых уравнений
$$
\begin{cases}
y_1 = a_1 + b_{11}x_{1} + b_{12}x_{2} + \epsilon \\
y_2 = a_2 + b_{21}x_{1} + b_{22}x_{2} + b_{23}x_{3} + b_{24}x_{4} + b_{25}x_{5} + b_{26}x_{6} + b_{27}x_{7} + b_{28}x_{8} + b_{29}x_{9} + \epsilon \\
y_3 = a_3 + b_{31}x_{1} + b_{32}x_{2} + b_{33}x_{3} + b_{34}x_{4} + b_{35}x_{5} + b_{36}x_{6} + b_{37}x_{7} + b_{38}x_{8} + b_{39}x_{9} + b_{310}x_{10} + b_{311}x_{11} + \epsilon \\
\end{cases}
$$

### Система рекурсивных уравнений
$$
\begin{cases}
y_1 = a_1 + b_{11}x_{1} + b_{12}x_{2} + \epsilon \\
y_2 = a_2 + a_{21}y_1 + b_{21}x_{3} + b_{22}x_{4} + b_{23}x_{5} + b_{24}x_{6} + b_{25}x_{7} + b_{26}x_{8} + b_{27}x_{9} + \epsilon \\
y_3 = a_3 + a_{31}y_1 + a_{32}y_2 + b_{31}x_{11} + \epsilon \\
\end{cases}
$$

### Система одновременных уравнений
$$
\begin{cases}
y_1 = a_1 + b_{11}x_{1} + b_{12}x_{2} + \epsilon \\
y_2 = a_2 + a_{21}y_1 + b_{23}x_{3} + b_{24}x_{4} + b_{25}x_{5} + b_{26}x_{6} + b_{27}x_{7} + b_{28}x_{8} + b_{29}x_{9} + \epsilon \\
y_3 = a_3 + a_{31}y_1 + b_{310}x_{10} + b_{311}x_{11} + \epsilon \\
\end{cases}
$$

## Система независимых уравнений

$$
\begin{cases}
y_1 = a_1 + b_{11}x_{1} + b_{12}x_{2} + \epsilon \\
y_2 = a_2 + b_{21}x_{1} + b_{22}x_{2} + b_{23}x_{3} + b_{24}x_{4} + b_{25}x_{5} + b_{26}x_{6} + b_{27}x_{7} + b_{28}x_{8} + b_{29}x_{9} + \epsilon \\
y_3 = a_3 + b_{31}x_{1} + b_{32}x_{2} + b_{33}x_{3} + b_{34}x_{4} + b_{35}x_{5} + b_{36}x_{6} + b_{37}x_{7} + b_{38}x_{8} + b_{39}x_{9} + b_{310}x_{10} + b_{311}x_{11} + \epsilon \\
\end{cases}
$$

### Первое уравнение

$$y_1 = a_1 + b_{11}x_{1} + b_{12}x_{2} + \epsilon $$

#### Исходная система уравнений


```python
fn1_nez = pd.concat([x1_train, x2_train], axis=1)
CustomLinearRegression(y1_train, fn1_nez).summary()
```

#### Исходная система уравнений после отбора признаков методом обратного исключения (Backward elimination)


```python
CustomLinearRegression(y1_train, fn1_nez).backward_elimination()
```

#### Конечная система уравнений и веса


```python
final_fn1_nez = pd.concat([x1_train, x2_train], axis=1)
CustomLinearRegression(y1_train, final_fn1_nez).get_coefficients()
```

$$\hat y_1 = 0.5478x_{1} + 0.2297x_{2}$$


```python
fig = make_subplots(rows=1, cols=2)
fig.update_yaxes(title_text='Y1', row=1, col=1)

fig.add_trace(px.scatter(x=x1_train, y=y1_train).data[0], row=1, col=1)
fig.update_xaxes(title_text='X1', row=1, col=1)

fig.add_trace(px.scatter(x=x2_train, y=y1_train).data[0], row=1, col=2)
fig.update_xaxes(title_text='X2', row=1, col=2)

fig.update_layout(title_text="График рассеяния для X1 и X2 от Y1")
fig.show()
```

### Второе уравнение

$$y_2 = a_2 + b_{21}x_{1} + b_{22}x_{2} + b_{23}x_{3} + b_{24}x_{4} + b_{25}x_{5} + b_{26}x_{6} + b_{27}x_{7} + b_{28}x_{8} + b_{29}x_{9} + \epsilon$$

#### Исходная система уравнений


```python
fn2_nez = pd.concat([x1_train, x2_train, x3_train, x4_train, x5_train, x6_train, x7_train, x8_train, x9_train], axis=1)
CustomLinearRegression(y2_train, fn2_nez).summary()
```

#### Исходная система уравнений после отбора признаков методом обратного исключения (Backward elimination)


```python
CustomLinearRegression(y2_train, fn2_nez).backward_elimination()
```

#### Конечная система уравнений и веса


```python
final_fn2_nez = pd.concat([x1_train, x4_train, x6_train, x8_train], axis=1)
CustomLinearRegression(y2_train, final_fn2_nez).get_coefficients()
```

$$\hat y_2 = 0.1732x_{1} + 0.3493x_{4} + 0.3101x_{6} + 0.1358x_{8}$$


```python
fig = make_subplots(rows=2, cols=2)
fig.update_yaxes(title_text='Y2', row=1, col=1)
fig.update_yaxes(title_text='Y2', row=2, col=1)

fig.add_trace(px.scatter(x=x1_train, y=y2_train).data[0], row=1, col=1)
fig.update_xaxes(title_text='X1', row=1, col=1)

fig.add_trace(px.scatter(x=x4_train, y=y2_train).data[0], row=1, col=2)
fig.update_xaxes(title_text='X4', row=1, col=2)

fig.add_trace(px.scatter(x=x6_train, y=y2_train).data[0], row=2, col=1)
fig.update_xaxes(title_text='X6', row=2, col=1)

fig.add_trace(px.scatter(x=x8_train, y=y2_train).data[0], row=2, col=2)
fig.update_xaxes(title_text='X8', row=2, col=2)

fig.update_layout(height=800, title='Графики рассеяния для X1, X4, X6, X8 от Y2')
fig.show()
```

### Третье уравнение

$$y_3 = a_3 + b_{31}x_{1} + b_{32}x_{2} + b_{33}x_{3} + b_{34}x_{4} + b_{35}x_{5} + b_{36}x_{6} + b_{37}x_{7} + b_{38}x_{8} + b_{39}x_{9} + b_{310}x_{10} + b_{311}x_{11} + \epsilon$$

#### Исходная система уравнений


```python
fn3_nez = pd.concat(
    [x1_train, x2_train, x3_train, x4_train, x5_train, x6_train, x7_train, x8_train, x9_train, x10_train, x11_train],
    axis=1)
CustomLinearRegression(y3_train, fn3_nez).summary()
```

#### Исходная система уравнений после отбора признаков методом обратного исключения (Backward elimination)


```python
CustomLinearRegression(y3_train, fn3_nez).backward_elimination()
```

#### Конечная система уравнений и веса


```python
final_fn3_nez = pd.concat([x3_train, x4_train, x10_train, x11_train], axis=1)
CustomLinearRegression(y3_train, final_fn3_nez).get_coefficients()
```

$$\hat y_3 = 0.2826x_{3} + 0.2021x_{4} + 0.2524x_{10} + 0.2532x_{11}$$


```python
fig = make_subplots(rows=2, cols=2)
fig.update_yaxes(title_text='Y3', row=1, col=1)
fig.update_yaxes(title_text='Y3', row=2, col=1)

fig.add_trace(px.scatter(x=x3_train, y=y3_train).data[0], row=1, col=1)
fig.update_xaxes(title_text='X3', row=1, col=1)

fig.add_trace(px.scatter(x=x4_train, y=y3_train).data[0], row=1, col=2)
fig.update_xaxes(title_text='X4', row=1, col=2)

fig.add_trace(px.scatter(x=x10_train, y=y3_train).data[0], row=2, col=1)
fig.update_xaxes(title_text='X10', row=2, col=1)

fig.add_trace(px.scatter(x=x11_train, y=y3_train).data[0], row=2, col=2)
fig.update_xaxes(title_text='X11', row=2, col=2)

fig.update_layout(height=800, title='Графики рассеяния для X3, X4, X10, X11 от Y3')
fig.show()
```

## Система рекурсивных уравнений

$$
\begin{cases}
y_1 = a_1 + b_{11}x_{1} + b_{12}x_{2} + \epsilon \\
y_2 = a_2 + a_{21}y_1 + b_{21}x_{3} + b_{22}x_{4} + b_{23}x_{5} + b_{24}x_{6} + b_{25}x_{7} + b_{26}x_{8} + b_{27}x_{9} + \epsilon \\
y_3 = a_3 + a_{31}y_1 + a_{32}y_2 + b_{31}x_{11} + \epsilon \\
\end{cases}
$$

### Первое уравнение

$$y_1 = a_1 + b_{11}x_{1} + b_{12}x_{2} + \epsilon$$

#### Исходная система уравнений


```python
fn1_rec = pd.concat([x1_train, x2_train], axis=1)
CustomLinearRegression(y1_train, fn1_rec).summary()
```

#### Исходная система уравнений после отбора признаков методом обратного исключения (Backward elimination)


```python
CustomLinearRegression(y1_train, fn1_rec).backward_elimination()
```

#### Конечная система уравнений и веса


```python
final_fn1_rec = pd.concat([x1_train, x2_train], axis=1)
CustomLinearRegression(y1_train, final_fn1_rec).get_coefficients()
```

$$\hat y_1 = 0.5478x_{1} + 0.2297x_{2}$$


```python
fig = make_subplots(rows=1, cols=2)
fig.update_yaxes(title_text='Y1', row=1, col=1)

fig.add_trace(px.scatter(x=x1_train, y=y1_train).data[0], row=1, col=1)
fig.update_xaxes(title_text='X1', row=1, col=1)

fig.add_trace(px.scatter(x=x2_train, y=y1_train).data[0], row=1, col=2)
fig.update_xaxes(title_text='X2', row=1, col=2)

fig.update_layout(title_text="График рассеяния для X1 и X2 от Y1")
fig.show()
```

### Второе уравнение

$$y_2 = a_2 + a_{21}y_1 + b_{21}x_{3} + b_{22}x_{4} + b_{23}x_{5} + b_{24}x_{6} + b_{25}x_{7} + b_{26}x_{8} + b_{27}x_{9} + \epsilon$$

#### Исходная система уравнений


```python
fn2_rec = pd.concat(
    [y1_train, x1_train, x2_train, x3_train, x4_train, x5_train, x6_train, x7_train, x8_train, x9_train], axis=1)
CustomLinearRegression(y2_train, fn2_rec).summary()
```

#### Исходная система уравнений после отбора признаков методом обратного исключения (Backward elimination)


```python
CustomLinearRegression(y2_train, fn2_rec).backward_elimination(start=1)
```

#### Конечная система уравнений и веса


```python
final_fn2_rec = pd.concat([y1_train, x1_train, x4_train, x6_train], axis=1)
CustomLinearRegression(y2_train, final_fn2_rec).get_coefficients()
```

$$\hat y_2 = 0.2164y_{1} + 0.2852x_{1} + 0.1989x_{4} + 0.2861x_{6}$$


```python
fig = make_subplots(rows=2, cols=2)
fig.update_yaxes(title_text='Y2', row=1, col=1)
fig.update_yaxes(title_text='Y2', row=2, col=1)


fig.add_trace(px.scatter(x=y1_train, y=y2_train).data[0], row=1, col=1)
fig.update_xaxes(title_text='Y1', row=1, col=1)

fig.add_trace(px.scatter(x=x1_train, y=y2_train).data[0], row=1, col=2)
fig.update_xaxes(title_text='X1', row=1, col=2)

fig.add_trace(px.scatter(x=x4_train, y=y2_train).data[0], row=2, col=1)
fig.update_xaxes(title_text='X4', row=2, col=1)

fig.add_trace(px.scatter(x=x6_train, y=y2_train).data[0], row=2, col=2)
fig.update_xaxes(title_text='X6', row=2, col=2)

fig.update_layout(height=1000, title='Графики рассеяния для Y1, X1, X4, X6 от Y2')
fig.show()
```

### Третье уравнение

$$y_3 = a_3 + a_{31}y_1 + a_{32}y_2 + b_{31}x_{11} + \epsilon$$

#### Исходная система уравнений


```python
fn3_rec = pd.concat([y1_train, y2_train, x11_train], axis=1)
CustomLinearRegression(y3_train, fn3_rec).summary()
```

#### Исходная система уравнений после отбора признаков методом обратного исключения (Backward elimination)


```python
CustomLinearRegression(y3_train, fn3_rec).backward_elimination(start=0)
```

#### Конечная система уравнений и веса


```python
final_fn3_rec = pd.concat([y2_train, x11_train], axis=1)
CustomLinearRegression(y3_train, final_fn3_rec).get_coefficients()
```

$$\hat y_3 = 0.5732y_{2} + 0.4683x_{11}$$


```python
fig = make_subplots(rows=1, cols=2)
fig.update_yaxes(title_text='Y3', row=1, col=1)

fig.add_trace(px.scatter(x=y2_train, y=y3_train).data[0], row=1, col=1)
fig.update_xaxes(title_text='Y2', row=1, col=1)

fig.add_trace(px.scatter(x=x11_train, y=y3_train).data[0], row=1, col=2)
fig.update_xaxes(title_text='X11', row=1, col=2)

fig.update_layout(height=500, title='Графики рассеяния для Y2, X11 от Y3')
fig.show()
```

## Система одновременных уравнений

$$
\begin{cases}
y_1 = a_1 + b_{11}x_{1} + b_{12}x_{2} + \epsilon \\
y_2 = a_2 + a_{21}y_1 + b_{23}x_{3} + b_{24}x_{4} + b_{25}x_{5} + b_{26}x_{6} + b_{27}x_{7} + b_{28}x_{8} + b_{29}x_{9} + \epsilon \\
y_3 = a_3 + a_{31}y_1 + b_{310}x_{10} + b_{311}x_{11} + \epsilon \\
\end{cases}
$$

### Первое уравнение

$$y_1 = a_1 + b_{11}x_{1} + b_{12}x_{2} + \epsilon$$

#### Исходная система уравнений


```python
fn1_odn = pd.concat([x1_train, x2_train], axis=1)
CustomLinearRegression(y1_train, fn1_odn, model_type=Poisson).summary()
```

#### Исходная система уравнений после отбора признаков методом обратного исключения (Backward elimination)


```python
CustomLinearRegression(y1_train, fn1_odn, model_type=Poisson).backward_elimination()
```

#### Конечная система уравнений и веса


```python
final_fn1_odn = pd.concat([x1_train], axis=1)
CustomLinearRegression(y1_train, final_fn1_odn, model_type=Poisson).get_coefficients()
```

$$\hat y_1 = -1.2598x_{1}$$


```python
fig = make_subplots(rows=1, cols=1)
fig.update_yaxes(title_text='Y1', row=1, col=1)

fig.add_trace(px.scatter(x=x1_train, y=y1_train).data[0], row=1, col=1)
fig.update_xaxes(title_text='X1', row=1, col=1)

fig.update_layout(height=500, title='Графики рассеяния для X1 от Y1')
fig.show()
```

### Второе уравнение

$$y_2 = a_2 + a_{21}y_1 + b_{23}x_{3} + b_{24}x_{4} + b_{25}x_{5} + b_{26}x_{6} + b_{27}x_{7} + b_{28}x_{8} + b_{29}x_{9} + \epsilon$$

#### Исходная система уравнений


```python
fn2_odn = pd.concat([y1_train, x3_train, x4_train, x5_train, x6_train, x7_train, x8_train, x9_train], axis=1)
CustomLinearRegression(y2_train, fn2_odn, model_type=Poisson).summary()
```

#### Исходная система уравнений после отбора признаков методом обратного исключения (Backward elimination)


```python
CustomLinearRegression(y2_train, fn2_odn, model_type=Poisson).backward_elimination(start=1)
```

#### Конечная система уравнений и веса


```python
final_fn2_odn = pd.concat([y1_train, x5_train], axis=1)
CustomLinearRegression(y2_train, final_fn2_odn, model_type=Poisson).get_coefficients()
```

$$\hat y_2 = 1.2379y_{1} - 2.0557x_{5}$$


```python
fig = make_subplots(rows=1, cols=2)
fig.update_yaxes(title_text='Y2', row=1, col=1)

fig.add_trace(px.scatter(x=y1_train, y=y2_train).data[0], row=1, col=1)
fig.update_xaxes(title_text='Y1', row=1, col=1)

fig.add_trace(px.scatter(x=x5_train, y=y2_train).data[0], row=1, col=2)
fig.update_xaxes(title_text='X5', row=1, col=2)

fig.update_layout(height=500, title='Графики рассеяния для Y1, X5 от Y2')
fig.show()
```

### Третье уравнение

$$y_3 = a_3 + a_{31}y_1 + b_{310}x_{10} + b_{311}x_{11} + \epsilon$$

#### Исходная система уравнений


```python
fn3_odn = pd.concat([y1_train, x10_train, x11_train], axis=1)
CustomLinearRegression(y3_train, fn3_odn, model_type=Poisson).summary()
```

#### Исходная система уравнений после отбора признаков методом обратного исключения (Backward elimination)


```python
CustomLinearRegression(y3_train, fn3_odn, model_type=Poisson).backward_elimination(start=0)
```

#### Конечная система уравнений и веса


```python
final_fn3_odn = pd.concat([y1_train, x10_train], axis=1)
CustomLinearRegression(y3_train, final_fn3_odn, model_type=Poisson).get_coefficients()
```

$$\hat y_3 = 0.9046y_{1} - 1.7170x_{10}$$


```python
fig = make_subplots(rows=1, cols=2)
fig.update_yaxes(title_text='Y3', row=1, col=1)

fig.add_trace(px.scatter(x=y1_train, y=y3_train).data[0], row=1, col=1)
fig.update_xaxes(title_text='Y1', row=1, col=1)

fig.add_trace(px.scatter(x=x10_train, y=y3_train).data[0], row=1, col=2)
fig.update_xaxes(title_text='X10', row=1, col=2)

fig.update_layout(height=500, title='Графики рассеяния для Y1, X10 от Y3')
fig.show()
```

# Прогнозирование

На данный момент мы прошли практически все этапы построения модели. Осталось только прогнозирование. Для этого нам необходимо построить системы уравнений, которые будут состоять из уравнений, полученных путем отбора параметров на предыдущих этапах работы. После этого мы сможем провести прогнозирование и сравнить прогнозные значения с их реальными значениями. Проведем прогнозирование для каждого уравнения по отдельности, а также для всех систем уравнений в целом.

Структура этапа прогнозирования:
1. Построение системы уравнений
2. Вывод параметров для каждого уравнения
3. Прогнозирование для каждого уравнения и оценка резульатов по ряду метрик

### Система независимых уравнеий

$$
\begin{cases}
\hat y_1 = 0.5478x_{1} + 0.2297x_{2} \\
\hat y_2 = 0.1732x_{1} + 0.3493x_{4} + 0.3101x_{6} + 0.1358x_{8} \\
\hat y_3 = 0.2826x_{3} + 0.2021x_{4} + 0.2524x_{10} + 0.2532x_{11} \\
\end{cases}
$$


```python
print(f'Парметры первого: {final_fn1_nez.columns.values.tolist()}\n'
      f'Парметры второго: {final_fn2_nez.columns.values.tolist()}\n'
      f'Парметры третьего: {final_fn3_nez.columns.values.tolist()}')
```

#### Первое уравнение


```python
CustomLinearRegression(y1_train, final_fn1_nez).predict(pd.concat([x1_test, x2_test], axis=1), y1_test)
```

#### Второе уравнение


```python
CustomLinearRegression(y2_train, final_fn2_nez).predict(pd.concat([x1_test, x4_test, x6_test, x8_test], axis=1), y2_test)
```

#### Третье уравнение


```python
CustomLinearRegression(y3_train, final_fn3_nez).predict(pd.concat([x3_test, x4_test, x10_test, x11_test], axis=1), y3_test)
```

### Система рекурсивных уравнеий

$$
\begin{cases}
\hat y_1 = 0.5478x_{1} + 0.2297x_{2} \\
\hat y_2 = 0.2164y_{1} + 0.2852x_{1} + 0.1989x_{4} + 0.2861x_{6} \\
\hat y_3 = 0.5732y_{2} + 0.4683x_{11} \\
\end{cases}
$$


```python
print(f'Парметры первого: {final_fn1_rec.columns.values.tolist()}\n'
      f'Парметры второго: {final_fn2_rec.columns.values.tolist()}\n'
      f'Парметры третьего: {final_fn3_rec.columns.values.tolist()}')
```

#### Первое уравнение


```python
CustomLinearRegression(y1_train, final_fn1_rec).predict(pd.concat([x1_test, x2_test], axis=1), y1_test)
```

#### Второе уравнение


```python
CustomLinearRegression(y2_train, final_fn2_rec).predict(pd.concat([y1_test, x1_test, x4_test, x6_test], axis=1), y2_test)
```

#### Третье уравнение


```python
CustomLinearRegression(y3_train, final_fn3_rec).predict(pd.concat([y2_test, x11_test], axis=1), y3_test)
```

### Система одновременных уравнеий

$$
\begin{cases}
\hat y_1 = -1.2598x_{1} \\
\hat y_2 = 1.2379y_{1} - 2.0557x_{5} \\
\hat y_3 = 0.9046y_{1} - 1.7170x_{10} \\
\end{cases}
$$


```python
print(f'Парметры первого: {final_fn1_odn.columns.values.tolist()}\n'
      f'Парметры второго: {final_fn2_odn.columns.values.tolist()}\n'
      f'Парметры третьего: {final_fn3_odn.columns.values.tolist()}')
```

#### Первое уравнение


```python
CustomLinearRegression(y1_train, final_fn1_odn).predict(pd.concat([x1_test], axis=1), y1_test)
```

#### Второе уравнение


```python
CustomLinearRegression(y2_train, final_fn2_odn).predict(pd.concat([y1_test, x5_test], axis=1), y2_test)
```

#### Третье уравнение


```python
CustomLinearRegression(y3_train, final_fn3_odn).predict(pd.concat([y1_test, x10_test], axis=1), y3_test)
```

# Прогнозирование с помощью методов машинного обучения

Дополнительно было проведено прогнозирование с помощью методов машинного обучения.

В качестве методов машинного обучения были выбраны:
1. Логистическая регрессия
2. Метод опорных векторов
3. Метод k-ближайших соседей
4. Метод случайного леса
5. Метод градиентного бустинга
6. Метод нейронных сетей


```python
class MLPrediction():
    def __init__(self, X_train, X_test, y_train):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = preprocessing.LabelEncoder().fit_transform(y_train)

    def logreg(self):  # Logistic Regression
        self.logreg = LogisticRegression()
        self.logreg.fit(self.X_train, self.y_train)
        Y_pred = self.logreg.predict(self.X_test)
        self.acc_log = round(self.logreg.score(self.X_train, self.y_train) * 100, 2)

    def logreg_score(self):
        self.logreg()
        print(self.acc_log)
        print(self.logreg.coef_)

    def svc(self):  # Support Vector Machines
        self.svc = SVC(kernel='linear')
        self.svc.fit(self.X_train, self.y_train)
        Y_pred = self.svc.predict(self.X_test)
        self.acc_svc = round(self.svc.score(self.X_train, self.y_train) * 100, 2)

    def svc_score(self):
        self.svc()
        print(self.acc_svc)
        print(self.svc.coef_)

    def knn(self):
        self.knn = KNeighborsClassifier(n_neighbors=3)  # 3
        self.knn.fit(self.X_train, self.y_train)
        Y_pred = self.knn.predict(self.X_test)
        self.acc_knn = round(self.knn.score(self.X_train, self.y_train) * 100, 2)

    def knn_score(self):
        self.knn()
        print(self.acc_knn)

    def gaussian(self):  # Gaussian Naive Bayes
        self.gaussian = GaussianNB()
        self.gaussian.fit(self.X_train, self.y_train)
        Y_pred = self.gaussian.predict(self.X_test)
        self.acc_gaussian = round(self.gaussian.score(self.X_train, self.y_train) * 100, 2)

    def gaussian_score(self):
        self.gaussian()
        print(self.acc_gaussian)

    def perceptron(self):
        self.perceptron = Perceptron()
        self.perceptron.fit(self.X_train, self.y_train)
        Y_pred = self.perceptron.predict(self.X_test)
        self.acc_perceptron = round(self.perceptron.score(self.X_train, self.y_train) * 100, 2)

    def perceptron_score(self):
        self.perceptron()
        print(self.acc_perceptron)

    def linear_svc(self):  # Linear SVC
        self.linear_svc = LinearSVC()
        self.linear_svc.fit(self.X_train, self.y_train)
        Y_pred = self.linear_svc.predict(self.X_test)
        self.acc_linear_svc = round(self.linear_svc.score(self.X_train, self.y_train) * 100, 2)

    def linear_svc_score(self):
        self.linear_svc()
        print(self.acc_linear_svc)

    def sgd(self):  # Stochastic Gradient Descent
        self.sgd = SGDClassifier()
        self.sgd.fit(self.X_train, self.y_train)
        Y_pred = self.sgd.predict(self.X_test)
        self.acc_sgd = round(self.sgd.score(self.X_train, self.y_train) * 100, 2)

    def sgd_score(self):
        self.sgd()
        print(self.acc_sgd)

    def decision_tree(self):  # Decision Tree
        self.decision_tree = DecisionTreeClassifier()
        self.decision_tree.fit(self.X_train, self.y_train)
        Y_pred = self.decision_tree.predict(self.X_test)
        self.acc_decision_tree = round(self.decision_tree.score(self.X_train, self.y_train) * 100, 2)

    def decision_tree_score(self):
        self.decision_tree()
        print(self.acc_decision_tree)

    def random_forest(self):  # Random Forest
        self.random_forest = RandomForestClassifier(n_estimators=1000, max_depth=100, min_samples_leaf=1)
        self.random_forest.fit(self.X_train, self.y_train)
        self.Y_pred = self.random_forest.predict(self.X_test)
        self.random_forest.score(self.X_train, self.y_train)
        self.acc_random_forest = self.random_forest.score(self.X_train, self.y_train) * 100

    def random_forest_score(self):
        self.random_forest()
        print(self.acc_random_forest)

    def rfc_plot(self):
        self.random_forest()
        importances = self.random_forest.feature_importances_
        std = np.std([tree.feature_importances_ for tree in self.random_forest.estimators_], axis=0)
        indices = np.argsort(importances)[::-1]
        df = pd.DataFrame({
            'Features': indices,
            'Importance': importances[indices],
            'Importance std': std[indices]
        })
        fig = px.bar(df, x='Features', y='Importance', error_y='Importance std',
                     labels={'Importance': 'Importance score'})
        fig.update_layout(title='Важность переменных для Random Forest Classifier')
        return fig.show()

    def compare(self):
        self.logreg()
        self.svc()
        self.knn()
        self.gaussian()
        self.perceptron()
        self.linear_svc()
        self.sgd()
        self.decision_tree()
        self.random_forest()
        models = pd.DataFrame({
            'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression',
                      'Random Forest', 'Naive Bayes', 'Perceptron',
                      'Stochastic Gradient Decent', 'Linear SVC',
                      'Decision Tree'],
            'Score': [self.acc_svc, self.acc_knn, self.acc_log,
                      self.acc_random_forest, self.acc_gaussian, self.acc_perceptron,
                      self.acc_sgd, self.acc_linear_svc, self.acc_decision_tree]})
        return models.sort_values(by='Score', ascending=False)
```


```python
MLPrediction(X_train, X_test, y_train['y1']).compare()
```


```python
MLPrediction(X_train, X_test, y_train['y2']).compare()
```


```python
MLPrediction(X_train, X_test, y_train['y3']).compare()
```


```python
MLPrediction(X_train, X_test, y_train['y3']).rfc_plot()
```

# Заключение

После проведения анализа данных и построения моделей можно сделать следующие выводы:

