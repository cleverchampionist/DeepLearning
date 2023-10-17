import pandas
import numpy as np
import datetime
from sklearn import metrics
from math import sqrt
from statsmodels.graphics.tsaplots import plot_acf
import matplotlib.pyplot as plt

df = pandas.read_csv("dataset/AirQualityUCI.csv", sep = ";", decimal= ",")
df = df.iloc[:, 0:14]
df = df[df['Date'].notnull()]
df['DateTime'] = (df.Date) + ' ' + (df.Time)
df.DateTime = df.DateTime.apply(lambda x: datetime.datetime.strptime(x, "%d/%m/%Y %H.%M.%S"))
df.index = df.DateTime

print(
    'Mean: ', np.mean(df['T']), 
    '; Standard Deviation: ', np.std(df['T']),
    '; Maximum Temperature: ', max(df['T']),
    '; Minimum Temperature: ', min(df['T'])
)
df['T_t-1'] = df['T'].shift(1)
df_naive = df[['T', 'T_t-1']][1:]

true = df_naive['T']
prediction = df_naive['T_t-1']
error = sqrt(metrics.mean_squared_error(true, prediction))
print('RMSE fro Naive Method 1:', error)

df['T_rm'] = df['T'].rolling(3).mean().shift(1)
df_naive = df[['T', 'T_rm']].dropna()

true = df_naive['T']
prediction = df_naive['T_rm']
error = sqrt(metrics.mean_squared_error(true, prediction))
print('RMSE for Naive Method 2: ', error)

split = len(df) - int(0.2 * len(df))
train, test = df['T'][0:split], df['T'][split:]
plot_acf(train, lags = 100)
plt.show()