import pandas
import matplotlib.pyplot as plt
import datetime

df = pandas.read_csv("dataset/AirQualityUCI.csv", sep = ";", decimal = ",")
df = df.iloc[:, 0:14]

len(df)
df.head()

print(len(df))
print(df.head())

df = df[df['Date'].notnull()]
print(df.isna().sum())
df['DateTime'] = (df.Date) + ' ' + (df.Time)
print(df['DateTime'])
df.DateTime = df.DateTime.apply(lambda x: datetime.datetime.strptime(x, '%d/%m/%Y %H.%M.%S'))
print (type(df.DateTime[0]))
df.index = df.DateTime
plt.plot(df['T'])
plt.show()