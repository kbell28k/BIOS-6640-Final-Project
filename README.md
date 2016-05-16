#BIOS 6640 Final Project
#Kayla Bell
#May 10, 2016

import pandas as pd
import numpy as np
import statsmodels.api as sm
import pylab as pl

df = pd.read_csv("/Users/kaylabell/Desktop/language.csv", skipinitialspace=True)
df.describe()
print df.describe()

df = pd.read_csv("/Users/kaylabell/Desktop/ID2.csv", skipinitialspace=True)
df.describe()
print df.describe()


df = pd.read_csv("/Users/kaylabell/Desktop/Final.csv")
print df.head()
df.columns = ["Date", "Day", "ID2", "Followers", "Friends", "Status_Count", "Time_zone", "Geo_enabled", "Language"]
print df.columns
print df.describe()
print df.std()
df.hist()
pl.show()
print pd.crosstab(df['Geo_enabled'], df['Time_zone'], rownames=['Geo_enabled'])
print pd.crosstab(df['Geo_enabled'], df['Day'], rownames=['Geo_enabled'])

dummy_ranks = pd.get_dummies(df['Date'], prefix='Date')
print dummy_ranks.head()

cols_to_keep = ['Geo_enabled', 'Status_Count', 'Time_zone']
data = df[cols_to_keep].join(dummy_ranks.ix[:, 'Date_2':])
print data.head()
data['intercept'] = 1.0
train_cols = data.columns[1:]
logit = sm.Logit(data['Geo_enabled'], data[train_cols])
result = logit.fit()
print result.summary()
print result.conf_int()
print np.exp(result.params)
