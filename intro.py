'''
MCAR : Missing Completely Random / Tamamen tesadüf eseri eksik veriler
MAR : Missing at Random / Bazı değişlenlere bağlı eksik veriler
MNAR : Missing not Random / Kati olarak eksik veriler
'''

import seaborn as sns
import pandas as pd
import functions
import matplotlib.pyplot as plt

df = sns.load_dataset("titanic")

#Data info
'''
print(df.head())
print(df.describe())
print(functions.data_info(df))
print(functions.column_unique(df))
print(df.isnull().sum()) # kolonlarda boş olan veri miktarını verir.
print(df.shape)
'''


#Data imputation
#Age column histogram
'''
sns.histplot(data=df,
             x="age",
             hue="alive",
             multiple="dodge",
             kde=True)
plt.show()
'''

#Pick the mean of Age and complete the missing datas
age_group=df.groupby("sex")["age"].transform("mean")
print(type(age_group))
print(age_group)
df["age_adcj"]=df["age"].fillna(age_group)
print(df.isnull().sum())
print(df.head(10))

#"embarked" kolonunda 2 adet boş veri var. düzelt 

mode_values = df[["embarked","embark_town"]].mode().iloc[0]


df[["embarked_adj","embark_town_adj"]] = df[["embarked","embark_town"]].fillna(mode_values)
print(df.head(10))
print(df.isnull().sum())



