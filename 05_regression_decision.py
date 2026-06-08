from lazypredict.Supervised import LazyRegressor
from sklearn import datasets
from sklearn.utils import shuffle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler,MinMaxScaler,RobustScaler
from functions import data_info

diabetes = datasets.load_diabetes()

X,y = shuffle(diabetes.data,diabetes.target,random_state=13)
X=X.astype(np.float32)
offset = int(X.shape[0]*0.9)
X_train,y_train = X[:offset],y[:offset]
X_test,y_test = X[offset:],y[offset:]

reg = LazyRegressor(verbose=0, ignore_warnings=False, custom_metric=None)
models, predictions = reg.fit(X_train, X_test, y_train, y_test)
print(models)

df=pd.read_csv('co2.csv')
df_column_info = data_info(df)
df_column_info_str = df_column_info[df_column_info["Dtype"] == "str"]
column_str_list = list(df_column_info_str["Columns"])
df.drop(column_str_list,inplace=True,axis=1)
scalers = {
    "StandardScaler":StandardScaler(),
    "MinMaxScaler":MinMaxScaler(),
    "RobutScaler":RobustScaler()
}

scaled_dfs = {}
for name,scaler in scalers.items():
    scaled = scaler.fit_transform(df)
    scaled_dfs[name] = pd.DataFrame(scaled,columns=df.columns)

print(scaled_dfs)

print (scalers.items())
deneme = [15,20,30,50]
print(list(enumerate(deneme)))