import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from functions import data_review,value_counter,numeric_encoding,calculate_model_metrics_reg
from sklearn.model_selection import train_test_split,RandomizedSearchCV
from sklearn.ensemble import AdaBoostRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error

print("="*30,"Data reading & review","="*30)
df = pd.read_csv('cardekho.csv')
data_review(df)
print(f"Data Frame NanRow_Count: {len(df[df.isna().any(axis=1)])}")
categorical = df.select_dtypes("str").columns
numerical = df.select_dtypes("number").columns
value_count_list = ['seller_type','fuel_type','transmission_type']
for col in value_count_list:
    print("-"*60)
    print(df[col].value_counts())
    print("-"*60)
print("="*30,"Data cleaning","="*30)
df.drop('Unnamed: 0',axis=1,inplace=True)
print(f"Duplicated row number: {df[df.duplicated()].shape[0]}")
print(df['seats'].value_counts())
print("-"*60)
df['seats']=df['seats'].replace(0,5)
print(df['seats'].value_counts())

print("="*30,"Dropping outlier values","="*30)
print("selling price column adjustment")
print(df['selling_price'].sort_values(ascending=False))
drop_idx = df[df['selling_price']>15000000].index.tolist()
print(drop_idx)
df.drop(drop_idx,axis=0,inplace=True)
'''
print("="*30,"Visualition","="*30)
sns.scatterplot(data=df,x="vehicle_age",y='selling_price',hue='fuel_type')
plt.show()
'''
print("km_driven column adjustment")
df = df[df['km_driven']<1000000]
'''
print("="*30,"Visualition","="*30)
sns.scatterplot(data=df,x="km_driven",y='selling_price')
plt.show()
'''
print("correlations")
print(df.corr(numeric_only=True)) #İçerisinde Kategorik dataları atmak için bunu yapmazsan hata fırlatır

print("="*30,"X,y definitions and Encoding","="*30)
X = df.drop("selling_price",axis=1)
y = df['selling_price']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=15)
cat_cols = df.select_dtypes("str").columns.tolist()
num_cols = df.select_dtypes("number").columns.tolist()
#calculation of uniq values of categorical values
unique_values = df[cat_cols].nunique()
print(unique_values)
#seller_type,fuel_type,transmission_type ---> one-hot encoding
#car_name,brand,model ---> frequency encoding (Burada ordinal encoding yapılamaz çünkü yapsak bile numara çevirince aralarında bir hiyerarşi olmayacak.)
one_hot_columns = ['seller_type','fuel_type','transmission_type']
freq_columns = ['car_name','brand','model']
#Frequency Encoding
for col in freq_columns:
    freq = X_train[col].value_counts(normalize=True)
    mean_freq = freq.mean()
    X_train[col+'_freq'] = X_train[col].map(freq)
    X_test[col+'_freq'] = X_test[col].map(freq).fillna(mean_freq)
X_train=X_train.drop(freq_columns,axis=1)
X_test=X_test.drop(freq_columns,axis=1)
print(X_train)
#One-Hot Encoding
transformer = ColumnTransformer(
    transformers=[
        (
            "onehot",
            OneHotEncoder(
                drop='first',handle_unknown='ignore'),
                one_hot_columns)
    ],remainder="passthrough"
)
X_train_encoded = transformer.fit_transform(X_train)
X_test_encoded = transformer.transform(X_test)

new_columns = transformer.get_feature_names_out()
X_train = pd.DataFrame(X_train_encoded,columns=new_columns,index=X_train.index)
X_test = pd.DataFrame(X_test_encoded,columns=new_columns,index=X_test.index)

print(X_train.head())

print("="*30,"Ada boost","="*30)

model = AdaBoostRegressor()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
mae,rmse,score=calculate_model_metrics_reg(y_test,y_pred)
print("RMSE:",rmse)
print("Mean Absolute Error: ",mae)
print("Model Score: ",score)

print("="*30,"Ada boost Tuning","="*30)
params = {
    "n_estimators":[50,80,100,120],
    "learning_rate":[0.001,0.01,0.1,1,2],
    "loss":["linear","square","exponential"]
}

rcv=RandomizedSearchCV(
    estimator=AdaBoostRegressor(),
    param_distributions=params,
    scoring='r2',
    cv=5
)
rcv.fit(X_train,y_train)
y_pred=rcv.predict(X_test)
mae,rmse,score=calculate_model_metrics_reg(y_test,y_pred)
print(rcv.best_params_)
print("RMSE:",rmse)
print("Mean Absolute Error: ",mae)
print("Model Score: ",score)

print("="*30,"Ada boost Deep Tuning","="*30)
#Eğer adaboost içerisindeki regressorları ellemazsen varsayılan decisiontreeRegressor gelir.
params = {
    "estimator__max_depth":[3,4,5],
    "n_estimators":[50,80,100,120],
    "learning_rate":[0.001,0.01,0.1,1,2],
    "loss":["linear","square","exponential"]
}

rcv=RandomizedSearchCV(
    estimator=AdaBoostRegressor(DecisionTreeRegressor()), #İşte burada içine ekliyoruz
    param_distributions=params,
    scoring='r2',
    cv=5
)
rcv.fit(X_train,y_train)
y_pred=rcv.predict(X_test)
mae,rmse,score=calculate_model_metrics_reg(y_test,y_pred)
print(rcv.best_params_)
print("RMSE:",rmse)
print("Mean Absolute Error: ",mae)
print("Model Score: ",score)