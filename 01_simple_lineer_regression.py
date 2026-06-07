import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from functions import data_info
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

print("="*30,"Lineer Regration","="*30)
print("-"*30,"- Read File '1-studyhours.csv' -","-"*30)
df=pd.read_csv('1-studyhours.csv')
print("df info:")
print(data_info(df))
print("df description:")
print(df.describe())
print(df.head(10))
'''
print("-"*30,"- Drawing Graph -","-"*30)
plt.figure(figsize=(5,5))
plt.scatter(x=df["Study Hours"],y=df["Exam Score"])
plt.show()
'''
#independent(X):Bağımmsız Değişken / dependent(y): Bağımlı Değişken - features
'''
    Kütüphaneler X'ler için DataFrame, y ler için Series olarak ister (Çıktı tek olduğu için) (Target)
'''
print("-"*30,"- Defining the variables X,y -","-"*30)
X = df[["Study Hours"]]
y = df["Exam Score"]
print(f"Type X: {type(X)}, Type y: {type(y)}")

print("-"*30,"Using train-test split","-"*30)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=15)
print(type(X_train))
print(y_train)

print("-"*30,"Standardize the data set / StandardScaler","-"*30)
#balanced feature values
#efficient gradient descent
#l1,l2 ödül - ceza işlemleri
scaler = StandardScaler()
X_train_org=X_train.copy()
X_test_org = X_test.copy()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print("X_train and X_test has been standized by StandardScaller")
print("X_Train:")
print(X_train)
print("X_Test:")
print(X_test)

print("-"*30,"Make an object from LineerRegreesion","-"*30)
regression = LinearRegression()
regression.fit(X_train,y_train)
print(f"Coefficient: {regression.coef_}")
print(f"Intersept: {regression.intercept_}")

print("-"*30,"Drawing Graph of predict_train","-"*30)
'''
plt.scatter(X_train,y_train)
plt.plot(X_train,regression.predict(X_train),"r")
plt.show()
'''
print("-"*30,"Predict of a value","-"*30)
print(regression.predict(scaler.transform([[20]])))

print("-"*30,"Prediction with test data","-"*30)
y_perdiction = regression.predict(X_test)
print("X_test:")
print(X_test)
print("y_test_prediction:")
print(y_perdiction)
'''
df_compare = pd.DataFrame({
    "index": range(len(y_test)),
    "Gerçek": y_test.values,
    "Tahmin":y_perdiction
})
df_melted = df_compare.melt(
    id_vars="index",
    value_vars=["Gerçek","Tahmin"],
    var_name="Tip",
    value_name="Değer"
)
print(df_melted)
'''
print("-"*30,"Compare Reals and Predictions","-"*30)
df_compare = X_test_org.copy()
df_compare["y_predict"] = y_perdiction
df_compare["y_test"] = y_test
print(df_compare)
'''
plt.scatter(df_compare["Study Hours"], df_compare["y_test"], label="Gerçek Not")
plt.scatter(df_compare["Study Hours"], df_compare["y_predict"], label="Tahmin Not")

plt.xlabel("Study Hours")
plt.ylabel("Score")
plt.legend()
plt.show()
'''
print("-"*30,"Error Metrics -MSE / MAE / RMSE / R_square","-"*30)
mse = mean_squared_error(y_test,y_perdiction)
mae = mean_absolute_error(y_test,y_perdiction)
rmse = np.sqrt(mse)
r2 = r2_score(y_test,y_perdiction)
print("mse: ",mse)
print("mae: ",mae)
print("rmse: ",rmse)
print("R2_score: ",r2)

r2_adj = 1-(1-r2)*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1)
print("R2_adjusted: ",r2_adj)