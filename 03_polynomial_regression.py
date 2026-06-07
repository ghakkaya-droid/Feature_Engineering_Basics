import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from functions import data_info
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from sklearn.pipeline import Pipeline

print("="*30,"Polynomial Regression","="*30)
print("-"*30,"-Reading file-","-"*30)
df = pd.read_csv('3-customersatisfaction.csv')
print("df info:")
print(data_info(df))
print("df description:")
print(df.describe())
print(df.head(10))
df.drop("Unnamed: 0",axis=1,inplace=True)
print(df.head(10))
print("-"*30,"Dravind Graph Customer Satisfaction Vs Incentive","-"*30)
#plt.scatter(x=df['Customer Satisfaction'],y=df['Incentive'])
#plt.xlabel("Customer Satisfaction")
#plt.ylabel("Incentive")
#plt.show()

print("-"*30,"Definition of X and y","-"*30)
#independent and dependent featurs
X = df[['Customer Satisfaction']]
y = df["Incentive"]
print("X independents:")
print(X)
print("y dependents: ")
print(y)

print("-"*30,"train-test split & standardScaler","-"*30)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=15)
scale = StandardScaler()
X_train_org = X_train.copy()
X_test_org = X_test.copy()
X_train = scale.fit_transform(X_train)
X_test = scale.transform(X_test)
regression = LinearRegression()
regression.fit(X_train,y_train)
print("Standardized X values:")
print(X_train)
print(X_test)

print("-"*30,"prediction on test data set","-"*30)
y_pred = regression.predict(X_test)
df_compare = X_test_org.copy()
df_compare["y_predict"] = y_pred
df_compare["y_test"] = y_test
print(df_compare)

print("-"*30,"Error Metrics -MSE / MAE / RMSE / R_square","-"*30)
mse = mean_squared_error(y_test,y_pred)
mae = mean_absolute_error(y_test,y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test,y_pred)
print("mse: ",mse)
print("mae: ",mae)
print("rmse: ",rmse)
print("R2_score: ",r2)
r2_adj = 1-(1-r2)*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1)
print("R2_adjusted: ",r2_adj)

print("residuals : differences between y_test Vs y_pred")
residuals = y_test-y_pred
print(residuals)
'''
plt.scatter(x=X_test,y=y_test)
plt.scatter(x=X_test,y=y_pred)
plt.xlabel("Customer Satisfaction")
plt.ylabel("Incentive")
plt.show()
'''

print("-"*30,"Polynomial Regression Degree = 2","-"*30)
poly = PolynomialFeatures(degree=2,include_bias=True)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)
print("X_train_poly:")
print(X_train_poly)
regression = LinearRegression()
regression.fit(X_train_poly,y_train)
y_pred_poly = regression.predict(X_test_poly)
r2_poly = r2_score(y_test,y_pred_poly)
print(r2_poly)

print("-"*30,"Polynomial Regression Degree = 3","-"*30)
poly = PolynomialFeatures(degree=3,include_bias=True)
X_train_poly3 = poly.fit_transform(X_train)
X_test_poly3 = poly.transform(X_test)
print("X_train_poly:")
print(X_train_poly3)
regression = LinearRegression()
regression.fit(X_train_poly3,y_train)
y_pred_poly3 = regression.predict(X_test_poly3)
r2_poly3 = r2_score(y_test,y_pred_poly3)
print(r2_poly3)

print("="*30,"Prediction on New Data by using Created Model","="*30)
new_df = pd.read_csv('3-newdatas.csv')
new_df.rename(columns={"0":"Customer Satisfaction"},inplace=True)
X_new = new_df[["Customer Satisfaction"]]
X_new_org = X_new.copy()
X_new=scale.fit_transform(X_new)
X_new_poly = poly.transform(X_new)
y_new_pred = regression.predict(X_new_poly)
X_new_org["y_new_pred"] = y_new_pred
print(X_new_org)
'''
plt.plot(X_new,y_new_pred,label="New Predictions",color="r")
plt.scatter(X_train,y_train,label="Training Points")
plt.scatter(X_test,y_test,label="Test Data",color="y")
plt.legend()
plt.show()
'''

print("="*30,"Pipeline Function Creation","="*30)
def poly_regration(degree):
    scaler = StandardScaler()
    poly_features = PolynomialFeatures(degree=degree)
    lin_reg = LinearRegression()
    pipeline = Pipeline([
        ("standard_scaler",scaler),
        ("poly_features",poly_features),
        ("lin_reg",lin_reg)
    ])
    pipeline.fit(X_train,y_train)
    score = pipeline.score(X_test,y_test)
    print(score)

for degree in range(1,10):
    poly_regration(degree)

