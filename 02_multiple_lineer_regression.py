import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from functions import data_info
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

print("="*30,"Multiple Lineer Regration","="*30)
print("-"*30,"-Reading file-","-"*30)
df = pd.read_csv('2-multiplegradesdataset.csv')
print("df info:")
print(data_info(df))
print("df description:")
print(df.describe())
print(df.head(10))

'''
print("-"*30,"-Pair_Plot-","-"*30)
sns.pairplot(df)
plt.show()
'''

print("-"*30,"-Corralation info-","-"*30)
corr_df = df.corr(numeric_only=True)
strong_corr = {}
for i, row in enumerate(corr_df.index):
    for j,col in enumerate(corr_df.columns):
        if j<=i:
            continue
        value = corr_df.loc[row,col]

        if abs(value) > 0.5:
            strong_corr[(row,col)] = value

print(strong_corr)
print(df.corr())
print(strong_corr.keys())

'''
print("-"*30,"seaborn bestfitline drawing","-"*30)
sns.regplot(x=df['Study Hours'],y=df['Exam Score'])
plt.show()
'''

print("-"*30,"Definition of X and y","-"*30)
#independent and dependent featurs
X = df.copy().drop("Exam Score",axis=1)
y = df["Exam Score"]
print("X independents:")
print(X)
print("y dependents: ")
print(y)

print("-"*30,"train-test split & standardScaler","-"*30)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=15)
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

print("-"*30,"Make predictions within the boundaries of X_train.","-"*30)
print("first row of DataFrame")
print(df.iloc[0])
new_student = [[5,7,90,2]]
print(f"new student independent values: \n {new_student}")
print("Prediction for the new student Target:")
new_student_scaled = scale.transform(new_student)
new_student_predic = regression.predict(new_student_scaled)
print(new_student_predic)


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
sns.displot(residuals,kind="kde")
#plt.show()

print("-"*30,"coefficient and constrain","-"*30)
coeff = regression.coef_
cons = regression.intercept_
print("Coefficients")
print(coeff)
print("Constrain")
print(cons)