import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from functions import data_info
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.linear_model import LinearRegression,Lasso,Ridge,ElasticNet,LassoCV,RidgeCV,ElasticNetCV
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from sklearn.pipeline import Pipeline

print("="*30,"RidgeLassoElasticNet trainings","="*30)
print("-"*30,"-Reading file-","-"*30)
df=pd.read_csv("4-Algerian_forest_fires_dataset.csv")
print("df info:")
print(data_info(df))
print("df description:")
print(df.describe())
print(df.head(10))
print(df[df.isna().any(axis=1)])
df.drop(index=[122,168],inplace=True)
print("DataSet has regions. From 0 to 123 Region:0, from124toEnd Region:1")
print(df[df.isna().any(axis=1)])
df.loc[:123,"Region"] = 0
df.loc[124:,"Region"] = 1
print(df.head())
print(df.tail())
print("-"*30,"-Cleaning Na values-","-"*30)
df = df.dropna().reset_index(drop=True)
print(df.iloc[121])
print(df.iloc[122])
print("-"*30,"-Cleaning Column Names-","-"*30)
df.columns = df.columns.str.strip() # Eğer içerisine bir şey vermezsek boşlukları siler
print(df.columns)
print("-"*30,"-Convert column data to ineger & float-","-"*30)
print(df.head())
print(df[df["day"]=="day"])
df.drop(index=122,inplace=True)
df.reset_index(drop=True)
df[['day','month','year','Temperature','RH','Ws']] = df[['day','month','year','Temperature','RH','Ws']].astype(int)
print(data_info(df))
df[['Rain','FFMC','DMC','DC','ISI','BUI','FWI']] = df[['Rain','FFMC','DMC','DC','ISI','BUI','FWI']].astype(float)
print(data_info(df))

print("-"*30,"-column:'Classes' rewiew and clean-","-"*30)
print(df['Classes'].value_counts())
df['Classes']=df['Classes'].str.strip()
print("Col: 'Classes' has been string clean")
print(df['Classes'].value_counts())
classes_dict = {
    "fire":1,
    "not fire":0
}
df['Classes'] = df['Classes'].map(classes_dict)
print(df.head())
print(df['Classes'].value_counts(normalize=True)*100)

print("-"*30,"-correlation headmap drawing-","-"*30)
def corr_effect_grt(data_set,per,target):
    corr_df = data_set.corr(numeric_only=True)
    strong_corr = []
    for i, row in enumerate(corr_df.index):
        for j,col in enumerate(corr_df.columns):
            if j<=i:
                continue
            value = corr_df.loc[row,col]

            if abs(value) > per:
                if col == target:
                    strong_corr.append({
                        "Column":row,
                        "Correlation":value
                    })
    return strong_corr

def corr_effect_less(data_set,per,target):
    corr_df = data_set.corr(numeric_only=True)
    strong_corr = []
    for i, row in enumerate(corr_df.index):
        for j,col in enumerate(corr_df.columns):
            if j<=i:
                continue
            value = corr_df.loc[row,col]

            if abs(value) <= per:
                if col == target:
                    strong_corr.append({
                        "Column":row,
                        "Correlation":value
                    })
    return strong_corr

df_strong_corr = pd.DataFrame(corr_effect_grt(df,0.5,"FWI"))
df_weak_corr = pd.DataFrame(corr_effect_less(df,0.5,"FWI"))
print("Strong Correlations")
print(df_strong_corr)
print("Weak Correlations")
print(df_weak_corr)
#sns.heatmap(df.corr(),annot=True)
#plt.show()

print("-"*30,"-drop collumns: day, month, year-","-"*30)
df.drop(["day","month","year"],axis=1,inplace=True)
print(df.head())

print("-"*30,"Definition of independent and dependent variables","-"*30)
X = df.drop("FWI",axis=1)
y = df["FWI"]
print(X)
print(y)

print("-"*30,"Train & Test Set creations","-"*30)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=15)

#redundancy, multicollinearity, overfitting
'''
    X data seri içerisinde birbiriyle çok aşırı derece kolerasyona sahip değişkenlerden birini
    devre dışı bırakmamız gerekiyor. Bunun en bütük sebebi aşırı öğrenmeye sebebiyet vermesi.
    bu bağlamda birbiri arasında %85 ve üzerinde bağlantı olan kolonlardan birini çıkartmak gerekiyor.
    Netice olarak iki kolon target değerine aynı bağıntı ile bağlı olacağından regresyonda bu 
    kolerasyon kalan bağlantıları ezebilir.
'''
print("-"*30,"Dropping strong correlated collumns in X_train","-"*30)
def correlation_for_dropping(df,threshold):
    column_to_drop = set()
    corr = df.corr()
    for i in range(len(corr.columns)):
        for j in range(i):
            if abs(corr.iloc[i,j]) > threshold:
                column_to_drop.add(corr.columns[i])
    return column_to_drop

columns_dropping = correlation_for_dropping(X_train,0.85)
X_train.drop(columns_dropping,axis=1,inplace=True)
X_test.drop(columns_dropping,axis=1,inplace = True)
print(X_train)
print(X_train.shape,X_test.shape)

print("-"*30,"scaling the data","-"*30)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


#Scaled data graph comperation
plt.subplots(figsize=(15,5))
plt.subplot(1,2,1)
sns.boxplot(data=X_train)
plt.title("X_train")
plt.subplot(1,2,2)
sns.boxenplot(data=X_train_scaled)
plt.title("X_train_scaled")
plt.show()


print("-"*30,"Lineer Regression","-"*30)
linear = LinearRegression()
linear.fit(X_train_scaled,y_train)
y_pred = linear.predict(X_test_scaled)
mae = mean_absolute_error(y_test,y_pred)
mse = mean_squared_error(y_test,y_pred)
score = r2_score(y_test,y_pred)
print("Mean Absolute Error: ",mae)
print("Mean Squared Error: ",mse)
print("R2 score: ",score)
#plt.scatter(y_test,y_pred)
#plt.show()

print("-"*30,"Lineer Regression Model: Lasso","-"*30)
lasso = Lasso()
lasso.fit(X_train_scaled,y_train)
y_pred = lasso.predict(X_test_scaled)
mae = mean_absolute_error(y_test,y_pred)
mse = mean_squared_error(y_test,y_pred)
score = r2_score(y_test,y_pred)
print("Mean Absolute Error: ",mae)
print("Mean Squared Error: ",mse)
print("R2 score: ",score)
#plt.scatter(y_test,y_pred)
#plt.show()

print("-"*30,"Lineer Regression Model: Ridge","-"*30)
ridge = Ridge()
ridge.fit(X_train_scaled,y_train)
y_pred = ridge.predict(X_test_scaled)
mae = mean_absolute_error(y_test,y_pred)
mse = mean_squared_error(y_test,y_pred)
score = r2_score(y_test,y_pred)
print("Mean Absolute Error: ",mae)
print("Mean Squared Error: ",mse)
print("R2 score: ",score)
#plt.scatter(y_test,y_pred)
#plt.show()

print("-"*30,"Lineer Regression Model: ElasticNet","-"*30)
elastic = ElasticNet()
elastic.fit(X_train_scaled,y_train)
y_pred = elastic.predict(X_test_scaled)
mae = mean_absolute_error(y_test,y_pred)
mse = mean_squared_error(y_test,y_pred)
score = r2_score(y_test,y_pred)
print("Mean Absolute Error: ",mae)
print("Mean Squared Error: ",mse)
print("R2 score: ",score)
#plt.scatter(y_test,y_pred)
#plt.show()

print("-"*30,"Cross Validation for Lasso with LassoCV","-"*30)
#Lasso yaparken Lamda kat sayılarını değiştirme CrossValidation
lassocv = LassoCV(cv=5)
lassocv.fit(X_train_scaled,y_train)
y_pred = lassocv.predict(X_test_scaled)
mae = mean_absolute_error(y_test,y_pred)
mse = mean_squared_error(y_test,y_pred)
score = r2_score(y_test,y_pred)
print("Mean Absolute Error: ",mae)
print("Mean Squared Error: ",mse)
print("R2 score: ",score)
print(f"Lasso_alpha: {lassocv.alpha_}")
print("lassocv_alphas:")
print(lassocv.alphas_)

print("-"*30,"Cross Validation for Ridge with RidgeCV","-"*30)
ridgecv = RidgeCV(cv=5)
ridgecv.fit(X_train_scaled,y_train)
y_pred = ridgecv.predict(X_test_scaled)
mae = mean_absolute_error(y_test,y_pred)
mse = mean_squared_error(y_test,y_pred)
score = r2_score(y_test,y_pred)
print("Mean Absolute Error: ",mae)
print("Mean Squared Error: ",mse)
print("R2 score: ",score)
print(f"Ridgecv_alpha: {ridgecv.alpha_}")

print("-"*30,"Cross Validation for ElasticNet with ElasticNetCV","-"*30)
elasticnetcv = ElasticNetCV(cv=5)
elasticnetcv.fit(X_train_scaled,y_train)
y_pred = elasticnetcv.predict(X_test_scaled)
mae = mean_absolute_error(y_test,y_pred)
mse = mean_squared_error(y_test,y_pred)
score = r2_score(y_test,y_pred)
print("Mean Absolute Error: ",mae)
print("Mean Squared Error: ",mse)
print("R2 score: ",score)
print(f"ElasticNetCV_alpha: {elasticnetcv.alpha_}")