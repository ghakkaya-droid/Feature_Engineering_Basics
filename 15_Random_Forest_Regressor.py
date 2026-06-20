import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from functions import data_review,value_counter,numeric_encoding
from sklearn.model_selection import train_test_split,RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score



print("="*30,"Data reading & review","="*30)
df=pd.read_csv('Campus_Gym.csv')
data_review(df)
print(f"Data Frame NanRow_Count: {len(df[df.isna().any(axis=1)])}")
print(df.columns)
print(f"duplicated row number:{df[df.duplicated].shape[0]}")
df['date'] = pd.to_datetime(df["date"],utc=True)
df['year'] = df["date"].dt.year
df.drop('date',axis=1,inplace=True)
print(df.head())
print("Years in df")
print(df["year"].unique())

print("Correlations")
print(df.corr())
df.drop('timestamp',axis=1,inplace=True)
print(df.head())

print("="*30,"Random Forest Regressor & Other Regressions","="*30)
X = df.drop('number_people',axis=1)
y = df['number_people']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=15)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

def calculate_model_metrics (true,predict):
    mae = mean_absolute_error(true,predict)
    mse = mean_squared_error(true,predict)
    rmse = np.sqrt(mean_squared_error(true,predict))
    score_r2 = r2_score(true,predict)
    return mae,rmse,score_r2 #these variables will be return as tuple

models ={
    "Linear Regression": LinearRegression(),
    "Ridge":Ridge(),
    "Lasso": Lasso(),
    "K-Neighbours Regression":KNeighborsRegressor(),
    "Decision Tree":DecisionTreeRegressor(),
    "Random Forest Regression":RandomForestRegressor()
}
for name,model in models.items():
    model.fit(X_train,y_train)
    y_trian_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    model_train_mae,model_train_rmse,model_train_r2 = calculate_model_metrics(y_train,y_trian_pred)
    model_test_mae,model_test_rmse,model_test_r2 = calculate_model_metrics(y_test,y_test_pred)
    print(name)
    print("Evaluation for Training Set")
    print("RMSE:",model_train_rmse)
    print("Mean Absolute Error: ",model_train_mae)
    print("Model Score: ",model_train_r2)
    print("-"*60)
    print("Evaluation for Test Set")
    print("RMSE:",model_test_rmse)
    print("Mean Absolute Error: ",model_test_mae)
    print("Model Score: ",model_test_r2)
    print("-"*60)
    print("\n")

print("="*30,"Random Forest Tuning","="*30)

knn_params = {"n_neighbors":[2,3,10,20,40,50]}
rf_params = {
    "max_depth":[5,8,10,15,None],
    "max_features":["sqrt","log2",5,7,10],
    "min_samples_split":[2,8,12,20],
    "n_estimators":[100,200,500,1000]
}
randomcv_models = [
    ("KNN",KNeighborsRegressor(),knn_params),
    ("RF",RandomForestRegressor(),rf_params)]
#Tuple Unpacking
for name,model,params in randomcv_models:
    randomcv = RandomizedSearchCV(
        estimator= model,
        param_distributions= params,
        n_iter=100,
        cv=3,
        n_jobs=-1)
    randomcv.fit(X_train,y_train)
    print("Best Parameters for: ",name,randomcv.best_params_)
    y_test_pred = randomcv.predict(X_test)
    y_trian_pred = randomcv.predict(X_train)
    model_train_mae,model_train_rmse,model_train_r2 = calculate_model_metrics(y_train,y_trian_pred)
    model_test_mae,model_test_rmse,model_test_r2 = calculate_model_metrics(y_test,y_test_pred)
    print("Evaluation for Training Set")
    print("RMSE:",model_train_rmse)
    print("Mean Absolute Error: ",model_train_mae)
    print("Model Score: ",model_train_r2)
    print("-"*60)
    print("Evaluation for Test Set")
    print("RMSE:",model_test_rmse)
    print("Mean Absolute Error: ",model_test_mae)
    print("Model Score: ",model_test_r2)
    

'''
    Best paraetreler:
    Best Parameters for:  KNN {'n_neighbors': 2} /Model Score:  0.9567459314563896
    Best Parameters for:  RF {'n_estimators': 500, 'min_samples_split': 2, 'max_features': 5, 'max_depth': None}
    Model Score:  0.9209271719986976
'''




