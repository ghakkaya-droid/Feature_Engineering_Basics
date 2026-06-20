import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from functions import data_review,value_counter,numeric_encoding
from sklearn.preprocessing import LabelEncoder,StandardScaler,PolynomialFeatures
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from lazypredict.Supervised import LazyRegressor
import time



import warnings
warnings.filterwarnings("ignore")

print("="*30,"File Reading","="*30)
df = pd.read_csv('turkey_car_market.csv')
data_review(df)
'''
Notes for columns:
İlan tarihi --> Drop
Marka,Arac Tip Grubu,Renk,Kasa Tipi --> Label Encode
Arac Tip --> drop (Maybe look the none values and unique)
Model Yıl,Km --> keep
CCM,Kimden,Durum,Vites,Yakıt Turu,Beygir Gucu --> Ordinal Encoding,
Fiyat --> Target

CCM kolonu silizenecek veriler: 'Bilmiyorum', '-'
Kimden Kolonu Değerleri --> Galeriden:1,Sahibinden:2,Yetkili:3
'''

print("="*30,"Feature Engineering","="*30)
df.drop(['İlan Tarihi','Arac Tip'],axis=1,inplace=True)
df['Model Yıl'].astype(int)
#df'in kopyası alındı
df_org = df.copy()
#Cleanin Column 'CCM'
deleting_list=['Bilmiyorum','-']
delete_idx = df[df['CCM'].isin(deleting_list)].index
df = df.drop(index=delete_idx)
dublicated_data_count = df[df.duplicated()].shape[0]
if dublicated_data_count !=0:
    print(f"Dublicated data count: {dublicated_data_count}")
    df=df.drop_duplicates()
    print(f"control: {df.duplicated().sum()}")

#CCM Replacement
ccm_keys = df['CCM'].value_counts().index.to_list()
ccm_values = [1600,1300,2000,1800,3000,2500,4000,3500,4500,5500,5000,7000,6000]
ccm_dict = dict(zip(ccm_keys,ccm_values))
df['CCM']=df['CCM'].replace(ccm_dict)
df['CCM']=df['CCM'].astype(int)

#Beygir Gucu optimization
bg_keys = df['Beygir Gucu'].value_counts().index.to_list()
bg_values = [pd.NA,125,100,150,100,75,200,175,225,275,250,300,50,325,350,400,600,475]
bg_dict = dict(zip(bg_keys,bg_values))
df['Beygir Gucu']=df['Beygir Gucu'].replace(bg_dict)
df['Beygir Gucu'] = df['Beygir Gucu'].fillna(
    df.groupby("CCM")['Beygir Gucu'].transform("median")
)


#Kimden Replacement
kimden_keys = df['Kimden'].value_counts().index.to_list()
kimden_values = [1,2,3]
kimden_dict = dict(zip(kimden_keys,kimden_values))
df['Kimden']=df['Kimden'].replace(kimden_dict)
df['Kimden']=df['Kimden'].astype(int)

#Label Encoding Process
encoding_col_list = ["Marka","Arac Tip Grubu","Renk","Kasa Tipi","Yakıt Turu","Vites","Durum"]
encoded_values_dict ={}
for col in encoding_col_list:
    label_encoder = LabelEncoder()
    df[col] = label_encoder.fit_transform(df[col])
    encoded_values_dict[col] = pd.Series(
        data=label_encoder.classes_,
        index=range(len(label_encoder.classes_))
    )
encoding_df = pd.DataFrame(encoded_values_dict)
encoding_df = encoding_df.where(pd.notna(encoding_df),None)
print(df)
data_review(df)
print(df[df['Beygir Gucu'].isna()])
df.drop(index=[1151,1425],inplace=True)
print(df[df['Beygir Gucu'].isna()])

#Definition of X,y Features and Target data
X = df.drop('Fiyat',axis=1)
y = df['Fiyat']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=15)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#scaler for y
scaler_y = StandardScaler()
#converting the y_train in 2D, scaling and convert 1D
y_train_scaled = scaler_y.fit_transform(
    y_train.values.reshape(-1,1)
).ravel()

print("X,y data sets have been defined,scaled and splitted")
#Decision Tree Regression
model_tree = DecisionTreeRegressor(random_state=0)
model_tree.fit(X_train,y_train)
y_pred = model_tree.predict(X_test)
print("Decision Tree Metrics:")
score_tree = r2_score(y_test,y_pred)
mae = mean_absolute_error(y_test,y_pred)
mse = mean_squared_error(y_test,y_pred)
print(f"Score: {score_tree}")
print(f"MAE: {mae}")
print(f"MSE: {mse}")

print("Hyperparameter Tuning for decision Tree...")
param = {
        "criterion": ["squared_error","absolute_error","poisson"],
        "splitter": ["best","random"],
        "max_depth":[1,2,3,4,5,15,None],
        "max_features":["sqrt","log2",None]
}
grid = GridSearchCV(
    estimator=DecisionTreeRegressor(random_state=0),
    param_grid=param,
    cv=5,
    scoring="r2")
grid.fit(X_train,y_train)
y_pred_grid = grid.best_estimator_.predict(X_test)
print("Best Parameters:")
print(grid.best_params_)
score_tree_best = r2_score(y_test,y_pred_grid)
print(score_tree_best)

print("Support Vector Regression...")
svr = SVR()
time_start = time.time()
svr.fit(X_train_scaled,y_train_scaled)
time_end = time.time()
duration = time_end-time_start
print(f"Calculation Time: {duration}")
y_pred_SVR_scaled = svr.predict(X_test_scaled)
#converting the scaled y to orginal
y_pred_SVR=scaler_y.inverse_transform(
    y_pred_SVR_scaled.reshape(-1,1)
).ravel()
mae = mean_absolute_error(y_test,y_pred_SVR)
mse = mean_squared_error(y_test,y_pred_SVR)
score_lineer_SVR = r2_score(y_test,y_pred_SVR)
print(f"Mean Absolute Error (MAE)= {mae}")
print(f"Mean Squared Error (MSE)= {mse}")
print(f"Score= {score_lineer_SVR}")

print("Support Vector Regression Grid Search Hyperparameter Tuning...")
grid_duration = duration*150
if grid_duration>600:
    print(f"Tahmini süre çok usun: {grid_duration:.2f} saniye")
    raise SystemExit("Program Durduruldu")
param_grid = {
    "C": [1, 10, 100],
    "gamma": [0.1, 0.01, 0.001, "scale"],
    "kernel": ["rbf"]
}
grid = GridSearchCV(
    estimator=SVR(),
    param_grid=param_grid,
    scoring="r2",
    n_jobs=-1,
    verbose=1
)
grid.fit(X_train_scaled,y_train_scaled)
y_pred_SVR_CV = grid.best_estimator_.predict(X_test_scaled)
y_pred_SVR_CV_org = scaler_y.inverse_transform(
    y_pred_SVR_CV.reshape(-1,1)
).ravel()
mae = mean_absolute_error(y_test,y_pred_SVR_CV_org)
mse = mean_squared_error(y_test,y_pred_SVR_CV_org)
score_lineer_SVR_CV = r2_score(y_test,y_pred_SVR_CV_org)
print(f"Mean Absolute Error (MAE)= {mae}")
print(f"Mean Squared Error (MSE)= {mse}")
print(f"Score= {score_lineer_SVR_CV}")

print("Linear Regression")
linreg = LinearRegression()
linreg.fit(X_train_scaled,y_train_scaled)
y_pred_linear = linreg.predict(X_test_scaled)
y_pred_linear_org = scaler_y.inverse_transform(
    y_pred_linear.reshape(-1,1)
).ravel()
mae = mean_absolute_error(y_test,y_pred_linear_org)
mse = mean_squared_error(y_test,y_pred_linear_org)
score_lineer_SVR_CV = r2_score(y_test,y_pred_linear_org)
print(f"Mean Absolute Error (MAE)= {mae}")
print(f"Mean Squared Error (MSE)= {mse}")
print(f"Score= {score_lineer_SVR_CV}")


print("Polynomial Regression degree:3")
poly = PolynomialFeatures(degree=3,include_bias=True)
X_train_poly = poly.fit_transform(X_train_scaled)
X_test_poly = poly.transform(X_test_scaled)
poly_reg = LinearRegression()
poly_reg.fit(X_train_poly,y_train_scaled)
y_pred_poly = poly_reg.predict(X_test_poly)
y_pred_poly_org = scaler_y.inverse_transform(
    y_pred_poly.reshape(-1,1)
).ravel()
mae = mean_absolute_error(y_test,y_pred_poly_org)
mse = mean_squared_error(y_test,y_pred_poly_org)
score_lineer_SVR_CV = r2_score(y_test,y_pred_poly_org)
print(f"Mean Absolute Error (MAE)= {mae}")
print(f"Mean Squared Error (MSE)= {mse}")
print(f"Score= {score_lineer_SVR_CV}")

print("Lazy Regressor iterations")
y_test_scaled = scaler_y.transform(
    y_test.values.reshape(-1,1)
).ravel()
reg = LazyRegressor(verbose=0,ignore_warnings=False,custom_metric=None)
models,predictions = reg.fit(X_train_scaled,X_test_scaled,y_train,y_test)
print(models)
