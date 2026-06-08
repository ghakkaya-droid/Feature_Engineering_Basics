'''
    Ödevin amacı: Lineer Regresyon ile alakalı olarak şu ana kadar öğrendiklerimi pekiştirmek
    Öğrendiklerim:
        Veriyi Pandas ile okuma
        Veriyi EDA ile anlama ve Temizleme
        Veriyi standardize etme
        Linner Regresyon
        Polinom Regresyon
        Pipeline ile regresyon otomasyonu
        Veriyi regresyon modellerine sokarak eğitme:
            LinearRegression
            Lasso
            Ridge
            ElasticNet
        Veriyi Eğitirken Cross-Validation ile:
            LassoCV
            RidgeCV
            ElasticNetCV ile Skoru iyileştirme
'''
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
from lazypredict.Supervised import LazyRegressor

#Veri Kaggle üzerinden alınan 'Lineer Regression CO2 Emissions' data seti üzerinde çalışıldı

print("="*30,"Read and review the dataset","="*30)
df = pd.read_csv("co2.csv")
print(data_info(df))
print(df.describe())
print(df.head())

print("="*30,"unique values for str columns","="*30)
df_column_info = data_info(df)
df_column_info_str = df_column_info[df_column_info["Dtype"] == "str"]
column_str_list = list(df_column_info_str["Columns"])
unique_count = []
for col in column_str_list:
    unique_count.append(len(df[col].unique()))
df_column_info_str["Unique_count"] = unique_count
print(df_column_info_str)

print("-"*30,"Model counts by Make","-"*30)
print(df.groupby("Make")["Model"].count().sort_values(ascending=False))
print("-"*30,"Vehicle Class Types","-"*30)
vehicle_class_list=list(df["Vehicle Class"].unique())
vehicle_class_dict = {}
for idx,vehicle in enumerate(vehicle_class_list):
    vehicle_class_dict[vehicle] = idx
df["Vehicle_Class_Num"] = df["Vehicle Class"].map(vehicle_class_dict)
print(df)

print("-"*30,"relation of Vehicle Class between emission","-"*30)
grpby_Class = df.groupby("Vehicle Class")["CO2 Emissions(g/km)"].mean().sort_values(ascending=True).reset_index()
print(grpby_Class)

grpby_Class_dict = dict(
    zip(
        grpby_Class["Vehicle Class"],
        grpby_Class.index
    )
)
df["Vehicle_Class_Num"] = df["Vehicle Class"].map(grpby_Class_dict)
print("="*30,"numeric columns","="*30)
df_numeric_list = list(df.select_dtypes(include="number").columns)
print(df_numeric_list)

print("-"*30,"Definition of independent and dependent variables","-"*30)
df_numeric = df.drop(column_str_list,axis=1)
print(df_numeric)
X = df_numeric.drop("CO2 Emissions(g/km)",axis=1)
y = df_numeric["CO2 Emissions(g/km)"]

def correlation_for_dropping(df,threshold):
    column_to_drop = set()
    corr = df.corr()
    for i in range(len(corr.columns)):
        for j in range(i):
            if abs(corr.iloc[i,j]) > threshold:
                column_to_drop.add(corr.columns[i])
    return column_to_drop
col_drop_corr_list = correlation_for_dropping(X,0.85)
print(col_drop_corr_list)

X = X.drop(col_drop_corr_list,axis=1)
print(X)

print("="*30,"train-test split & standardization","="*30)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=15)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#Scaled data graph comperation
'''
plt.subplots(figsize=(15,5))
plt.subplot(1,2,1)
sns.boxplot(data=X_train)
plt.title("X_train")
plt.subplot(1,2,2)
sns.boxenplot(data=X_train_scaled)
plt.title("X_train_scaled")
plt.show()
'''
print("="*30,"Pipeline Function Creation and choise of model","="*30)
models = {
    "Linear Regression": LinearRegression(),
    "Lasso": Lasso(),
    "Ridge": Ridge(),
    "ElasricNet": ElasticNet(),
    "LassoCV": LassoCV(cv=5),
    "RidgeCV":RidgeCV(cv=5),
    "ElasticNetCV":ElasticNetCV(cv=5)
}
results=[]
for name,model in models.items():
    pipeline = Pipeline([
        ("model", model)
    ])
    pipeline.fit(X_train_scaled,y_train)
    y_pred = pipeline.predict(X_test_scaled)
    r2 = r2_score(y_test,y_pred)
    mae = mean_absolute_error(y_test,y_pred)
    mse = mean_squared_error(y_test,y_pred)
    rmse = np.sqrt(mse)

    results.append({
        "Model":name,
        "R2": r2,
        "MAE": mae,
        "MSE": mse,
        "RMSE":rmse
    })
result_df = pd.DataFrame(results).sort_values("R2",ascending=False)
print(result_df)
print(f"Selected model is: {result_df["Model"][0]}")


selected_model = models[result_df["Model"][0]]
y_pred = selected_model.predict(X_test_scaled)

plot_df = pd.DataFrame({
    "X_test":X_test["Engine Size(L)"],
    "y_test":y_test,
    "y_pred":y_pred
})

sns.scatterplot(data=plot_df,x="X_test",y="y_test",label="Real_Values")
sns.scatterplot(data=plot_df,x="X_test",y="y_pred",label="Predictions")
plt.xlabel("Engine Size(L)")
plt.ylabel("CO2 Emissions(g/km)")
plt.legend()
plt.show()
