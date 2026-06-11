import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from functions import data_info, data_review
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.svm import SVR
from sklearn.preprocessing import LabelEncoder,StandardScaler

#data set information
'''
price price in US dollars (\$326--\$18,823)
carat weight of the diamond (0.2--5.01)
cut quality of the cut (Fair, Good, Very Good, Premium, Ideal)
color diamond colour, from J (worst) to D (best)
clarity a measurement of how clear the diamond is (I1 (worst), SI2, SI1, VS2, VS1, VVS2, VVS1, IF (best))
x length in mm (0--10.74)
y width in mm (0--58.9)
z depth in mm (0--31.8)
depth total depth percentage = z / mean(x, y) = 2 * z / (x + y) (43--79)
table width of top of diamond relative to widest point (43--95)
'''
#data review
print("="*30,"Data Review","="*30)
df = pd.read_csv("diamonds.csv")
data_info(df)
data_review(df)
#search of x,y,z data = 0 
print("any xyz ==0") 
print(df[(df[['x','y','z']]==0).any(axis=1)].shape[0])

print("="*30,"Data Cleaning and Encoding","="*30)
df_clean = df.drop('Unnamed: 0',axis=1) #meaningless column has been deleted
print(df_clean.columns)
mask = (df_clean[['x','y','z']]==0).any(axis=1)
df_clean = df_clean.drop(df[mask].index,axis=0)
print(df_clean.shape)

#Handling outlier data
print("="*30,"Handling outlier data","="*30)
print("Pair_Plot")
df_numeric = df_clean.select_dtypes(include="number")
#sns.heatmap(df_numeric.corr(),annot=True)
#plt.show()
'''
x_cols=['x','y','z','table','depth']
fig,axes = plt.subplots(2,3,figsize=(15,8))
for ax,col in zip(axes.flatten(),x_cols):
    sns.scatterplot(data=df_clean,x=col,y='price',ax=ax)
    ax.set_title(f'price Vs. {col}')
plt.tight_layout()
plt.show()
'''

'''
    for x >=9; for y>=15;for z>=10; for table>=>70; for depth 50< and 75>
'''
df_clean = df_clean[(df_clean['depth']<75)&(df_clean["depth"]>45)]
df_clean = df_clean[(df_clean['table']<75)&(df_clean["table"]>40)]
df_clean = df_clean[(df_clean['z']<30)&(df_clean["z"]>2)]
df_clean = df_clean[(df_clean['y']<20)]
print(df_clean.shape)
print(df_clean.columns)

print("="*30,"Categorical Datas","="*30)
df_string_data:pd.DataFrame = df_clean.select_dtypes("str")
print("String columns:")
print(df_string_data.columns)
print("String columns value counts")

unique_srt_count_df = pd.DataFrame({
    "Column": df_string_data.columns,
    "Count": [df_string_data[col].nunique() for col in df_string_data.columns]
})
print(unique_srt_count_df)

print("Unique values for column: 'cut':")
print(df_clean['cut'].unique())
print("Unique values for column: 'color':")
print(df_clean['color'].unique())
print("Unique values for column: 'clarity':")
print(df_clean['clarity'].unique())

#X,y definitions
print("="*30,"X,y Definitions, Label Encoding & Scaling","="*30)
X=df_clean.drop('price',axis=1)
y=df_clean['price']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=15)

#Label Encoding
label_encoder = LabelEncoder()
for col in ['cut','color','clarity']:
    X_train[col] = label_encoder.fit_transform(X_train[col])
    X_test[col] = label_encoder.transform(X_test[col])

print("Encoded X_train_head")
print(X_train.head())

#Scaling X data
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(
    scaler.fit_transform(X_train),
    columns=X_train.columns,
    index=X_train.index)
X_test_scaled = pd.DataFrame(
    scaler.transform(X_test),
    columns=X_test.columns,
    index=X_test.index)
print("Scaled X_train Description")
print(X_train_scaled.describe())

print("="*30,"Lİneer Regression","="*30)
lineer = LinearRegression()
lineer.fit(X_train_scaled,y_train)
y_pred_lineer = lineer.predict(X_test_scaled)
mae = mean_absolute_error(y_test,y_pred_lineer)
mse = mean_squared_error(y_test,y_pred_lineer)
score_lineer = r2_score(y_test,y_pred_lineer)
print(f"Mean Absolute Error (MAE)= {mae}")
print(f"Mean Squared Error (MSE)= {mse}")
print(f"Score= {score_lineer}")

#plt.scatter(x=y_test,y=y_pred_lineer)
#plt.show()

'''
# SVR Tekbaşına ilaç olmadı. %49 başarılı oldu.
print("="*30,"SVR model regresion","="*30)
svr = SVR()
svr.fit(X_train_scaled,y_train)
y_pred_SVR = svr.predict(X_test_scaled)
mae = mean_absolute_error(y_test,y_pred_SVR)
mse = mean_squared_error(y_test,y_pred_SVR)
score_lineer = r2_score(y_test,y_pred_SVR)
print(f"Mean Absolute Error (MAE)= {mae}")
print(f"Mean Squared Error (MSE)= {mse}")
print(f"Score= {score_lineer}")

plt.scatter(x=y_test,y=y_pred_lineer)
plt.show()
'''
#GridSearchCV hyperparameter

'''
baya uzun sürüyor. 
en iyi parametreler C=1000, gamma:0.1, kernel:rbf iken %94 küsür çıkıyor.
param_grid = {
    "C": [0.1,1,10,100,1000],
    "gamma": [1,0.1,0.001],
    "kernel":["rbf","linear"]
}
grid = GridSearchCV(
    estimator=SVR(),
    param_grid=param_grid,
    n_jobs=-1,
    verbose=3
)
grid.fit(X_train_scaled,y_train)

'''