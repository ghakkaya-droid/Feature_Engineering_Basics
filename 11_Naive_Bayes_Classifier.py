import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from functions import data_review
from sklearn.preprocessing import LabelEncoder,StandardScaler,PolynomialFeatures
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression,Ridge,RidgeCV,ElasticNet,ElasticNetCV,Lasso,LassoCV,LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from lazypredict.Supervised import LazyRegressor
from sklearn.svm import SVC,SVR
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score,mean_absolute_error,mean_squared_error,r2_score


print("="*30,"Dara Review & Label_Encoding","="*30)
df = pd.read_csv("Iris.csv")
data_review(df)

df.drop("Id",axis=1,inplace=True)
label_encoder = LabelEncoder()
df['Species']=label_encoder.fit_transform(df['Species'])
encoded_series = pd.Series(label_encoder.classes_)
print(encoded_series)


print("="*30,"X,y definitions and Scaling Data Sets","="*30)
X = df.drop('Species',axis=1)
y = df['Species']
scaler = StandardScaler()
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=15)
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("X,y defined and Scaled...")

print("="*30,"Gaus Navi Baes Regression","="*30)
gnb = GaussianNB()
gnb.fit(X_train_scaled,y_train)
y_pred_gnb = gnb.predict(X_test_scaled)

print("Gaus Navi Baes Metrics:")
score_gnb = accuracy_score(y_test,y_pred_gnb)
conf_matrix_gnb = confusion_matrix(y_test,y_pred_gnb)
classification_rep_gnb = classification_report(y_test,y_pred_gnb)
print("Accuracy Score",score_gnb)
print("Classification Report \n",classification_rep_gnb)
print("Confusion Matrix \n",conf_matrix_gnb)

print("="*30,"Linear Regression","="*30)
linear = LinearRegression()
linear.fit(X_train_scaled,y_train)
y_pred_linear = linear.predict(X_test_scaled)
print("Linear Regression Metrics:")
score_linear = r2_score(y_test,y_pred_linear)
mae = mean_absolute_error(y_test,y_pred_linear)
mse = mean_squared_error(y_test,y_pred_linear)
score_linear = r2_score(y_test,y_pred_linear)
print("Score: ",score_linear)
print("MAE: ",mae)
print("MSE: ",mse)

print("="*30,"Polynomial Regression Various Degree with Pipeline","="*30)
score_poly=[]
for i in range(1,5):
    pipeline = Pipeline([
        ("poly_features",PolynomialFeatures(degree=i)),
        ("lin_reg",LinearRegression())
    ])
    pipeline.fit(X_train_scaled,y_train)
    score_poly_reg = pipeline.score(X_test_scaled,y_test)
    score_poly.append((i,score_poly_reg))

print("Score changes across various polynomial degrees : \n",score_poly)

print("="*30,"Lasso, Ridge, ElasticNet and Cross Validations","="*30)
models = [Ridge(),Lasso(),ElasticNet(),RidgeCV(cv=5),LassoCV(cv=5),ElasticNetCV(cv=5)]
score_poly_CV = []
for model in models:
    pipeline = Pipeline([
        ("lin_reg",model)
    ])
    pipeline.fit(X_train_scaled,y_train)
    score_poly_reg = pipeline.score(X_test_scaled,y_test)
    score_poly_CV.append((model.__class__.__name__,score_poly_reg))

print("Score changes across Ridge,Lasso,ElasticNet and theri Class-Validation : \n",score_poly_CV)

print("="*30,"LazyPredict","="*30)
reg = LazyRegressor(verbose=0,ignore_warnings=False,custom_metric=None)
models_lazy_pred,predictions = reg.fit(X_train_scaled,X_test_scaled,y_train,y_test)
print(models_lazy_pred)

print("="*30,"Logistic Regression","="*30)
logisric = LogisticRegression()
logisric.fit(X_train_scaled,y_train)
y_pred_logistic = logisric.predict(X_test_scaled)
print("logistic Metrics:")
score_logistic = accuracy_score(y_test,y_pred_logistic)
conf_matrix_logistic = confusion_matrix(y_test,y_pred_logistic)
classification_rep_logistic = classification_report(y_test,y_pred_logistic)
print("Accuracy Score",score_logistic)
print("Classification Report \n",classification_rep_logistic)
print("Confusion Matrix \n",conf_matrix_logistic)

print("="*30,"Support Vector Machines - SVC","="*30)
svc = SVC(kernel="linear")
svc.fit(X_train_scaled,y_train)
y_pred_svc = svc.predict(X_test_scaled)

print("SVC Metrics:")
score_svc = accuracy_score(y_test,y_pred_svc)
conf_matrix_svc = confusion_matrix(y_test,y_pred_svc)
classification_rep_svc = classification_report(y_test,y_pred_svc)
print("Accuracy Score",score_svc)
print("Classification Report \n",classification_rep_svc)
print("Confusion Matrix \n",conf_matrix_svc)

print("="*30,"Support Vector Machines - SVR","="*30)
svr = SVR()
svr.fit(X_train_scaled,y_train)
y_pred_svr = svc.predict(X_test_scaled)

print("SVR Regression Metrics:")
score_svr = r2_score(y_test,y_pred_svr)
mae = mean_absolute_error(y_test,y_pred_svr)
mse = mean_squared_error(y_test,y_pred_svr)
score_svr = r2_score(y_test,y_pred_svr)
print("Score: ",score_svr)
print("MAE: ",mae)
print("MSE: ",mse)