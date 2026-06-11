import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from functions import data_info
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split,GridSearchCV,StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier


df = pd.read_csv("7-cyber_attack_data.csv")
print("="*50,"Data review ...","="*50)
print(data_info(df))
print(df.columns)
print(df.head())
print(df.describe())
print("unique_counts of colun:'attack_type")
print(df['attack_type'].value_counts())

print("="*50,"X,y definitions and regressions","="*50)
X = df.drop('attack_type',axis=1)
y = df['attack_type']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train,X_test,y_train,y_test = train_test_split(X_scaled,y,test_size=0.3,random_state=15)
model = LogisticRegression()
start = time.time()
model.fit(X_train,y_train)
end = time.time()
single_fit_time = end-start
y_pred = model.predict(X_test)
print("-"*30,"... Logisic Regression Metrics ...","-"*30)
score = accuracy_score(y_test,y_pred)
print(f"Accuracy Score: {score}")
print("Classification_Report:")
classification_report1=classification_report(y_pred,y_test)
print(classification_report(y_pred,y_test))
confusion_matrix1 = confusion_matrix(y_test,y_pred)
print("Confusion Matrix:")
conf_matrix_1 = confusion_matrix(y_test,y_pred)
print(conf_matrix_1)

print("="*50,"GridSearchCV (Hyperparameter Tuning)","="*50)

model = LogisticRegression()
penalty = ["l1","l2","elasticnet"]
c_values = [100,10,1,0.1,0.001]
solver = ["newton-cg","lbfgs","liblinear","sag","saga","newton-cholesky"]
params = dict(penalty=penalty,C=c_values,solver=solver)
cv = StratifiedKFold()
n_candidates = len(penalty)*len(c_values)*len(solver)
total_fit_count = n_candidates*5
grid:GridSearchCV = GridSearchCV(
    estimator=model,
    param_grid=params,
    cv=cv, 
    scoring="accuracy",
    n_jobs=-1
)
start_real_time = time.time()
grid.fit(X_train,y_train)
finish_real_time = time.time()
real_time_duration = finish_real_time-start_real_time
print("Best Params:")
print(grid.best_params_)

print("Best CV Score:")
print(grid.best_score_)

best_model = grid.best_estimator_
y_pred2 = best_model.predict(X_test)
print("="*30,"logistic regression metrics","="*30)
score2 = accuracy_score(y_test, y_pred2)
print(f"Default Model Test Accuracy: {score}")
print(f"GridSearch Best CV Accuracy: {grid.best_score_}")
print(f"Tuned Model Test Accuracy: {score2}")
print("Confusion Matrix for GridSearchCV:")
conf_matrix_2 = confusion_matrix(y_test, y_pred2)
print(confusion_matrix(y_test, y_pred2))
print("Confusion Matrix for LogisticRegression:")
print(conf_matrix_1)

print("="*50,"One vs Rest","="*50)
one_vs_one_model = OneVsOneClassifier(LogisticRegression())
one_vs_rest_model = OneVsRestClassifier(LogisticRegression())
start_time = time.time()
one_vs_one_model.fit(X_train,y_train)
end_time = time.time()
one_fit_duration_OneVsOne = end_time-start_time
y_pred3= one_vs_one_model.predict(X_test)
score3 = accuracy_score(y_test, y_pred3)
conf_matrix_3 = confusion_matrix(y_test, y_pred3)
start_time = time.time()
one_vs_rest_model.fit(X_train,y_train)
end_time = time.time()
one_fit_duration_OneVsRest = end_time-start_time
y_pred4 = one_vs_rest_model.predict(X_test)
score4 = accuracy_score(y_test, y_pred4)
conf_matrix_4 = confusion_matrix(y_test, y_pred4)

print("="*30,"logistic regression metrics","="*30)
print("Scores according to models")
print(f"Default Model Test Accuracy: {score}")
print(f"Tuned Model Test Accuracy: {score2}")
print(f"Tuned Model Test One vs One: {score3}")
print(f"Tuned Model Test One vs Rest: {score4}")

print("Confusion Matrix according to models")
print("Default Model :\n",
      conf_matrix_1,"\n",
      "GridSearch Model :\n",
      conf_matrix_2,"\n",
      "One vs One Model :\n",
      conf_matrix_3,"\n",
      "One vs Rest Model :\n",
      conf_matrix_4,"\n",
      )

print("="*30,"Calculation and prediction of Regression and Cross-Validation time","="*30)
print(f"One fit duration: {single_fit_time} sn.")
print(f"Estimated duration: {total_fit_count*single_fit_time} sn.")
print(f"Real time duration for Grid_Search: {real_time_duration} sn.")
print(f"Real one fit time for OneVsOne {one_fit_duration_OneVsOne}")
print(f"Real one fit time for OneVsRest {one_fit_duration_OneVsRest}")
