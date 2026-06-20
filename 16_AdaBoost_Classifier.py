import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from functions import data_review,value_counter,numeric_encoding
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import AdaBoostClassifier,RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

#LogisticRegression,SVC,GaussianNB,KNeighborsClassifier,RandomForestClassifier,AdaBoostClassifier

print("="*30,"Data reading & review","="*30)
df=pd.read_csv("diabetes.csv")
data_review(df)
print(f"Data Frame NanRow_Count: {len(df[df.isna().any(axis=1)])}")
print("-"*60)
print("Column zero values")
zero_list=[]
total_count=df.shape[0]
for col in df.columns:
    zero_count = (df[col]==0).sum()
    zero_list.append({
        "column":col,
        "count":zero_count,
        "percentage":round(zero_count/total_count*100,2)
    })
zero_df = pd.DataFrame(zero_list)
print(zero_df)
'''
#sns.stripplot
sns.stripplot(data=df,x='Outcome',y='DiabetesPedigreeFunction',jitter=True,alpha=0.6)
plt.title('Pedegree Değerleri (Outcome\'a Göre)')
plt.show()
'''

print("="*30,"X,y Definications","="*30)
X = df.drop('Outcome',axis=1)
y = df['Outcome']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=15)
col_2_fill =['Glucose','BloodPressure','SkinThickness','Insulin','BMI']
medians_dict = {}
for col in col_2_fill:
    col_median = X_train[X_train[col]!=0][col].median()
    medians_dict[col]=col_median
    X_train[col] = X_train[col].replace(0,col_median)
for col in col_2_fill:
    X_test[col] = X_test[col].replace(0,medians_dict[col])

print(X_train.describe())

print("="*30,"Standard Scaling (for AdaBoost it not necessery to scale)","="*30)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print("Scaling X_tain, X_test completed ...")
print("="*30,"Adaboost Solutions","="*30)
ada = AdaBoostClassifier()
ada.fit(X_train,y_train)
y_pred = ada.predict(X_test)
print(classification_report(y_test,y_pred))
print(accuracy_score(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))

print("="*30,"Adaboost Hyperparameter Tuning","="*30)
adaboost_param = {
    "n_estimators":[50,70,100,120,200],
    "learning_rate":[0.001,0.01,1,10]
}
grid = GridSearchCV(
    estimator=AdaBoostClassifier(),
    param_grid=adaboost_param,
    cv=5,
    verbose=1,
    n_jobs=-1
)
grid.fit(X_train,y_train)
print(grid.best_params_)
grid.best_estimator_.fit(X_train,y_train)
y_pred=grid.predict(X_test)
print(classification_report(y_test,y_pred))
print(accuracy_score(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))

#Homework  Calculate the scores by using other classification methods
#LogisticRegression,SVC,GaussianNB,KNeighborsClassifier,RandomForestClassifier,AdaBoostClassifier
print("="*30,"Homework - Evaluating the data set by various classifiers","="*30)
def calculate_model_metrics (true,predict):
    cls_report = classification_report(true,predict)
    acc_score = accuracy_score(true,predict)
    conf_matrix = confusion_matrix(true,predict)
    return cls_report,acc_score,conf_matrix 

models ={
    "Logistic Regression": LogisticRegression(),
    "Support Vector Classifier": SVC(),
    "Naive Bayes":GaussianNB(),
    "K-Neigbours": KNeighborsClassifier(),
    "Random Forest Classifier":RandomForestClassifier(),
    "Ada Boost":AdaBoostClassifier()
}
result_scores=[]

for name,model in models.items():
    model.fit(X_train,y_train)
    y_trian_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    model_train_report,model_train_score,model_train_matrix = calculate_model_metrics(y_train,y_trian_pred)
    model_test_report,model_test_score,model_test_matrix = calculate_model_metrics(y_test,y_test_pred)
    result_score_dict = {
        "model":name,
        "train score":model_train_score,
        "test score":model_test_score
    }
    result_scores.append(result_score_dict)
    print(name)
    print("Evaluation for Training Set")
    print(model_train_report)
    print(model_train_score)
    print(model_train_matrix)
    print("-"*60)
    print("Evaluation for Test Set")
    print(model_test_report)
    print(model_test_score)
    print(model_test_matrix)
    print("-"*60)
result_score_df = pd.DataFrame(result_scores)
print(result_score_df.sort_values('test score'))
