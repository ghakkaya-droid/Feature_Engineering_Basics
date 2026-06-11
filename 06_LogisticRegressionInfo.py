import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from functions import data_info
from sklearn.model_selection import train_test_split,GridSearchCV,StratifiedKFold,RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

print("data reading...")
df = pd.read_csv("6-bank_customers.csv")
print(data_info(df))
print(df.describe())
print(df.head())

print("="*30,"Creating X,y data sets and teaching set","="*30)

X = df.drop(["subscribed"],axis=1)
y = df["subscribed"]
X_train,X_test,y_train,y_test = train_test_split(X,y,train_size=0.7,random_state=15)
logistic = LogisticRegression()
logistic.fit(X_train,y_train)
y_pred = logistic.predict(X_test)
print("Predictions:")
print(y_pred)

print("="*30,"logistic regression metrics","="*30)
score = accuracy_score(y_pred,y_test)
print(f"Accuracy Score: {score}")
print("Classification_Report:")
classification_report1=classification_report(y_pred,y_test)
print(classification_report(y_pred,y_test))
confusion_matrix1 = confusion_matrix(y_test,y_pred)
print("Confusion Matrix:")
print(confusion_matrix(y_pred,y_test))

print("="*30,"trying parameters of logistic regression (hperparameter tuning)","="*30)
model = LogisticRegression()
penalty = ["l1","l2","elasticnet"]
c_values = [100,10,1,0.1,0.001] # This is the beta value in the formula
solver = ["newton-cg","lbfgs","liblinear","sag","saga","newton-cholesky"]
params = dict(penalty=penalty,C=c_values,solver=solver) #Kullanılacak kütüphane parametreleri sözlük olarak istiyor.
print(params)
cv = StratifiedKFold() #k-fold yani data seti eşit parçalara bölme miktarıdır. varsayılan değeri 5
grid:GridSearchCV = GridSearchCV(
    estimator=model, #kullanılacak modeli velirtr.
    param_grid=params, #kullanılacak parametreler.
    cv=cv, #k-fold miktarı. Ancak direk sayı olarak veremiruz. StratifiedKFold() olarak vermemiz gerekir. Varsayılan 5 geliyor. değiştirmek için StratifiedKFold(cv='SAYI') yapabilirsin.
    scoring="accuracy", # Skorlamayı neye göre yapacak. İsterse r2, veya recision(keskinlik) veya recall(duyarlılık), F1, Fbeta değerlerine göre skorlayabilirsin.
    n_jobs=-1  #Bütün CPU kaynaklarını kullanabilir.
)

grid.fit(X_train,y_train)
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

print("Classification Report:")
print(classification_report(y_test, y_pred2))
print(classification_report1)


print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred2))
print(confusion_matrix1)

print("="*30,"Randomized Cross-Validation","="*30)
model = LogisticRegression()
randomcv=RandomizedSearchCV(
    estimator=model,
    param_distributions=params,
    cv=5,
    n_iter=10,
    scoring="accuracy"
)
randomcv.fit(X_train,y_train)
best_model_randomized = randomcv.best_estimator_
#y_pred3 = best_model_randomized.predict(X_test)
y_pred3 = randomcv.predict(X_test)
print("Best Param & Score by RandomizedCV")
print("Best Parameters:")
print(randomcv.best_params_)
print("Best Score:")
print(randomcv.best_score_)

print("Classification Report:")
print("RandomizedCV:")
print(classification_report(y_test, y_pred3))
print("Logistic Regression:")
print(classification_report1)

print("Confusion Matrix:")
print("RandomizedCV:")
print(confusion_matrix(y_test, y_pred3))
print("Logistic Regression:")
print(confusion_matrix1)
