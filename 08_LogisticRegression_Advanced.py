import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from functions import data_info
from sklearn.model_selection import train_test_split,GridSearchCV,StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix,classification_report,roc_curve,roc_auc_score
import warnings
import time

print("="*50,"Data read & review","="*50)
df = pd.read_csv("8-fraud_detection.csv")
print(df.columns)
print(df.head())
print(data_info(df))
print(df['is_fraud'].value_counts(normalize=True)*100)
print("y set is imbalanced dataset")


print("="*50,"X,y set creation","="*50)
X = df.drop("is_fraud",axis=1)
y = df["is_fraud"]
#sns.scatterplot(x=X['transaction_amount'],y=X['transaction_risk_score'],hue=y)
#plt.show()
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=15)

print("="*50,"Default Logistic Regression","="*50)
logistic = LogisticRegression()
logistic.fit(X_train,y_train)
y_pred_logistic = logistic.predict(X_test)
print("Score: ",accuracy_score(y_test,y_pred_logistic))
print(classification_report(y_test,y_pred_logistic))
print("Confision Matrix :\n",confusion_matrix(y_test,y_pred_logistic))


print("="*50,"Class_Weight Logistic Regression","="*50)
model = LogisticRegression()
penalty = ["l1","l2","elasticnet"]
c_values = [100,10,1,0.1,0.001]
solver = ["newton-cg","lbfgs","liblinear","sag","saga","newton-cholesky"]
class_weight = [{0:w,1:y} for w in [1,10,50,100] for y in [1,10,50,100]]
params = dict(penalty=penalty,C=c_values,solver=solver,class_weight=class_weight)
cv = StratifiedKFold()
grid : GridSearchCV = GridSearchCV(
    estimator= model,
    param_grid= params,
    cv = cv,
    scoring="accuracy",
    n_jobs=-1
)
warnings.filterwarnings("ignore")
time_start=time.time()
grid.fit(X_train,y_train)
time_finih =time.time()
time_duration = time_finih-time_start
y_pred = grid.predict(X_test)
#printing metrics of logistic regression
print(f"Calculation Duration: {time_duration} sn.")
print("Best Parameters: \n",grid.best_params_)
print("Score: ",accuracy_score(y_test,y_pred))
print(classification_report(y_test,y_pred))
print("Confision Matrix :\n",confusion_matrix(y_test,y_pred))

#ROC (Receiver Operating Characteristic) & AUC (Area Under the Curve)
print("="*50,"Probablities","="*50)
model_prob_array = grid.predict_proba(X_test) #Bu array her bir y değeri için sonucun ne olacağında dair olasılık verir.
model_prob = model_prob_array[:,1] # bu ise pozirif sınıf için olasılıkları aldık. Matrisin 2. Kolonu
model_auc = roc_auc_score(y_test,model_prob) # grafiğin integrali yani altta kalan alan. Eğer 1 olursa mükemmel ideal ütopik olan
print(f"model AUC: {model_auc}")
model_fpr,model_tpr,model_thresholds = roc_curve(y_test,model_prob) # bu bize bir array içerisinde 3 adet dizi verecek. Bunlar fpr,tpr,thresholds
#plotting ROC curve 
plt.plot(model_fpr,model_tpr,marker=".",label="Logistic")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.show()



