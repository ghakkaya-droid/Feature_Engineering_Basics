import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from functions import data_info, data_review
import time

#e-mail classification
print("="*30,"e-mail classification","="*30) 
df = pd.read_csv("9-email_classification_svm.csv")
print(df.columns)
print(df.head())
print(data_info(df))
print(df.describe())
print(df['email_type'].value_counts(normalize=True)*100)

#X,y definition and train-test split also test predictions
X = df.drop('email_type',axis=1)
y = df['email_type']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=15)
svs = SVC(kernel="linear")
svs.fit(X_train,y_train)
y_pred_lineer = svs.predict(X_test)

#model reports and score metrics
print(classification_report(y_test,y_pred_lineer))
print(confusion_matrix(y_test,y_pred_lineer))

#loan-risc data_set
print("="*30,"loan-risc data_set","="*30) 
df_2 = pd.read_csv("9-loan_risk_svm.csv")
data_review(df_2)
print(df_2['loan_risk'].value_counts(normalize=True)*100)
#visualition of data
#sns.scatterplot(x=df_2["credit_score_fluctuation"],y=df_2['recent_transaction_volume'],hue=df_2["loan_risk"])
#plt.show()
#X,y definition and train-test split also test predictions
X = df_2.drop("loan_risk",axis=1)
y = df_2["loan_risk"]
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=15)
accuracy=[]
confusion_matrix_list = []
#SVC_Lineer
lineer = SVC(kernel="linear")
lineer.fit(X_train,y_train)
y_pred_lineer2=lineer.predict(X_test)
#model reports and score metrics
print("Metrics for #SVC_Lineer:")
print(accuracy_score(y_test,y_pred_lineer2))
print(classification_report(y_test,y_pred_lineer2))
print(confusion_matrix(y_test,y_pred_lineer2))
accuracy.append(accuracy_score(y_test,y_pred_lineer2))
confusion_matrix_list.append(confusion_matrix(y_test,y_pred_lineer2))


#SVC_rbf
rbf = SVC(kernel="rbf")
rbf.fit(X_train,y_train)
y_pred_rbf2=rbf.predict(X_test)
#model reports and score metrics
print("Metrics for #SVC_RBF:")
print(accuracy_score(y_test,y_pred_rbf2))
print(classification_report(y_test,y_pred_rbf2))
print(confusion_matrix(y_test,y_pred_rbf2))
accuracy.append(accuracy_score(y_test,y_pred_rbf2))
confusion_matrix_list.append(confusion_matrix(y_test,y_pred_rbf2))

#SVC_poly
poly = SVC(kernel="poly")
poly.fit(X_train,y_train)
y_pred_poly2=poly.predict(X_test)
#model reports and score metrics
print("Metrics for #SVC_Poly:")
print(accuracy_score(y_test,y_pred_poly2))
print(classification_report(y_test,y_pred_poly2))
print(confusion_matrix(y_test,y_pred_poly2))
accuracy.append(accuracy_score(y_test,y_pred_poly2))
confusion_matrix_list.append(confusion_matrix(y_test,y_pred_poly2))

print(accuracy)
print(confusion_matrix_list)

#Hyperparameter Tuning
print("="*30,"Hyperparameter Tuning","="*30)
params_grid = {
    "C":[0.1,1,10,100,1000],
    "kernel":["rbf"],
    "gamma":["scale","auto"],
}

grid = GridSearchCV(
    estimator=SVC(),
    param_grid=params_grid,
    cv=5
)
start_time = time.time()
grid.fit(X_train,y_train)
finish_time = time.time()
duration_time = finish_time-start_time
print("Best Parameters in Grid_SVC")
print(f"Grid Fit Duration: {duration_time}")
print(grid.best_params_)
print("Best Prediction and Metrics")
y_pred_grid_best = grid.predict(X_test)
print("Metrics for Grid_BEST:")
print(accuracy_score(y_test,y_pred_grid_best))
print(classification_report(y_test,y_pred_grid_best))
print(confusion_matrix(y_test,y_pred_grid_best))

print("="*30,"seismic_activity","="*30)
df_3 = pd.read_csv("9-seismic_activity_svm.csv")
data_review(df_3)
print(df_3['seismic_event_detected'].value_counts(normalize=True)*100)

#visualition of data
#sns.scatterplot(x=df_3["underground_wave_energy"],y=df_3['vibration_axis_variation'],hue=df_3["seismic_event_detected"])
#plt.show()
#Additional Diametion input to dataset
print("="*30,"kernel 3D model","="*30)
df_3["WE_sq"]=df_3["underground_wave_energy"]**2
df_3["AV_sq"] = df_3["vibration_axis_variation"]**2
df_3["WE*AV"] = df_3["underground_wave_energy"]*df_3["vibration_axis_variation"]

#X,y definition and train-test split also test predictions
X_3D = df_3.drop("seismic_event_detected",axis=1)
y_3D = df_3["seismic_event_detected"]
X_train,X_test,y_train,y_test = train_test_split(X_3D,y_3D,test_size=0.25,random_state=15)
#fig=px.scatter_3d(df_3,x="WE_sq",y="AV_sq",z="WE*AV",color="seismic_event_detected")
#fig.show()

#SVC_Lineer for 3D kernel model

lineer_3D = SVC(kernel="linear")
lineer_3D.fit(X_train,y_train)
y_pred_lineer3D=lineer_3D.predict(X_test)
#model reports and score metrics
print("Metrics for #SVC_Lineer:")
print(accuracy_score(y_test,y_pred_lineer3D))
print(classification_report(y_test,y_pred_lineer3D))
print(confusion_matrix(y_test,y_pred_lineer3D))
