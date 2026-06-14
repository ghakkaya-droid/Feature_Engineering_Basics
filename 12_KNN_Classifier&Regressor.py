#Kütüphaneler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from functions import data_review,value_counter
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score,mean_absolute_error,mean_squared_error,r2_score


#Data okuma
def health_data_read():
    print("="*30,"Data Review & Visualition","="*30)
    df = pd.read_csv("12-health_risk_classification.csv")
    data_review(df)
    value_counter(df,'high_risk_flag')
    return df
#Data visualition
def health_visual(df):
    sns.scatterplot(data=df,x='activity_level_index',y='blood_pressure_variation',hue='high_risk_flag')
    plt.xlabel('activity_level_index')
    plt.ylabel('blood_pressure_variation')
    plt.show()
    sns.scatterplot(data=df,x='bmi_score',y='blood_pressure_variation',hue='high_risk_flag')
    plt.xlabel('bmi_score')
    plt.ylabel('blood_pressure_variation')
    plt.show()

#X,y Definition
def X_y_health(df):
    print("="*30,"X,y definitions & Scaling","="*30)
    X=df.drop('high_risk_flag',axis=1)
    y=df['high_risk_flag']
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=15)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("X,y defined & scaled...")
    return X,y,X_train_scaled,X_test_scaled
#KNN Clasifier
def KNN():
    print("="*30,"KNN Clasifier","="*30)
    knn_classifier = KNeighborsClassifier(n_neighbors=5,#En yakın komşu sayısı parametresi
                                      weights="uniform",#Distance: uzaklıklara göre ağırlık alır. Unifor ise her komşunun uzaklığı eşit sayılır. O bakımdan sayısı az ama uzaklığı da az olan komşular seçim kararı olabilir. 
                                      leaf_size=100,#ağaç adım sayısı. Varsayılan 30 arttırırsan belki başarı artar.
                                      p=2)#Eğer 1 yaparsak mesafeler manhattan, 2 olursa öklid
    knn_classifier.fit(X_train_scaled,y_train)
    y_pred_knn_classifier = knn_classifier.predict(X_test_scaled)

    print("KNN Classifier Metrics:")
    score_knn_class = accuracy_score(y_test,y_pred_knn_classifier)
    conf_matrix_knn_class = confusion_matrix(y_test,y_pred_knn_classifier)
    classification_rep_knn_class = classification_report(y_test,y_pred_knn_classifier)
    print("Accuracy Score",score_knn_class)
    print("Classification Report \n",classification_rep_knn_class)
    print("Confusion Matrix \n",conf_matrix_knn_class)

#Gaus Navi
def Gauss():
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

#SVC
def SVC_Classifier():
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
    #SVC_GrigSearch
    print("="*30,"SVC GridSearchCV","="*30)
    param_grid = {
        "C": [0.1,1,10,100,1000],
        "gamma": [1,0.1,0.001],
        "kernel":["rbf","linear","sigmoid"]
    }
    grid = GridSearchCV(
        estimator=SVC(),
        param_grid=param_grid,
        n_jobs=-1
    )
    grid.fit(X_train_scaled,y_train)

    print("Best params:", grid.best_params_)
    print("Best CV score:", grid.best_score_)
    best_model = grid.best_estimator_
    y_pred_grid = best_model.predict(X_test_scaled)

    print("SVC Best by GridSearchCV Metrics:")
    score_grid = accuracy_score(y_test,y_pred_grid)
    conf_matrix_grid = confusion_matrix(y_test,y_pred_grid)
    classification_rep_gird = classification_report(y_test,y_pred_grid)
    print("Accuracy Score",score_grid)
    print("Classification Report \n",classification_rep_gird)
    print("Confusion Matrix \n",conf_matrix_grid)

#House Energy Data Set Reading
print("="*30,"House Energy Data Set Review","="*30)
df_reg = pd.read_csv("12-house_energy_regression.csv")
data_review(df_reg)
value_counter(df_reg,'daily_energy_consumption_kwh')
print(df_reg['daily_energy_consumption_kwh'].value_counts())

#
sns.scatterplot(data=df_reg,x='avg_indoor_temp_change',y='daily_energy_consumption_kwh')
plt.xlabel('avg_indoor_temp_change')
plt.ylabel('daily_energy_consumption_kwh')
plt.show()

# X,y data definitions
X_enrgy = df_reg.drop('daily_energy_consumption_kwh',axis=1)
y_enrgy = df_reg['daily_energy_consumption_kwh']
X_enrgy_train,X_enrgy_test,y_enrgy_train,y_enrgy_test = train_test_split(X_enrgy,y_enrgy,test_size=0.25,random_state=15)
scaler = StandardScaler()
X_enrgy_train_scaled = scaler.fit_transform(X_enrgy_train)
X_enrgy_test_scaled = scaler.transform(X_enrgy_test)
knn_regressor = KNeighborsRegressor()
knn_regressor.fit(X_enrgy_train_scaled,y_enrgy_train)
y_pred_knn_reg = knn_regressor.predict(X_enrgy_test_scaled)
print("KNN_Regressor Regression Metrics:")
score_knnR = r2_score(y_enrgy_test,y_pred_knn_reg)
mae = mean_absolute_error(y_enrgy_test,y_pred_knn_reg)
mse = mean_squared_error(y_enrgy_test,y_pred_knn_reg)
print("Score: ",score_knnR)
print("MAE: ",mae)
print("MSE: ",mse)



