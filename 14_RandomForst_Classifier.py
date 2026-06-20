import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from functions import data_review,value_counter,numeric_encoding
from sklearn.model_selection import train_test_split,RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder,RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

bold = "\033[1m"
reset = "\033[0m"

print("="*30,"Data reading & review","="*30)
df= pd.read_csv("income_evaluation.csv")
data_review(df)
print(f"Data Frame NanRow_Count: {len(df[df.isna().any(axis=1)])}")
#column name adjustment
adj_column_name = []
for idx,col in enumerate(df.columns):
    col = col.strip()
    if "-" in col:
        col = col.replace("-","_")
    adj_column_name.append(col)
df.columns = adj_column_name
df.rename(columns={"fnlwgt":"finalweigth"},inplace=True)
print(df.columns)
#Duplicated columns
print(f"duplicated row number:{df[df.duplicated].shape[0]}")
print(df[df.duplicated])
df.drop_duplicates(inplace=True)
print(f"duplicated row number:{df[df.duplicated].shape[0]}")
#Categorical Columns
categorical = df.select_dtypes(include=["str","object"]).columns
numerical = df.select_dtypes(include="number").columns
print(f"Categorical Columns: \n{categorical}")
print(f"Numerical columns: \n{numerical}")

print(df.select_dtypes(include="str").head()) #print(df[categorical].head()) ikisi aynı şey
#Unique counst of categorical columns
'''
for col in categorical:
    print(df[col].value_counts())
    print(len(df[col].value_counts()))
'''

print("="*30,"Data Cleaning","="*30)
#col: workclass cleaning
print("Column: Workclass data adjusting")
print(df["workclass"].unique())
data_column = list(df["workclass"].unique())
data_adj =[]
for data in data_column:
    data = data.strip()
    data_adj.append(data)
df['workclass'] = df['workclass'].replace(data_column,data_adj)
#Yukarıdaki kodun kısa hali: df[workclass] = df['workclass'].str.strip()
df['workclass'] = df['workclass'].replace("?",np.nan)
print(df["workclass"].value_counts(dropna=False))
#col: occupation cleaning
print("Column: occupation data adjusting")
print(df["occupation"].unique())
df['occupation'] = df['occupation'].str.strip()
df['occupation'] = df['occupation'].replace("?",np.nan)
print(df["occupation"].value_counts(dropna=False))
#col: native_country cleaning
print("Column: native_country data adjusting")
print(df["native_country"].unique())
df['native_country'] = df['native_country'].str.strip()
df['native_country'] = df['native_country'].replace("?",np.nan)
print(df["native_country"].value_counts(dropna=False))
#data null control
print("Null data counts")
print(df.isnull().sum())

print("="*30,"Encoding & one-hot Encoding","="*30)
X = df.drop('income',axis=1)
y = df['income']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=15)
categorical_X = X.select_dtypes(include='str').columns
numerical_X =X.select_dtypes(include='number').columns
print("X_train null:")
print(X_train[categorical_X].isnull().sum())
print("X_test null:")
print(X_test[categorical_X].isnull().sum())
#to fill the NA values we will use mode value (this value is the most counted value in sets)
print("columns 'workclass' mode value:")
print(X_train['workclass'].mode()[0])
print("Replacing Na values with column mode value")
for col in X_train.columns:
    col_mode_train = X_train[col].mode()[0]
    X_train[col] = X_train[col].fillna(col_mode_train)
    X_test[col] = X_test[col].fillna(col_mode_train)
print("X_train null:")
print("-"*80)
print(X_train[categorical_X].isnull().sum())
print("-"*80)
print("X_test null:")
print("-"*80)
print(X_test[categorical_X].isnull().sum())
print("-"*80)
print(f"{bold}Unique values in columns df:{reset}")
print("-"*80)
print(df[categorical_X].nunique())
print("-"*80)
'''
    If we use one-hot encoding on all columns, only 'native_country' will create additional 41 columns
'''
y_train_binary:pd.DataFrame = y_train.apply(lambda x: 1 if x.strip()=='>50K' else 0)
print("y binary")
print(y_train_binary)
print("calculation of native_countries mean value by income")
print("-"*80)
print(y_train_binary.groupby(X_train['native_country']).mean())
target_means = y_train_binary.groupby(X_train['native_country']).mean()
X_train['native_country_encoded']=X_train['native_country'].map(target_means) #creating a new column according to means
X_train['native_country_encoded']=X_train['native_country_encoded'].fillna(y_train_binary.mean())
X_test['native_country_encoded']=X_test['native_country'].map(target_means)
X_test['native_country_encoded']=X_test['native_country_encoded'].fillna(y_train_binary.mean())
X_train.drop('native_country',axis=1,inplace=True)
X_test.drop('native_country',axis=1,inplace=True)
print(X_train.head())
one_hot_categories =['workclass', 'education', 'marital_status', 'occupation','relationship', 'race', 'sex']

encoder = ColumnTransformer(
    transformers=[
        ('cat',OneHotEncoder(handle_unknown="ignore",sparse_output=False),one_hot_categories)
    ],remainder="passthrough"
)

X_train_encoded = encoder.fit_transform(X_train)
X_test_encoded = encoder.transform(X_test)

columns_encoded = encoder.get_feature_names_out()
print(columns_encoded)
X_train = pd.DataFrame(X_train_encoded,columns=columns_encoded,index=X_train.index)
X_test = pd.DataFrame(X_test_encoded,columns=columns_encoded,index=X_test.index)
print("X_train encoded")
print(X_train.head())
columns_with_all = X_train.columns
print(columns_with_all)

scaler = RobustScaler() # Çıktı numpy array olacak o yüzden tekrar df'e çeviriyoruz
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_train = pd.DataFrame(X_train_encoded,columns=columns_with_all)
X_test = pd.DataFrame(X_test_encoded,columns=columns_with_all)

print("="*30,"Training Random Forest Classifier","="*30)
print("estimator number: 10")
rfc = RandomForestClassifier(n_estimators=10,random_state=15)
'''
    n_estimators: How many decision tree will be use
'''
rfc.fit(X_train,y_train)
y_pred = rfc.predict(X_test)
print(f"Accuracy Score: {accuracy_score(y_test,y_pred)}")
print(f"Confusion Matrix: \n{confusion_matrix(y_test,y_pred)}")
print(f"Classification Report: \n{classification_report(y_test,y_pred)}")

print("estimator number: 100")
rfc = RandomForestClassifier(n_estimators=100,random_state=15)
'''
    n_estimators: How many decision tree will be use
'''
rfc.fit(X_train,y_train)
y_pred = rfc.predict(X_test)
print(f"Accuracy Score: {accuracy_score(y_test,y_pred)}")
print(f"Confusion Matrix: \n{confusion_matrix(y_test,y_pred)}")
print(f"Classification Report: \n{classification_report(y_test,y_pred)}")

#feature importances 
print("Feature Importances:")
print(f"column lenght: {len(rfc.feature_importances_)}")
'''
Converting the feature importance to df
feature_imp_df = pd.DataFrame({
    "Featur":X_train.columns,
    "Importance":rfc.feature_importances_
})
feature_imp_df = feature_imp_df.sort_values("Importance",ascending=False)
print(feature_imp_df)
'''
# Also can be converte to series using pd.series maybe convert to df by reset index
feature_scores = pd.Series(rfc.feature_importances_,index=X_train.columns).sort_values(ascending=False)
print(feature_scores)
# featue_scores tail values maybe 10 row can drop and trying rfc again
print("Dropping 10 not important features and trying randomForest again")
not_important_features=feature_scores.tail(10).index
X_train = X_train.drop(not_important_features,axis=1)
X_test = X_test.drop(not_important_features,axis=1)
rfc = RandomForestClassifier(n_estimators=100,random_state=15)
rfc.fit(X_train,y_train)
y_pred = rfc.predict(X_test)
print(f"Accuracy Score: {accuracy_score(y_test,y_pred)}")
print(f"Confusion Matrix: \n{confusion_matrix(y_test,y_pred)}")
print(f"Classification Report: \n{classification_report(y_test,y_pred)}")

print("="*30,"Hyperparameter Tuning in RandomForest","="*30)

rf_params = {
    "n_estimators":[100,200,500,1000],
    "max_depth":[5,8,10,15,None],
    "max_features":["sqrt","log2",5,6,7,8],
    "min_samples_split":[2,8,15,20]
}

rfc =RandomForestClassifier()
rscv = RandomizedSearchCV(
    estimator=rfc,
    param_distributions=rf_params,
    cv=3,
    n_jobs=-1
)

rscv.fit(X_train,y_train)
y_pred = rscv.predict(X_test)
print(f"Best Parameters:\n {rscv.best_params_}")
print(f"Accuracy Score: {accuracy_score(y_test,y_pred)}")
print(f"Confusion Matrix: \n{confusion_matrix(y_test,y_pred)}")
print(f"Classification Report: \n{classification_report(y_test,y_pred)}")