import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from functions import data_review,value_counter
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score,mean_absolute_error,mean_squared_error,r2_score
from sklearn import tree
import warnings

warnings.filterwarnings("ignore")

print("="*30,"File Reading","="*30)
df = pd.read_csv('car_evaluation.csv')
data_review(df)
dict_values = ["buying","maint","doors","persons","lug_boot","safety","class"]
column_description_dict = dict(zip(df.columns,dict_values))
df.rename(columns=column_description_dict,inplace=True)
print(df.head())
print("Unique Values counts in columns")
for col in df.columns:
    value_counter(df,col)

print("="*30,"Feature Engineering","="*30)
df['doors'] = df["doors"].replace('5more','5')
df['doors'] = df['doors'].astype(int)
df['persons'] = df["persons"].replace('more','5')
df['persons'] = df['persons'].astype(int)
print("Columns: doors and persons have been converted to integer type")
data_review(df)

print("="*30,"X,y Definitions","="*30)
print("X,y data sets defined")
X = df.drop('class',axis=1)
y = df['class']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=15)
print("Shape of X_train \n",X_train.shape)

print("="*30,"Ordinal Encoding","="*30)
categorical_columns = ["buying","maint","lug_boot","safety"]
numeric_columns = ["doors","persons"]
ordinal_encoder = OrdinalEncoder(
    categories=[
        ["low","med","high","vhigh"],#buying
        ["low","med","high","vhigh"],#maint
        ["small","med","big"],#lug_boot
        ["low","med","high"]#safety
    ])
preprosessor = ColumnTransformer(transformers=[
    ('transformation_name_not_matter',ordinal_encoder,categorical_columns)
],remainder="passthrough")
#ColumnTransformer geriye numpy array döndürür.
X_train_transformed = pd.DataFrame(preprosessor.fit_transform(X_train))
X_test_transformed = pd.DataFrame(preprosessor.transform(X_test))
print("X_train Colums:\n",X_train.columns)
print(X_train_transformed)
#Önce Tranformation'a soktuğun kolonlar çıkar sonrasında ise tranformationa girmeyen kolonlar gelir (remainder="passthrough" dediğimiz için. Eğer bunu demeseydik sadece transformationa girenler gelecekti)
transformed_columns = categorical_columns + numeric_columns
X_train_transformed.columns = transformed_columns
X_test_transformed.columns = transformed_columns
print(X_train_transformed)

print("="*30,"Tree decision Classifier","="*30)
tree_model = DecisionTreeClassifier(
    criterion="gini",
    max_depth=3,
    random_state=0
)
tree_model.fit(X_train_transformed,y_train)
y_pred = tree_model.predict(X_test_transformed)
print("Decision Tree Metrics:")
score_tree = accuracy_score(y_test,y_pred)
conf_matrix_tree = confusion_matrix(y_test,y_pred)
classification_tree = classification_report(y_test,y_pred)
print(f"max depth: {tree_model.max_depth}")
print("Accuracy Score",score_tree)
print("Classification Report \n",classification_tree)
print("Confusion Matrix \n",conf_matrix_tree)

'''
print("="*30,"Plotting the Tree","="*30)
plt.figure(figsize=(12,8))
tree.plot_tree(tree_model.fit(X_train_transformed,y_train))
plt.show()
'''

print("="*30,"Tree decision Hyperparameter tuning","="*30)
param = {
        "criterion": ["gini","entropy","log_loss"],
        "splitter": ["best","random"],
        "max_depth":[1,2,3,4,5,15,None],
        "max_features":["sqrt","log2",None]
}

grid = GridSearchCV(
    estimator=DecisionTreeClassifier(),
    param_grid=param,
    cv=5,
    scoring="accuracy")

grid.fit(X_train_transformed,y_train)

y_pred_grid = grid.best_estimator_.predict(X_test_transformed)

print("Decision Tree Tuning Metrics:")
print("Best Parameters:")
print(grid.best_params_)
score_tree_best = accuracy_score(y_test,y_pred_grid)
conf_matrix_tree_best = confusion_matrix(y_test,y_pred_grid)
classification_tree_best = classification_report(y_test,y_pred_grid)
print(f"max depth: {grid.best_estimator_.max_depth}")
print("Accuracy Score",score_tree_best)
print("Classification Report \n",classification_tree_best)
print("Confusion Matrix \n",conf_matrix_tree_best)


