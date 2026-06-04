import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from functions import data_info

pd.set_option("display.float_format", "{:,.2f}".format)


df = pd.read_csv("17-googleplaystore.csv")
#missing data
print(df.head(10))
print(data_info(df))
print("-"*30,"Reviews column value_counts","-"*30)
print(df['Reviews'].value_counts())

#converting string data column to numeric (to_numeric)
print("="*30,"converting string data column to numeric","="*30)
print("-"*30,"to_numeric & .isna()","-"*30)
df["Reviews_num"]=pd.to_numeric(df['Reviews'],errors="coerce")
print(df[df["Reviews_num"].isna()]) # Hangi değerler dönüştürelemiyor

print("-"*30,"'Data'['Kolon'].str.isnumeric()","-"*30)
print(df[df["Reviews"].str.isnumeric()==False])
sum_numeric = df["Reviews"].str.isnumeric().sum()
sum_non_numeric = df["Reviews"].count() - sum_numeric
print(f"sum of numeric values: {sum_numeric}")
print(f"sum of numeric values: {sum_non_numeric}")
print(df[~df["Reviews"].str.isnumeric()])


#cleanin Data
print("="*30,"Cleanin the data_set","="*30)
print("-"*30,"take a copy of the original data","-"*30)
df_clean = df.copy()
print("Data has been coppied")
print("-"*30,"data update for column:'Reviews'","-"*30)
df_clean.loc[10472,"Reviews"] = "3"
df_clean["Reviews"] = pd.to_numeric(df_clean["Reviews"],errors="coerce")
print(df_clean)
print("Data has been adjusted")
df_clean.drop("Reviews_num",axis=1,inplace=True)
print(df_clean.describe())

print("-"*30,"review df_clean","-"*30)
print(data_info(df_clean))
print("-"*30,"column: 'Size' review and cleaning","-"*30)
print(df_clean['Size'].value_counts())

print("-"*30,"Replace of strings","-"*30)
df_clean['Size'] = df_clean['Size'].str.replace("M","000")
df_clean['Size'] = df_clean['Size'].str.replace("k","")
df_clean['Size'] = df_clean['Size'].replace("Varies with device",np.nan)
df_clean['Size'] = pd.to_numeric(df_clean["Size"],errors="coerce")
#df_clean['Size'] = df_clean["Size"].astype(float)
print(f"'M' and 'k' values has been replaced ...")
print(df_clean[df_clean["Size"].astype(str).str.contains("1,000", na=False)])
print(data_info(df_clean))

print("-"*30,"column: 'Installs' review and cleaning","-"*30)
print("Value_count")
print(df_clean["Installs"].value_counts())

print("-"*30,"Replace of strings for column:'Installs' ","-"*30)
df_clean['Installs'] = df_clean['Installs'].str.replace("+","")
df_clean['Installs'] = df_clean['Installs'].str.replace(",","")
df_temp = pd.DataFrame()
df_temp["Installs"] = df_clean["Installs"].copy()
df_temp["Installs_num"] = None
for i in df_temp.index:
    try:
        df_temp.loc[i,"Installs_num"] = float(df_temp.loc[i,"Installs"])
    except:
        df_temp.loc[i,"Installs_num"] = df_temp.loc[i,"Installs"]
print(f"hatalı indeksler: {list(df_temp[df_temp["Installs_num"].apply(type) == str].index)}")
print("-"*30,"dropping the non numeric index","-"*30)
df_clean.drop(index=10472,inplace=True)
df_clean['Installs'] = df_clean['Installs'].astype("float")

print("-"*30,"Replace of strings for column:'Price' ","-"*30)
print("Unique values in column 'Price'")
print(df_clean['Price'].unique())
df_clean["Price"] = df_clean["Price"].str.replace("$","")
df_temp = pd.DataFrame()
df_temp["Price"] = df_clean["Price"].copy()
df_temp["Price_num"] = None
for i in df_temp.index:
    try:
        df_temp.loc[i,"Price_num"] = float(df_temp.loc[i,"Price"])
    except:
        df_temp.loc[i,"Price_num"] = df_temp.loc[i,"Price"]
print(f"hatalı indeksler: {list(df_temp[df_temp["Price_num"].apply(type) == str].index)}")
df_clean['Price'] = df_clean['Price'].astype("float")
print("Columns: 'Installs' and 'Price' have been adjusted ...")
print(data_info(df_clean))

# cleaning identified characters by using for loop
'''
char_clean_list = ["+",",","$"]
column_list = ["Installs","Price"]
for col in column_list:
    for char in char_clean_list:
        df_clean[col] = df_clean[col].str.replace(char,"")
'''
print("-"*30,"sub describe of df","-"*30)
print(df_clean.describe())


print("="*30,"converting date type values","="*30)
print("-"*30,"unique items for column: Last Updated ","-"*30)
df_clean["Last Updated"] = pd.to_datetime(df_clean["Last Updated"])
print(data_info(df_clean))
print(df_clean.head(10))

print("-"*30,"take year,month and day value from date kolumn ","-"*30)
df_clean["Day"] = df_clean["Last Updated"].dt.day
df_clean["Month"] = df_clean["Last Updated"].dt.month
df_clean["Year"] = df_clean["Last Updated"].dt.year

print(df_clean[["Last Updated","Day","Month","Year"]])

print("="*30,"Data Distribution Graphs","="*30)
app_counts = df_clean["App"].value_counts()
duplicated_app_values = app_counts[app_counts>1]
print("Duplicated values in column: 'App'")
print(duplicated_app_values)

print("-"*30,"Cleaning duplicated rows","-"*30)
df_clean = df_clean.drop_duplicates(subset=['App'],keep="first")
print(df_clean.info())

for feature in df_clean.columns:
    print(f"{feature} type : {df_clean[feature].dtype}")

print("-"*30,"taking numeric columns","-"*30)
#Hocanın gösterdiği
'''
numeric_features = [feature for feature in df_clean.columns if df_clean[feature].dtype in ['float64','int32']]
date_features = [feature for feature in df_clean.columns if df_clean[feature].dtype =='datetime64[us]']
categorical_features = [feature for feature in df_clean.columns if df_clean[feature].dtype == 'str']
'''
#ChatGPT  kısa kod
numeric_features = df_clean.select_dtypes(include="number").columns.to_list()
date_features = df_clean.select_dtypes(include="datetime").columns.to_list()
categorical_features = df_clean.select_dtypes(include=["object","string","str"]).columns.tolist()
data_type_column = pd.DataFrame()
data_type_column["Column_Name"] = None
data_type_column["Data_type"] = None
for idx,col in enumerate(df_clean.columns):
    data_type_column.loc[idx,"Column_Name"] = col
    data_type_column.loc[idx,"Data_type"] = df_clean[col].dtype
print(data_type_column)

'''
print("-"*30,"Drawing Graphs - Numeric","-"*30)
plt.figure(figsize=(15,10))
for i in range(0,len(numeric_features)):
    plt.subplot(5,3,i+1)
    sns.kdeplot(x=df_clean[numeric_features[i]],color="b",fill=True)
    plt.xlabel(numeric_features[i])
    plt.tight_layout()
plt.show()
'''

'''
print("-"*30,"Drawing Graphs - Categorical","-"*30)
category = ["Type","Content Rating"]
plt.figure(figsize=(15,4))
for i in range(0,len(category)):
    plt.subplot(1,2,i+1)
    sns.countplot(x=df_clean[category[i]],color="b")
    plt.xlabel(category[i])
    plt.tight_layout()
plt.show()
'''
# top app categories by installment
print("-"*30,"top app categories by installment","-"*30)
print(df_clean.groupby("Category")["Installs"].sum().sort_values(ascending=False))
top_10_category = df_clean.groupby("Category")["Installs"].sum().sort_values(ascending=False).head(10).reset_index()
print(top_10_category)
'''
print("-"*30,"Top 10 Category install pie graph","-"*30)
counts= top_10_category["Installs"]
percentage =top_10_category["Installs"]/sum(top_10_category["Installs"])*100
summary=pd.DataFrame({
    "count":counts,
    "percentage":percentage
})

print(summary)

plt.figure(figsize=(8,8))
plt.pie(
    counts,
    labels=top_10_category["Category"],
    autopct="%1.1f%%",
    startangle=90
)
plt.title("Top 10 Category install")
plt.tight_layout()
plt.show()
'''

#top 5 app in categories
print("-"*30,"Top 10 Category install by app name","-"*30)
apps = ['GAME','COMMUNICATION','TOOLS','PRODUCTIVITY','SOCIAL']
df_app_categoriy = df_clean.groupby(["Category","App"])["Installs"].sum().reset_index().sort_values("Installs",ascending=False)
result = []
for i,app in enumerate(apps):
    df2 = df_app_categoriy[df_app_categoriy['Category']==app]
    df2 = df2.head(5)
    result.append(df2)
result_df = pd.concat(result,ignore_index=True)
print(result_df) 
print(df_clean.duplicated(subset=["Category","App"]).sum())