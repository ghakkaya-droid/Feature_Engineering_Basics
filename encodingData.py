import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

df = sns.load_dataset('titanic')

#çalışılacak datalardaki boş satırlar
print("-"*30,"boş data kontrolü","-"*30)
result_empty_data_set= df[["sex","class","embark_town"]].isna().sum()
print(result_empty_data_set)
print(f"Toplam boş data: {sum(result_empty_data_set)}")

#belli bir kolondaki boş satırları düşürmek
print("-"*30,"boş kolonlardan kurtulduk","-"*30)
df.dropna(subset=['embark_town'],inplace=True)
result_empty_data_set= df[["sex","class","embark_town"]].isna().sum()
if sum(result_empty_data_set==0):
    print("Bos Data yok")
else:
    print("Beceremedin aq()")

#one-hot encoding
print("="*30,"one-hot encoding","="*30)

print("-"*30,"data sayıları","-"*30)
def data_count(data:pd.DataFrame,list_columns):
    df_result = pd.DataFrame()
    for col in list_columns:
        temp = data[col].value_counts().reset_index()
        temp.columns = ["value","count"]
        temp["column"] = col

        df_result = pd.concat([df_result,temp],axis=0)
    
    df_result["Total_Count"] = df_result.groupby("column")["count"].transform("sum")

    return df_result[["column","value","count","Total_Count"]]

print(data_count(df,["sex","class","embark_town"]))

print("-"*30,"pd.get_dummies(DF,columns,drop_first=True)","-"*30)
df_onehot = pd.get_dummies(df,columns=["sex","embark_town"],drop_first=True,dtype=int)
#print(df)
print(df_onehot)

#label encoding
print("="*30,"label encoding","="*30)
label_encoder = LabelEncoder()
df_label = df.copy()
df_label["sex"] = label_encoder.fit_transform(df_label["sex"])
print(df_label)
print(label_encoder.classes_)

#ordinal encoding
print("="*30,"ordinal encoding","="*30)
df_ordinal=df.copy()
class_order = ["Third","Second","First"]
ordinal_encoder = OrdinalEncoder(categories=[class_order])
df_ordinal["class"] = ordinal_encoder.fit_transform(df_ordinal[["class"]])
print(df_ordinal)

#grafiğe çizdirme
fig,axes = plt.subplots(1,3,figsize=(15,5))
df["sex"].value_counts().plot(kind="bar",ax=axes[0],title="Original Categories")
df_label["sex"].value_counts().plot(kind="bar",ax=axes[1],title="Label Encoded")
df_onehot["sex_male"].value_counts().plot(kind="bar",ax=axes[2],title="One-hot Encoded")
plt.show()