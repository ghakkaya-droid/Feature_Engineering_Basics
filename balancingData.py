import pandas as pd
import numpy as np
from sklearn.utils import resample
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
'''
    gelen data bazen dengesiz olabilir. data kaynaklarını dengeleyeceğiz.
'''

#random seed
np.random.seed(42)

set1no = 900
set2no = 100

df1 = pd.DataFrame({
    "feature_1": np.random.normal(loc=0,scale=1,size=set1no), #Loc: Ortalaması 0, Std:1
    "feature_2": np.random.normal(loc=0,scale=1,size=set1no),
    "target": [0]*set1no
})

df2 = pd.DataFrame({
    "feature_1": np.random.normal(loc=0,scale=1,size=set2no), #Loc: Ortalaması 0, Std:1
    "feature_2": np.random.normal(loc=0,scale=1,size=set2no),
    "target": [1]*set2no
})

#df1 ve df2 nin alt alta eklenmesi concat
df = pd.concat([df1,df2]).reset_index(drop=True)

#df inceleme 
# print(df["target"].value_counts())

#upsampling --> eksik olan verileri arttırmak
#downsampling --> fazla olan veri sayısını azaltmak 

#upsampling yapıyoruz:
print("-"*30,"upsampling","-"*30)
df_minority = df[df["target"]==1] # target sadece 1 olanlar
#print(df_minority)
df_majority = df[df["target"]==0] # target sadece 0 olanlar

df_minority_upsampled:pd.DataFrame = resample(df_minority,replace=True,n_samples=len(df_majority),random_state=42)
df_upsample = pd.concat([df_majority,df_minority_upsampled])
print(df_upsample["target"].value_counts())

#downsampling yapıyoruz
print("-"*30,"downsampling","-"*30)
df_majority_downsampled = resample(df_majority,replace=False,n_samples=len(df_minority),random_state=42)
df_downsapmled:pd.DataFrame = pd.concat([df_majority_downsampled,df_minority])
print(df_downsapmled["target"].value_counts())

'''
    replace=True özelliği veriyi tekrarlıyor. Aslinda copy-paste yapıyor
'''

#SMOTE (Synthetic Minority Over-sampling Technique)
print("-"*30,"SMOTE","-"*30)
'''

scatter = plt.scatter(df["feature_1"],df["feature_2"],c=df["target"])
plt.xlabel("feature_1")
plt.ylabel("feature_2")
legant = plt.legend(
    *scatter.legend_elements(),
    title="target"
)
plt.show()
'''
print(df.info())
oversample = SMOTE()
(X,y) = oversample.fit_resample(df[["feature_1","feature_2"]],df["target"])

oversample_df = pd.concat([X,y],axis=1)
#print(oversample_df)
print(oversample_df["target"].value_counts())

fig, axes = plt.subplots(1,2,figsize=(12,5))
# 1. grafik: SMOTE sonrası data
scatter1 = axes[0].scatter(
    oversample_df["feature_1"],
    oversample_df["feature_2"],
    c=oversample_df["target"]
)
axes[0].set_xlabel("feature_1")
axes[0].set_ylabel("feature_2")
axes[0].set_title("SMOTE Sonrası")

axes[0].legend(
    *scatter1.legend_elements(),
    title="target"
)


# 2. grafik: Orijinal data
scatter2 = axes[1].scatter(
    df["feature_1"],
    df["feature_2"],
    c=df["target"]
)

axes[1].set_xlabel("feature_1")
axes[1].set_ylabel("feature_2")
axes[1].set_title("Orijinal Data")

axes[1].legend(
    *scatter2.legend_elements(),
    title="target"
)

plt.tight_layout()
plt.show()

'''
    SMOTE azınlık sınıfını çoğunluk sınıfına eşitler. 
    df de 100 adet azınlık (minority), 900 adet çoğunluk (majority) vardı
    SMOTE 100 adedi, 900 adede çıkarttı yani 800 adet daha ekleme yaptı. 
    Toplam data satısı 900 + 900 = 1800 oldu. 
'''