import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def data_info(data):
    result=pd.DataFrame({
        "Columns": data.columns,
        "Non-Null Count": data.notnull().sum().values,
        "Null Count": data.isnull().sum().values,
        "Null Percent": ((data.isnull().sum().values/len(data))*100).round(2),
        "Dtype": data.dtypes.values})
    print("-"*60)
    print(f"Data_shape: {data.shape}")
    print("-"*60)
    return result

def delete_colmns_from_list(column_list,*columns_to_delete):
    for column in columns_to_delete:
        if column in column_list:
            column_list.remove(column)
    return column_list

def column_unique(df):
    column_list = list(df.columns)
    result_list = []
    for col in column_list:
        unique_list_len = len(df[col].unique())
        duplicated_list_count = df[col].duplicated().sum()
        result_list.append([col,unique_list_len,duplicated_list_count])
    result=pd.DataFrame(
        result_list,
        columns=["Column","Unique_Count","Duplicate_Count"]
    )
    return result


def data_review(data:pd.DataFrame):
    print(data.columns)
    print(data.head())
    print(data_info(data))
    print(data.describe())