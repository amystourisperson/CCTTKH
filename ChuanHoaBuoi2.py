import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from tabulate import tabulate

#Doc du lieu (Doc bang pandas)
df = pd.read_csv("Example.xls", delimiter=",", skipinitialspace=True) #skipinitialspace: loai bo khoang trang
print("Du lieu ban dau: ")
print(df)

#Doi voi pandas cho phep truy cap du lieu cua cot theo 2 cach:
#Truy cap theo index va ten cot
index_num = [0, 1, 3, 4, 6]
index_sym = [2, 7]

name_num = ["a", "b", "c", "d", "e", "f", "g"]
name_sym = ["c", "D"]

#LableEncoder cac cot mang gia tri phan loai
df[name_sym] = df[name_sym].apply(lambda col: LabelEncoder().fit_transform(col))
print("Du lieu sau khi chuyen doi cac gia tri cot phan loai sang so: ")
print(df)

#Chuyen doi cac kieu du lieu cho df
df[name_sym] = df[name_sym].astype("int")
df[name_num] = df[name_num].astype("float")
print("Du lieu sau khi chuyen doi sang kieu dl phu hop")
print(df)

#Chuan hoa cot 3 theo min-max
df.iloc[:, [3]] = MinMaxScaler().fit_transform(df.iloc[:, [3]]) #Phai cho vao []
print("Du lieu sau khi chuan hoa min-max cot 3")
print(df.round(4))

#Chuan hoa cot 0, 1, 4 theo z-core (theo pandas)
df.iloc[:, [0, 1, 4]] = StandardScaler().fit_transform(df.iloc[:, [0, 1, 4]])
print("Du lieu sau khi chuan hoa Z-Score:")
print(df.round(4))

#Chuan hoa cot 6 trong doan [0.5, 0.9]
df.iloc[:, [6]] = MinMaxScaler(feature_range=(0.5, 0.9)).fit_transform(df.iloc[:, [6]])
print(df.round(4))

#Trong cot f, dua cac gia tri > 12 = 10
#df.loc[Dkien loc tung dong, cot can loc]
# df.loc[df['f'] > 12, 'f'] = 10
# print("Dữ liệu sau khi xử lý cột f:")
# print(df)

#Neu > 15 thi doi thanh 7 va nho hon 15 doi thanh 9
df['f'] = df['f'].apply(lambda x: 15 if x== 15 else 7 if x < 15 else 9) #Duyet tung ptu x trong cot f
print("Dữ liệu sau khi xử lý cột f:")
print(df)

#Cach 2
df["f"] = df["f"].mask(df["f"] > 15, 7).mask(df["f"] < 15, 9)
print(df)

'''

'''

df = pd.read_csv("Example.xls", delimiter=",", skipinitialspace=True)

num_col = df.select_dtypes(include= ["number"]).columns
print("Cac cot dlieu so: ", num_col)

sym_col = df.select_dtypes(include= ["object", "category", "string"]).columns
print("Cac cot dlieu phan loai: ", sym_col)

# df[sym_col] = df[sym_col].apply(lambda x: LabelEncoder().fit_transform(x))
# print("Du lieu sau khi chuyen doi cac gia tri cot phan loai sang dang so: ", df)


print("-------------------------------------")
#Phan tach du lieu thanh cac nhom theo so lop
D_valuse = df["D"].unique()
print(D_valuse)

sub_df = [df.groupby("D").get_group(d) for d in D_valuse]
print(tabulate(sub_df, floatfmt=".2f", tablefmt="grid"))
