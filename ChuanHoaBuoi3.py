'''
Trong pandas, truy cap vao cot va xu ly gom cac phuong thuc chinh sau:
1.  df["tên cột"].dtype          --> Xem kiểu dữ liệu trên cột
2.  df["tên cột"].size           --> Xem số lượng phần tử của cột (tính cả NaN)
3.  df["tên cột"].count()        --> Đếm số lượng giá trị khác NaN của cột
4.  df["tên cột"].unique()          --> Trả về các giá trị duy nhất trong cột
5.  df["tên cột"].between(left, right)  --> Lọc các giá trị trong khoảng
6.  df["tên cột"].nlargest(n)     --> Lấy n giá trị lớn nhất của cột
7.  df["tên cột"].nsmallest(n)    --> Lấy n giá trị nhỏ nhất của cột
8.  df["tên cột"].dropna()        --> Loại bỏ các giá trị NaN trong cột
9.  df["tên cột"].drop_duplicates()  --> Loại bỏ các giá trị trùng lặp trong cột
10. df["tên cột"].fillna(value)   --> Điền các giá trị thiếu
11. df["tên cột"].apply(func)     --> Áp dụng một hàm lên phần tử cột
12. df["tên cột"].value_counts()   --> Đếm tần suất xuất hiện của từng giá trị
13. df["tên cột"].mode()          --> Trả về các giá trị xuất hiện nhiều nhất
'''
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler

from prompt_toolkit.renderer import print_formatted_text

#Doc du lieu duoc cung cap bang thu vien Pandas
df = pd.read_csv("Dataset.csv", delimiter=",", skipinitialspace=True) #skipinitialspace: loai bo khoang trang
print("Du lieu ban dau: ")
print(df)

#Tim nhung cot co du lieu bi thieu
#Xac dinh toan bo ten cac cot: df.columns
#Quet toan bo bang, cho nao thieu la True, k thieu la False -> df.isnull
#any()
miss_index = df.columns[df.isnull().any()].tolist()
print("Cac cot thuoc tinh bi thieu: ")
print(miss_index)

#Dien cac gia tri khuyet thieu tuong ung voi gia tri trung binh moi cot
df[miss_index] = df[miss_index].fillna((df[miss_index]).mean())
print("Du lieu sau khi bo sung cac gia tri bi thieu")
print(df.round(4))

#Bo sung them cot k la hieu cua cot b va cot a, sau do chen vao sau cot i
#df.insert(vi tri cot, ten cot can them, tap gia tri cot)
val_k = df['b'] - df['a']  #Tap gia tri cot
#Cach 1: Python thuan
index_i = df.columns.tolist().index('i')
print("Cach 1: ", index_i)

#Cach 2: Numpy
index_i = np.where(df.columns == "i")[0][0]
print("Cach 2: ", index_i)

#Cach 3: Pandas
index_i =df.columns.get_loc("i")
print("Cach 3: ", index_i)
#-> 3 cach tuong tu nhau

#Thuc hien bai toan
df.insert(index_i + 1, "k", val_k)
print("Bang sau khi thay doi cac gia tri \n", df)

#Giam toan bo cac gia tri trong cot k thuoc lop L3 di 1 don vi
#df.loc[dieu kien, ten cot can thay doi]
df.loc[df['D'] == 'L3', 'k'] = df.loc[df['D'] == 'L3', 'k'] - 1
print(df)

#Doi vi tri co e cho cot d
cols = df.columns.tolist()
index_d = df.columns.get_loc("d")
index_e = df.columns.get_loc("e")

cols[index_d], cols[index_e] = cols[index_e], cols[index_d]
df = df[cols]
print("Du lieu sau khi doi cho cot e va d: ")
print(df)

#Tim cac cot chua ky tu va chuyen ve dang so
sym_col = df.select_dtypes(include = ["object", "category", "string"]).columns.tolist()
print("Cac cot chua ky tu")
print(sym_col)

#Chuyen cac cot chua ky tu ve dang so
df[sym_col] = df[sym_col].apply(lambda x: LabelEncoder().fit_transform(x))
df[sym_col] = df[sym_col].astype("int")
print("Du lieu sau khi chuyen doi cac gia tri cot phan loai sang dang so: \n", df)


#Chuan hoa min-max cac cot chua cac gia tri so > 0
#Buoc 1: Tim cac cot chua gia tri so
num_cols = df.select_dtypes(include = ["number"]).columns

#Buoc 2: Lay cac cot gia tri so > 0
pos_cols = df.columns[(df[num_cols] > 0).all()].tolist()
print("Mang chua cac cot dang so co gia tri duong: \n")
print(pos_cols)

#Buoc 3: Tien hanh chuan hoa min-max
df[pos_cols] = MinMaxScaler().fit_transform(df[pos_cols])
print("Du lieu sau khi chuan hoa min-max: \n")
print(df)


# #Chuan hoa z-core cac cot co gia tri am
# #Cach 1: (lor)
# z_cols = df.columns[(df[num_cols] < 0).any()].tolist()
# df[z_cols] = StandardScaler().fit_transform([z_cols])
# print("Cach 1 (z-core): \n", df)

#Cach 2:
neg_cols = df.columns.difference(sym_col + pos_cols).tolist()
print("Mang chua cac cot dang so co gia tri duong: \n")
print(pos_cols)

df[neg_cols] = StandardScaler().fit_transform(df[neg_cols])
print("Cach 2 (z-core): \n", df)

