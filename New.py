
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.naive_bayes import CategoricalNB, GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
from mixed_naive_bayes import MixedNB
import pandas as pd

# Doc bo du lieu Bayes
df = pd.read_csv("Bayes.csv", skipinitialspace=True)
print("Du lieu sau khi doc")
print(df)

# Luu lai du lieu ban dau de chia Train va Test
df_goc = df.copy()

# Tim cac cot du lieu dang ky tu/phan loai
sym_cols = df.select_dtypes(include=['object', 'string', 'category']).columns.tolist()
print(sym_cols)

# Tim cac cot du lieu dang so
num_cols = df.select_dtypes(include=['number']).columns.tolist()
print(num_cols)

# Chuyen cac cot du lieu ky tu ve dang so
df[sym_cols] = df[sym_cols].apply(lambda col: LabelEncoder().fit_transform(col))
df[sym_cols] = df[sym_cols].astype('int')

print("Du lieu sau khi chuyen doi cac cot ky tu sang so:")
print(df)

# Chuan hoa min-max cac cot du lieu dang so
df[num_cols] = MinMaxScaler().fit_transform(df[num_cols])

print("Du lieu sau khi chuan hoa min-max cac cot du lieu dang so:")
print(df)

# Sap xep lai cac ban ghi theo thu tu tang dan cua thuoc tinh index 3
name_col = num_cols[3]
print(name_col)

df = df.sort_values(by=name_col, ascending=True)
print("Du lieu sau khi sap xep tang dan:")
print(df)

# Thay the cac gia tri > 0.8 thuoc lop Yes thanh 0.9
name_D = sym_cols[-1]

# Cach 1
index = np.where(df[name_D] == 1)[0]
print("Cac vi tri thuoc lop Yes", index)

df.iloc[index, :-1] = np.where(
    df.iloc[index, :-1] > 0.8,
    0.9,
    df.iloc[index, :-1]
)

print("Du lieu sau khi thay the (Cach 1):")
print(df)

# Cach 2
df.loc[df[name_D] == 1, num_cols] = df.loc[
    df[name_D] == 1, num_cols
].mask(df.loc[df[name_D] == 1, num_cols] > 0.8, 0.9)

print("Du lieu sau khi thay the (Cach 2):")
print(df)

# 2a: Lay cac dong 0, 7 va 9 dua vao tap Test
# Luu y: Chia theo cac dong ban dau, khong chia theo thu tu da sap xep

df_goc = df_goc.reset_index(drop=True)

# Chuyen doi du lieu goc theo cung cach xu ly
df_goc[sym_cols] = df_goc[sym_cols].apply(
    lambda col: LabelEncoder().fit_transform(col)
)
df_goc[sym_cols] = df_goc[sym_cols].astype('int')

df_goc[num_cols] = MinMaxScaler().fit_transform(df_goc[num_cols])

# Thay the gia tri > 0.8 cua lop Yes
df_goc.loc[df_goc[name_D] == 1, num_cols] = df_goc.loc[
    df_goc[name_D] == 1, num_cols
].mask(
    df_goc.loc[df_goc[name_D] == 1, num_cols] > 0.8,
    0.9
)

# Chia tap Test va Train
test_index = [0, 7, 9]

Test = df_goc.loc[test_index]
Train = df_goc.drop(test_index)

# Tach thanh X_train, Y_train, X_test, Y_test
X_train = Train[df_goc.columns[:-1]].values
Y_train = Train[df_goc.columns[-1]].values

X_test = Test[df_goc.columns[:-1]].values
Y_test = Test[df_goc.columns[-1]].values

print("X_train:")
print(X_train)

print("Y_train:")
print(Y_train)

print("X_test:")
print(X_test)

print("Y_test:")
print(Y_test)

# Khoi tao va huan luyen mo hinh NB
clf = GaussianNB()
clf.fit(X_train, Y_train)

# Du doan
y_pred = clf.predict(X_test)
print("Ket qua du doan:", y_pred)

# Tinh do chinh xac
correct = np.sum(y_pred == Y_test)
total = len(Y_test)

accuracy = correct / total
print(f"Cach 1: Do chinh xac cua mo hinh: {accuracy * 100:.2f}%")

accuracy = accuracy_score(Y_test, y_pred)
print(f"Cach 2: Do chinh xac cua mo hinh: {accuracy * 100:.2f}%")

# Chia tap du lieu thanh 70% Train va 30% Test
df_x = df_goc[df_goc.columns[:-1]].values
df_y = df_goc[df_goc.columns[-1]].values

X_train, X_test, Y_train, Y_test = train_test_split(
    df_x, df_y, random_state=0, test_size=0.3, shuffle=False
)

# Tim vi tri cac thuoc tinh phan loai
sym_index = [df_goc.columns.get_loc(col) for col in sym_cols[:-1]]

# Khoi tao va huan luyen MixedNB
clf = MixedNB(categorical_features=sym_index)
clf.fit(X_train, Y_train)

# Du doan
y_pred = clf.predict(X_test)

accuracy = accuracy_score(Y_test, y_pred)
print(f"Do chinh xac cua mo hinh MixedNB: {accuracy * 100:.2f}%")
