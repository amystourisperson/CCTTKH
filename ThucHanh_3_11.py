
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

# Chuan hoa min_max cac cot du lieu dang so
df[num_cols] = MinMaxScaler().fit_transform(df[num_cols])
print("Du lieu sau khi chuan hoa min-max cac cot du lieu dang so:")
print(df)

# Sap xep lai cac ban ghi theo thu tu tang dan cua thuoc tinh index 3 trong mang so tim duoc

# B1: Tim ten cua thuoc tinh co index 3 trong du lieu
name_col = num_cols[3]
print(name_col)

# B2: Sap xep tang dan
df = df.sort_values(by=name_col, ascending=True)
print("Du lieu sau khi sap xep tang dan:")
print(df)


'''Thay the toan bo cac gia tri > 0.8 thuoc lop Yes thanh 0.9'''
#Cach 1:
name_D = sym_cols[-1]
#Tim vi tri lop Yes
index = np.where(df[name_D] == 1)[0]
print("Cac vi tri thuoc lop Yes", index)
#Thay the cac gia tri tai cac index vua tim duoc thoa man dieu kien
df.iloc[index, :-1] = np.where(df.iloc[index, :-1] > 0.8, 0.9, df.iloc[index, :-1])
print("Du lieu sau khi thay the (Cach 1): ")
print(df)
#Cach 2:
df.loc[df[name_D] == 1, num_cols] = df[num_cols].mask(df[num_cols] > 0.8, 0.9)
print("Du lieu sau khi thay the (Cach 2): ")
print(df)


# 2a Lay cac ban ghi dong 0, 7 va 9 dua vao tap Test con lai dua vao tap Train
# 2b Tu hai tap da co tach thanh cac tep X_train, Y_train, X_test, Y_test

df = df.reset_index(drop=True)

test_index = [0, 7, 9]

# Loc theo dong dua vao tap test
Test = df.loc[test_index]

# Lay tap train
Train = df.drop(test_index)

# Tach thanh cac tap thuoc tinh va nhan
X_train = Train[df.columns[:-1]].values
Y_train = Train[df.columns[-1]].values

print("X_train: ")
print(X_train)

print("Y_train: ")
print(Y_train)

X_test = Test[df.columns[:-1]].values
Y_test = Test[df.columns[-1]].values

print("X_test: ")
print(X_test)

print("Y_test: ")
print(Y_test)

# Khoi tao va huan luyen mo hinh NB
clf = GaussianNB()
clf.fit(X_train, Y_train)

# Du doan mo hinh NB
y_pred = clf.predict(X_test)
print(y_pred)


#y_pred --> y_test
correct =  np.sum(y_pred == Y_test)
total = len(Y_test)

accuracy = correct / total
print(f'Cach 1: Do chinh xac cua mo hinh: {accuracy * 100:.2f}%')

accuracy = accuracy_score(y_pred, Y_test)
print(f'Cach 2: Do chinh xac cua mo hinh: {accuracy * 100:.2f}%')
#DL cua thay la 33% (!!!!)


# pip install mixed_naive_bayes
# from mexed_naive_bayes import MixedNB

#Chia tap du lieu thanh 2 phan, 70% dung tap train va 30% dung cho tap test
df_x = df[df.columns[:-1]].values
df_y = df[df.columns[-1]].values
X_train, X_test, Y_train, Y_test = train_test_split(
    df_x, df_y, random_state=0, test_size=0.3, shuffle=False
)

sym_index = [df.columns.get_loc(col) for col in sym_cols[:-1]]

#Khoi tao va huan luyen
clf = MixedNB(categorical_features=sym_index)
clf.fit(X_train, Y_train)

#Du doan
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_pred, Y_test)
print(f"Do chinh xac cua mo hinh: {accuracy * 100:.2f}%")



