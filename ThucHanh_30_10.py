from IPython.core.display_functions import display
from sklearn.naive_bayes import GaussianNB, CategoricalNB
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
import numpy as np
import pandas as pd

from New_29_10 import prior, x_test

print(pd.__version__)
df = pd.read_csv("DatasetB5.csv", delimiter=',', skipinitialspace=True)
print(display(df))

#Tim cac cot du lieu dang ky tu/phan loai
sym_cols = df.select_dtypes(include=['object', 'string']).columns.tolist()
print(sym_cols)

#Tim cac cot du lieu dang so
num_cols = df.select_dtypes(include=['number']).columns.tolist()
print(num_cols)

#Chuyen cac cot du lieu ky tu ve dang so
df[sym_cols] = df[sym_cols].apply(lambda col: LabelEncoder().fit_transform(col))
df[sym_cols] = df[sym_cols].astype('int')
print("Du lieu sau khi chuyen doi cac cot ky tu sang so: ")
print(df)

#Chuan hoa min_max cac cot du lieu dang so
df[num_cols] = MinMaxScaler().fit_transform(df[num_cols])
print("Du lieu sau khi chuan hoa min-max cac cot du lieu dang so: ")
print(df)


##Su dung phuong phap roi rac hoa du lieu
'''
Roi rac hoa cac cot du lieu dang so theo dieu kien sau:
-Neu gau tri <= 0.3 thi bang 0
-Neu gia tri > 0.3 va < 0.7 thi bang 1
-Neu gia tri >= 0.7 thi bang 2
'''

'''
#Cach 1
def bins(values):
    if values <= 0.3:
        return 0
    elif values > 0.3 and values < 0.7:
        return 1
    else:
        return 2
for col in num_cols:
    df[col] = df[col].apply(bins)
print("Output: ")
display(df)
'''
def bins(values):
    if values <= 0.3:
        return 0
    elif values > 0.3 and values < 0.7:
        return 1
    else:
        return 2
for col in num_cols:
    df[col] = df[col].apply(bins)
print("Output: ")
display(df)

# for col in num_cols:
#     x = df[col]
#     #Dinh nghia dieu kien
#     conditions = [
#         (x <= 0.3), (x > 0.3 and x < 0.7),
#         (x >= 7)
#     ]
#     #Tap nhan thay the
#     labels_change = [0, 1, 2]
#     #Gan gia tri
#     df[col] = np.select(conditions, labels_change, default=0)
#     #Gia tri khong thoa condition thi gan bang default
#
# print('Dữ liệu sau khi biến đổi toàn bộ các cột số:')
# display(df)


#Tach du lieu de huan luyen
# Tách X_train là tập huấn luyện giữa các đặc trưng
X_train = df[df.columns[:-1]].values
print("Tập huấn luyện:\n", X_train)

# Tách Y_train là nhan tương ứng
Y_train = df[df.columns[-1]].values
print("Nhãn của bộ dữ liệu:\n", Y_train)

# Huấn luyện mô hình Naive-Bayes
# Khởi tạo mô hình
clf = CategoricalNB(alpha = 1.0)

# Huấn luyện mô hình
clf.fit(X_train, Y_train)

#Mau sau day thuoc vao lop nao:
X_test = [[0, 2, 1, 0, 0, 1, 1, 2, 1]]
# X_test thuộc lớp nào
predict_x = clf.predict(X_test)
print("X_test thuộc lớp:", predict_x[0])

# Chia dữ liệu thành 2 phần
# Tập dữ liệu liên tục
X_cont = df[num_cols].values
print("Tập dữ liệu huấn luyện trên các cột số, liên tục:")
print(X_cont.round(4))

# Tập dữ liệu chia cho các thuộc tính rời rạc
X_disc = df[sym_cols[:-1]].values
print("Tập dữ liệu huấn luyện trên các cột phân loại, ký tự:")
print(X_disc)

# Nhấn các dữ liệu
Y_train = df[sym_cols[-1]].values
print("Nhãn dữ liệu:\n", Y_train)

#Huan luyen song song 2 mo hinh
#Lien tuc
clf_gaussian = GaussianNB()
clf_gaussian.fit(X_cont, Y_train)

#Roi rac
clf_category = CategoricalNB(alpha = 1)
clf_category.fit(X_disc, Y_train)

def mixed_NB(x_cont, x_disc):
    #Lay xac suat cua tung mo hinh P(y/x)
    #Lay xac suat cua tien luc
    p_cont = clf_gaussian.predict_proba(x_cont.reshape(1, -1))[0]
    #Lay xac suat cua roi rac
    p_disc = clf_category.predict_proba(x_disc.reshape(1, -1))[0]
    #Xac suat tien nghiem
    prior = clf_category.class_prior_
    p = (p_cont * p_disc) / 2

    normal_p = p / np.sum(p)
    return normal_p, clf_gaussian.classes_

X_test = np.array([0, 2, 1, 0, 0, 1, 1, 2, 1])

num_index = [df.columns.get_loc(col) for col in num_cols]
x_const_test = x_test[num_index]

sym_index = [df.columns.get_loc(col) for col in num_cols[:-1]]
x_disc_test = x_test[sym_index]

probs, classes = mixed_NB(x_const_test, x_disc_test)
predict = classes[np.argmax(probs)]

for c, p in zip(classes, probs):
    print(f"Xac suat lon {c}: {p:.4f}")
print(f"Lop cua x: D = {predict}")


