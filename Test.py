from IPython.core.display_functions import display
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
import numpy as np
import pandas as pd

from BTTH2 import labels

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
#         (x <= 0.3), (x > 0.3 & x < 0.7),
#         (x >= 7)
#     ]
#     #Tap nhan thay the
#     labels_change = [0, 1, 2]
#
#     #Gan gia tri
#     df[col] = np.select(conditions, labels_change, default=0)
#     #Gia tri khong thoa condition thi gan bang default
#
# print('Dữ liệu sau khi biến đổi toàn bộ các cột số:')
# display(df)





