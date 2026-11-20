import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from tabulate import tabulate
import time

'''
Thuc hien phep bien doi xuoi Fourier tren ma tran I, sau do xuat ra ket qua
-Thuc hien phep bien doi nguoc de thu duoc ma tran I nhu ban dau, xuat ra ket qua
'''
I = np.array([
    [1, 4, 2, 6, 5],
    [2, 7, 9, 2, 1],
    [3, 6, 5, 0, 1],
    [3, 6, 7, 0, 8],
    [8, 6, 4, 9, 1],
])

# Bien doi Fourier xuoi
I_new = np.fft.fft2(I)
I_new_mod = [[f"{r.real:.2f}{r.imag:+.2f}j"for r in row]for row in I_new]  #1 ben chua so thuc, 1 ben chua so ao
print("Ma tran sau bien doi Fourier xuoi:\n")
print(tabulate(I_new_mod, tablefmt="grid", stralign="right")) #lam dep

# Bien doi Fourier nguoc
I = np.fft.ifft2(I_new).real
print("Ma tran sau bien doi Fourier nguoc",I.round(1))


# '''
# Chuan hoa du lieu
# '''

'''
Doc file duoc gui bang numpy va thuc hien cac buoc sau:
- Chuyen cac cot thuoc tinh so ve float
- Chuyen ca cot chua thuoc tinh ky tu ve dang so
- Xuat ra dlieu moi sau khi chuyen doi
'''

#Doc du lieu
data = np.genfromtxt("Example.xls", delimiter=",", dtype="object", skip_header=1)
print("Du lieu ban dau\n")
print(data)

#Khai bao mang index chua cot cac thuoc tinh dang so
ind_num = [0, 1, 2, 3, 4, 5, 6]

#Khai bao mang index chua cot cac thuoc tinh dang ky tu
ind_sym = [2, 7]

#Chuyen doi cac thuoc tinh ky tu ve dang so
for col in ind_sym:
    data[:, col] = LabelEncoder().fit_transform(data[:, col])

print("Du lieu sao khi chuyen cac thuoc tinh ve dang so \n")
print(data)


#Chuyen cac thuoc tinh so ve dang float
data[:, ind_num] = data[:, ind_num].astype(float)
data[:, ind_sym] = data[:, ind_sym].astype(int)
print("Du lieu sau khi chuyen cac thuoc tinh ve dang float: ")
print(data)

'''
-Thuc hien chuan hoa:
+ Cot index 3 theo min-max
+ Cot 0, 1, 4 theo z-score
+ Cot 6 trong doan [0.5, 0.9]
+ Xuat toan bo cac ket qua chuan hoa
'''


# Chuan hoa cot index 3 theo Min-Max
data[:, [3]] = MinMaxScaler().fit_transform(data[:, [3]])
print("Du lieu sau khi chuan hoa Min-Max cot 3:")
print(tabulate(data, floatfmt= ".4f", tablefmt="grid"))

#Chuan hoa cot 0, 1, 4 theo z-score
data[:, [0, 1, 4]] = StandardScaler().fit_transform(data[:, [0, 1, 4]])
print("Du lieu sau khi chuan hoa Z-Score:")
print(tabulate(data, floatfmt= ".4f", tablefmt="grid"))

#Chuan hoa cot 6 trong doan [0.5, 0.9]
data[:, [6]] = MinMaxScaler(feature_range=(0.5, 0.9)).fit_transform(data[:, [6]])
print("Du lieu sau khi chuan hoa cot 6 trong doan [0.5, 0.9]")
print(tabulate(data, floatfmt= ".4f", tablefmt="grid"))


'''
Thuc hien cac thao tac sau:
- Tao mot cot = cot du lieu dau tien - cot du lieu thu 2 va thay vao vi tri dau tien
- Tao mot cot nam giua cot 4 va 5 = (cot 1- cot 2) *cot 3
'''
#1
#reshape(-1, 1)  : 4 hang 1 cot va nguoc lai
new_col = data[:, 0] - data[:, 1].reshape(-1, 1) #no dang nam ngang nen phai reshape cho no thanh cot
data_upd = np.hstack((new_col, data))
print("\nDu lieu #1\n")
print(tabulate(data_upd, floatfmt= ".4f", tablefmt="grid"))

#2
new_col = (data[:, 1] - data[:, 2])* data[:, 3]
data_upd = np.insert(data, 5, new_col, axis = 1) #(dl cu, vtri muon bsung, dlieu, theo chieu nao)
print("\nDu lieu #2\n")
print(tabulate(data_upd, floatfmt= ".4f", tablefmt="grid"))

#3 Tach du lieu thanh cac lop (Group)
labels = np.unique(data[:, -1])
#duyet qua tung lop de nhom
for label in labels:
    group = data[data[:, -1]] == label
    print(f"Lop quyet dinh: {int(label)} - So luong mau: {len(group)}")
print(tabulate(group, floatfmt= ".4f", tablefmt="fancy_grid"))

