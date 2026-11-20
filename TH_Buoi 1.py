from cmath import sqrt

import numpy as np
from tabulate import tabulate

# from jaraco.functools import print_yielded
#
# '''
# Cho mot mang x = [1, 1, 3, 3, 4]
# In ra man hinh cac gia tri duy nhat cua x va so luong cua chung
# '''
# x = np.array([1, 1, 3, 3, 4])
#
# #In ra ptu co gtri duy nhat
# unique = np.unique(x)
# print("Phan tu co gia tri duy nhat", unique)
# print(x)
# print(len(x))
# print("So luong phan tu duy nhat cua mang", unique.shape[0]) #.shape[0] so ptu  neu mang 1 chieu
#
# #Tao mot vector 0 co 10 ptu, tuy nhien ptu thu 5 bang 1
# vt_zeros = np.zeros(10)
# vt_zeros[4] = 1
# print(vt_zeros)
#
# #Tim gia tri co tan so xuat hien nhieu nhat trong 1 mang dc sinh ra nn
# z = np.random.randint(0, 5, 10)
# print(z)
# dem_sl = np.bincount(z)  #Dem so tan suat xh
# print("So lan xuat hien", dem_sl) #Dem so lan xhien
# index_max = np.where(dem_sl == np.max(dem_sl))[0]
# print("Cac so xuat hien nhieu nhat", index_max)
#
# #Tao 1 mang ngau nhien 10x10, sau do:
# '''
# Tim gia tri lon nhat, nho nhat, tbinh, do lech chuan cua mang
# '''
# a = np.random.randint(0, 5, (10, 10))
# print(a)
#
#
# max_arr = np.max(a)
# print("Gia tri lon nhat cua mang a: ", max_arr)
#
# min_arr = np.min(a)
# print("Gia tri lon nhat cua mang a: ", min_arr)
#
# lech = np.std(a)
# print("Do lech chuan", lech)
#
# '''
# Viet mot ham tinh khoang cach giua 2 vt bang numpy
# '''
# x1 = np.array([1, 2, 3, 4])
# x2 = np.array([7, 8, 9, 10])
# def eucli(x1, x2):
#     z = len(x1)
#     tong = 0
#     for i in range(len(x1)):
#         tong += pow(x1[i] - x2[i], 2)
#     return sqrt(tong)
# print("Cach 1: ", eucli(x1, x2))
#
# def ecl(x1, x2):
#     sum1 = np.sum(np.square(x2-x1))
#     result = sum1**0.5
#     return result
# print("Cach 2: ", ecl(x1, x2))
#
# '''
# Cho 1 vector x = [1, 2, 3, 4, 5, 6, 7, 8,100]
# xuat ra 2 mang vector rieng biet, x1 chua cac so le va x2 chua cac so chan
# '''
# a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 100])
#
# #cach 1
# x1 = a[a %2 == 0]
# x2 = a[a % 2 == 1]
# print("1 vector chan (c1): ", x1)
# print("1 vector le(c1): ", x2)
#
# #cach 2
# vt1 =np.where(a % 2 == 0)
# vt2 = np.where(a % 2 == 1)
# print("1 vector chan(c2): ", a[vt1])
# print("1 vector luong(c2): ", a[vt2])
#
#
# a1 = np.array([1, 2, 3, 2, 2, 7, 9, 4, 6])
# b1 = np.array([2, 5, 5, 9, 10, 2, 4, 1, 3])
# '''
# Xac dinh vt c chua cac ptu cua ca a va b
# '''
# #Cach 1 (chua on)
# c = set()
# a1_1 = set(a1)
# b1_1 = set(b1)
# c_1 = a1_1 | b1_1
# print("Loc (cach 1): ", np.array(c_1))
#
# #Cach 2
# a_unique = np.unique(a1) #lay cac gia tri duy nhat cua mang a1
# b_unique = np.unique(b1)  #lay cac gia tri duy nhat cua mang a2
# c_2 = [i for i in a_unique if i in b_unique]
# print("Loc (cach 2): ", np.array(c_2))
#
# #Cach 3
# c_3 = np.intersect1d(a_unique, b_unique)
# print("Loc (cach 3): ", c_3)  #in ra nhung gia tri chung cua 2 mang
#
# #In ra mang nam trong mang 2 nhung k nam trong mang 1
# only_b = np.setdiff1d(b_unique, c_3)
# print("Only : ", only_b)


# x1 = np.array([1, 2, 3,1, 3, 4, 3, 7, 5, 6])
# x2 = np.array([7, 2, 10,1, 7, 4, 9, 7, 9, 8])
# #Tim cac vi tri sao cho 2 mang co gia tri le bang nhau
# #CACH 1:
# c = np.where((x1 == x2) & (x1 % 2 == 1))[0]
# print(c)
#
# #CACH 2:
# index1 = np.where(x1 == x2)
# index2 = np.where(x1 % 2 == 1)
# index = np.intersect1d(index1, index2)
# print(index)


# Ma tran (Matrix)
A = np.array([
    [0.2, 1.0, 0.4, 0.8],
    [0.9, 0.6, 0.5, 0.7],
    [0.5, 1.4, 0.3, 0.2],
    [0.2, 0.1, 0.0, 0.9]])

# Sx cac dong cua mt A theo thu tu tang dan cua tong
B = A[np.argsort(np.sum(A, axis=1)), :]  #A[sx hang(Tong hang), giu nguyen cot]
print("Xong chua: \n", B)

# Sx cac cot cua mt A theo thu thu giam dan cua tong
C = A[:, np.argsort(np.sum(A, axis=0)[::-1])]
print("Xong chua: \n", C)

# Tim vector rieng va gia tri rieng cua ma tran. Trong do:
# Cac gia tri rieng tang dan
# In ra cac vector rieng tuong ung theo cac gtri rieng dc sx

# Tim gia tri rieng va vector rieng
eigenvalues, eigenvectors = np.linalg.eig(A)  # Moi cot la 1 vector

# Sap xep gia tri rieng tang dan
idx = np.argsort(eigenvalues)
eigenvalues_sorted = eigenvalues[idx]

print("\nGia tri rieng tang dan:")
print(eigenvalues_sorted)

print("\nCac vector rieng tuong ung:")
eigenvectors_sorted = eigenvectors[:, idx]

table = []
for i in range(len(eigenvalues_sorted)):
    value = f"{eigenvalues_sorted[i]}"
    vector = f"{eigenvalues_sorted[:, i]}"
    table.append([value, vector])

print("Các vector riêng tương ứng với các giá trị riêng:", table)
print(tabulate(table, headers=["Giá trị riêng", "Vector riêng"], tablefmt="grid"))
