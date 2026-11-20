import numpy as np
a=np.random.array[3]
print(a)

A = np.array([
    [ 1, 0, 2],
    [-1,-2, 1],
    [ 1, 4, 3]
])

'''
i + j
[0 1 2]
[1 2 3]
[2 3 4]
'''

#Tong, Hieu cac phan tu tren duong cheo chinh
#Tong cac phan tu duoi duong cheo phu
#Hang cua ma tran

tong = np.trace(A)  #tong cac phan tu tren duong cheo chinh (van tinh dc matrix k vuong)
tong_all = np.sum(A)   #tong tat ca cac phan tu cua ma tran A
diag_matrix = np.diag(A) #lay ra cac phan tu tren duong cheo chinh
hieu = A[0, 0] - A[1, 1] - A[2, 2]

print("Tong duong cheo chinh:", tong)
print("Hieu duong cheo chinh:", hieu)

tong_duoi_cheo_phu = 0  #khoi tao
z = A.shape #lay ca so hang va so cot
n = A.shape[0]  #lay so hang (row) cua mtran A
m = A.shape[1]  #lay so cot (col) cua mtran A

for i in range(n):
    for j in range(n):
        if i + j > n - 1:  #phia duoi duong cheo phu (tam giac ben phai)
            tong_duoi_cheo_phu += A[i, j]

print("Tong duoi duong cheo phu:", tong_duoi_cheo_phu)

#hang cua ma tran
hang = np.linalg.matrix_rank(A)
print("Hang cua ma tran:", hang)