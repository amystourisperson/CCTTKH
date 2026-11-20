import numpy as np

'''
X = np.array([
    [0.4, 2.2, 1.9],
    [0.5, 1.3, 1.6],
    [0.8, 2.7, 2.4]
])

#Sap xep lai ma tran X theo thu tu tang dan phan tu cuoi tung dong
print(X[:, -1])  #Lay cot cuoi cua cac hang
X = X[np.argsort(X[:, -1])]
print(X)

#SX tang dan theo Hieu giua phan tu cuoi va phan tu dau cua moi dong
X_hieu = np.argsort(X[:, -1] - X[:, 0])
print("Hieu: ", X[X_hieu])

#Cach 2:
X_new = sorted(X, key=lambda row:row[-1]-row[0])
X_new = np.array(X_new)
print(X_new)
'''

b = np.array([1, 3, 7, 10, 28, 30, 31])
#In ra mot mang b moi
#Neu la so nguyen to thi binh phuong, ngc lai giu nguyen
#Cach 1:
def check_snt(n):
    if(n < 2):
        return False
    else:
        for i in range(2, int(n**0.5)+1):
            if(n % i == 0):
                return False
        return True
b_new = b.copy()
for i in range(len(b)):
    if(check_snt(b[i])):
        b_new[i] = b[i]**2
b_new = np.array(b_new)
print("Cach 1: ", b_new)

#Cach 2:
c_new = [i**2 if check_snt(i) else i for i in b]
c_new = np.array(c_new)
print("Cach 2: ", c_new)

#Cach 3: dung Where
check = [check_snt(i) for i in b]
d_new = np.where(check, b**2, b)
d_new = np.array(d_new)
print("Cach 3: ", d_new)

#Cach 4:
mask = np.vectorize(check_snt)(b)
e_new = np.where(mask, b**2, b)
print("Cach 4: ", e_new)

#In ra mot mang, chi co binh phuong cac so nguyen to
#Cach 1:
z_new = [i**2 for i in b if check_snt(i)]
z_new = np.array(z_new)
print("Cach 1: ", z_new)

#Cach 2:
mask = np.vectorize(check_snt)(b)
z_new = b[mask]**2
print("Cach 2: ", z_new)

#Cach 3:
z_new = list(map(lambda i : i**2, filter(check_snt, b)))
z_new = np.array(z_new)
print("Cach 3: ", z_new)

A = np.array([
    [0.2, 1.0, 0.4, 0.8, 0.1],
    [0.9, 0.6, 0.6, 0.7, 0.2],
    [0.5, 1.4, 0.3, 0.2, 0.1],
    [0.2, 0.1, 0.0, 0.9, 0.0],
    [0.1, 0.3, 0.2, 0.8, 0.1]
])
print("Ma tran:", A)
#Tinh dinh thuc va tim ma tran nghich dao cua A
dt = np.linalg.det(A)
print("Dinh thuc: ", dt)

nghichdao = np.linalg.inv(A)
print("Nghich dao: \n", nghichdao)

#Tim gia tri rieng, vector rieng theo thu tu giam dan cac gia tri rieng
a, b = np.linalg.eig(A)
a = np.real(a)
b= np.real(b)
# Sắp xếp giá trị riêng theo thứ tự giảm dần
idx = np.argsort(a)[::-1]

a = a[idx]
b = b[:, idx]

print("\nGia tri rieng giam dan:")
print(a)

print("\nVector rieng tuong ung:")
print(b)

'''
Viet mot ham truyen vao 1 ma tran va mot so S, sau do:
+Tim vi tri cac dong, sao cho tong cua chung > S
+So luong dong thu duoc phai la nho nhat
+Tong gia tri cac dong phai la nho nhat
+In ra mang chua cac vi tri do
'''

#Cach 1:
def findRows(matrix, S):
    box = []
    sum_row = np.sum(matrix, axis=1)
    print(sum_row)
    f_row = np.argmin(sum_row)
    box.append(f_row)
    sum_tg = sum_row[f_row]
    sum_row[f_row] = np.max(sum_row) + 1000
    while sum_tg <= S and len(box) < len(matrix):
        inr = np.argmin(sum_row)
        sum_tg += sum_row[inr]
        box.append(inr)
        sum_row[inr] = np.max(sum_row) + 1000
    return box

test = findRows(A, 4.1)
test = np.array(test)
print(test)

#Cach 2:
def cach2_findRows(matrix, S):
    sum_row = np.sum(matrix, axis=1)
    ind_sort = np.argsort(sum_row)
    cumsum = np.cumsum(sum_row[ind_sort])
    k = np.argmax(cumsum > S)
    return ind_sort[:k+1]


test = findRows(A, 4.1)
print(test)

