import numpy as np
'''
x =np.array([1, 5, 3, 13, 7, 2])
x_cumsum = np.cumsum(x)
print("tính tổng tích lũy: ", x_cumsum)

val_mean = np.mean(x)
print("Gia tri tb: ",val_mean)

val_std = np.std(x)
print("Do lech chuan: ",val_std)

#Tim vi tri lon nhat
p1 = np.argmax(x)
#Tim vi tri nho nhat
p2 = np.argmax(x)

#Tim cac vi tri co gia tri lon nhat
ps_max = np.where(x == np.max(x))[0]  #[0] lay mang khoi tuple

'''


'''
x = np.array([1, 5, 3, 9,13, 7, 2, 13])
#Kiem tra dkien >=7 neu dung +4, neu sai -1
#Cach 1:
y = np.copy(x)
y[x>=7] = y[x>=7]-4  #Giu nguyen so ptu, chi sua cac ptu tman dkien
y[x<7] = y[x<7] + 1
print(y) 

#Cach 2:
x = np.where(x >= 7, x-4, x+1)  #np.where(dkien, neu dung, neu sai)

print(x)
'''


'''
x = np.array([1, 5, 3, 9, 13, 7, 2, 13])
#Kiem tra xem co phan tu nao trong x > 11 hay k?
check = np.any(x>11)  # == bool (voi 1 gia tri)
print(check) 
'''

'''
x = np.array([1, 5, 3, 9, 13, 7, 2, 13])
#Ktra tat ca cac phan tu trong x co >= 5 hay k?
check = np.all(x>=5)  # == bool (voi tat ca)
'''

'''
x1 = np.array([1, 5, 3, 9, 13, 7, 2, 13])
x2 = np.array([2, 8, 1, 3, 9, 0, 2, 8])
#Noi vector x1 vao sau vector x2
x = np.concatenate((x2, x1))
#Dem so luong phan tu co gia tri > 4
x2 = len(x2[x2>4])
print(x2)
count = np.count_nonzero(x > 4)  #dem so phan tu khac 0 hoac dem so ptu tman 1 dkien
print(count)
'''


'''
x1 = np.array([1, 5, 3, 9, 13, 7, 2, 13])
x2 = np.array([2, 8, 1, 3, 9, 0, 2, 8])
#In ra mot vector x gom cac gia tri nho nhat tuong ung giua 2 mang
x = np.minimum(x1, x2)  #lay gia tri nho hon tuong ung tung cap ptu
print(x)
x = np.maximum(x1, x2)
print(x)
'''


A = np.array([
    [1.4, 0.7, 2.9, 8.5, 7.2],
    [1.1, 1.5, 4.7, 2.9, 5.1],
    [2.8, 2.9, 6.2, 7.8, 8.8],
    [2.5, 6.8, 1.3, 0.0, 4.0],
    [2.3, 5.0, 2.7, 8.0, 9.0]]
)

# 1. Tinh dinh thuc cua ma tran A
det_A = np.linalg.det(A)
print("Dinh thuc cua A =", det_A)

# 2. Tim ma tran nghich dao cua A
A_inv = np.linalg.inv(A)
print("\nMa tran nghich dao cua A:")
print(A_inv)

# 3. Tinh cac gia tri rieng cua ma tran A
eigenvalues = np.linalg.eigvals(A)
print("\nCac gia tri rieng:")
print(eigenvalues)

# 4. Tim cac vector rieng tuong ung voi moi gia tri rieng
eigenvalues, eigenvectors = np.linalg.eig(A)

print("\nCac vector rieng:")
for i in range(len(eigenvalues)):
    print("Gia tri rieng:", eigenvalues[i])
    print("Vector rieng:")
    print(eigenvectors[:, i]) #do no la mang 2 chieu vector