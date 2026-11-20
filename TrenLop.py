import numpy as np

x =[0, 1, 2, 3]
def DFT_1D(x):
    X = np.zeros(len(x), dtype=complex)
    for k in range(len(X)):
        for n in range(len(x)):
            theta = -2 * np.pi * k *n / len(x)
            mu = np.cos(theta) + 1j * np.sin(theta)
            X[k] += x[n] * mu
    return X
X_1 = DFT_1D(x)
print("Cach 1: ", X_1)


#Cach 2: Chi dung numpy
def DFT_1D(x):
    n = np.arange(len(x))
    k = n.reshape(len(x), 1)
    X = np.exp(-2j * np.pi * k * n / len(x))
    X = np.dot(x, X)
    return X

test = DFT_1D(x)
print("Cach 2: ", test)

#Cach 3:
X = np.fft.fft(x)
print("Cach 3: ",  X)

#Tach thanh 2 mang
mask = np.isclose(X.imag, 0, atol=1e-12)
sothuc = X[mask == True].real  #neu muon .real
soao = X[mask == False]
print("So thuc: ", sothuc)
print("So ao: ", soao)


x_1 = np.fft.ifft(X).real
print("dua no ve lai: ", x_1)


#2-D
I = np.array([
    [2, 3, 6, 9],
    [4, 1, 3, 7],
    [2, 5, 6, 6],
    [1, 4, 2, 4]
])

I_new = np.fft.fft2(I)  #fft2 chi dung cho mang 2 chieu
print("I: ", I_new)

#Doc file
data = np.genfromtxt("Example.csv", delimiter=",", dtype=object, skip_header=1) #skip_header: k lay dong dau tien
print(data)

#Cot chua du lieu so
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
#Phan chia cot chu va cot so
in_num = [0, 1, 3, 4, 5, 6]
in_sym = [2, 7]
for col in in_sym:  #Chuyen du lieu cac cot 2 va 7 ve dang so
    data[:, col] = LabelEncoder().fit_transform(data[:, col])
data[:, in_num] = data[:, in_num].astype(float)
data[:, in_sym] = data[:, in_sym].astype(int)

#Chuan hoa cot thu 3 theo min_max
data[:, [3]] = MinMaxScaler().fit_transform(data[:, [3]])
data_new = data.astype(float)

#Cau hinh
np.set_printoptions(precision=4, suppress=True, linewidth=120)
print(data_new)


