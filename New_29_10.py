from sklearn.naive_bayes import GaussianNB
import numpy as np

# Refund, Status, Income
X_train = np.array([
    [1, 0, 125],  # Tid 1
    [0, 1, 100],
    [0, 0, 70],
    [1, 1, 120],
    [0, 2, 95],
    [0, 1, 60],
   # [1, 2, 220],
    [0, 0, 85],
    [0, 1, 75],
    [0, 0, 90]
])

#y_train = np.array([0, 0, 0, 0, 1, 0, 0, 1, 0, 1])
y_train = np.array([0, 0, 0, 0, 1, 0, 1, 0, 1])  #Khi xoa dong 7 (ca X_train va y_train)

# Mau can du doan: x = {refund = no, Status = Divorced, Income = 120}
x_test = np.array([0, 2, 120])

# So luong ban ghi
n_samples = len(y_train)
print("So luong ban ghi: ", n_samples)

# Thuoc tinh roi rac
sym_index = [0, 1]

# Thuoc tinh lien tuc
num_index = [2]

# Xay dung ham tinh cac xac suat tien nghiem
def prior(decision_array):
    priors = {}
    classes = np.unique(decision_array)

    for c in classes:
        count_c = np.sum(decision_array == c)
        priors[c] = count_c / n_samples

    return priors

priors = prior(y_train)
print("Xac suat tien nghiem: ", priors)

def posterior_class(c, X_train, y_train, prior_c, x_test):
    X_class = X_train[y_train == c]
    n = len(X_class)  # n trong laplace

    # Khoi tao gia tri xac suat ban dau bang tien nghiem
    posterior = prior_c

    # Voi cac thuoc tinh roi rac -> Laplace
    for col in sym_index:
        v = len(np.unique(X_train[:, col]))
        n_c = np.sum(X_class[:, col] == x_test[col])
        p = (n_c + 1) / (n + v)
        posterior *= p

    # Voi cac thuoc tinh lien tuc -> Gaussian
    for col in num_index:
        num_vals = X_class[:, col].astype(float)

        mean_val = np.mean(num_vals)
        var_val = np.var(num_vals)
        std_val = np.std(num_vals)

        x_val = x_test[col]

        p_num = (1 / (np.sqrt(2 * np.pi) * std_val)) * np.exp(
            -((x_val - mean_val) ** 2) / (2 * var_val)
        )

        posterior *= p_num

    return posterior

posterior = {}
classes = np.unique(y_train)
for c in classes:
    posterior[c] = posterior_class(c, X_train, y_train, priors[c], x_test)
    print(f"Posterior (x | Evade = {c}) = {posterior[c]:.8f}")
