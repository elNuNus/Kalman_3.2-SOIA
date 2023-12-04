""" Kalman 3.2
Ex 4.8 - Estimating the parameters of an electric motor using a linear estimator
p90
@Author: Agnus Oscar"""

# === Imports === #
import numpy as np
# ================ #

# === Parameters === #
n = 2
m = 5

Γ_β = np.identity(m)*9
C = np.array([[4, 0],
              [10, 1],
              [10, 5],
              [13, 5],
              [15, 3]])
Y = np.array([5, 10, 8, 14, 17]).reshape(m, 1)
# ================== #

# === Unknowns === #
Γ_x = np.identity(n)*4
X_ = np.array([[1, -1]]).reshape(n, 1)
# ================== #

if __name__ == '__main__':
    # Solve the system by using a linear estimator
    Y_tilde = Y - C @ X_

    Γ_y = C @ Γ_x @ C.T + Γ_β
    # Γ_y = Y_tilde @ Y_tilde.T # This is the definition of Γ_y, but in this case it is not true because X_ is arbitrarily chosen

    K = Γ_x @ C.T @ np.linalg.inv(Γ_y)

    Γ_ε = Γ_x - K @ C @ Γ_x
    X_hat = X_ + K @ Y_tilde
    print('Estimated prameters:\n')
    print(f'X_hat = {X_hat}')
    print(f'Γ_ε = {Γ_ε}')
