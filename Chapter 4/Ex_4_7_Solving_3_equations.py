""" Kalman 3.2
Ex 4.7 - Solving 3 equations using a linear estimator
p90
@Author: Agnus Oscar"""

# === Imports === #
import numpy as np
import matplotlib.pyplot as plt

# === Parameters === #
Γ_β = np.array([[1, 0, 0],
                [0, 4, 0],
                [0, 0, 4]])
C = np.array([[2, 3],
              [3, 2],
              [1, -1]])
Y = np.array([8, 7, 0]).reshape(3, 1)
# ================== #

# === Unknowns === #
Γ_x = 10**4 * np.identity(2)
X_ = np.zeros((n, 1))
# ================== #

if __name__ == '__main__':
    # Solve the system by using a linear estimator
    Y_tilde = Y - C @ X_

    Γ_y = C @ Γ_x @ C.T + Γ_β
    # Γ_y = Y_tilde @ Y_tilde.T # This is the definition of Γ_y, but in this case it is not true because X_ is arbitrarily chosen
    Γ_xy = X_ @ Y_tilde.T

    K = Γ_x @ C.T @ np.linalg.inv(Γ_y)

    Γ_ε = Γ_x - K @ C @ Γ_x
    X_hat = X_ - K @ Y_tilde
    print('Estimated prameters:\n')
    print(f'X_hat = {X_hat}')
    print(f'Γ_ε = {Γ_ε}')
