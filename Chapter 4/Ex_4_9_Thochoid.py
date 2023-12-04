""" Kalman 3.2
Ex 4.9  - Trochoid
p90
@Author: Agnus Oscar"""
import matplotlib.pyplot as plt
# === Imports === #
import numpy as np
# ================ #

# === Parameters === #
n = 2
m = 4

Γ_β = 0.1 * np.identity(m)

C = np.array([[1, np.cos(1)],
              [1, np.cos(2)],
              [1, np.cos(3)],
              [1, np.cos(7)]])

T = np.array([1, 2, 3, 7]).reshape(m, 1)
Y = np.array([0.38, 3.25, 4.97, -0.26]).reshape(m, 1)
# ================== #

# === Unknowns === #
Γ_x = 10**4 * np.identity(n)
X_ = np.zeros((n, 1))
# ================== #

if __name__ == '__main__':
    # Solve the system by using a linear estimator
    Y_tilde = Y - C @ X_
    Y_ = np.zeros((m, 1))

    Γ_y = C @ Γ_x @ C.T + Γ_β

    K = Γ_x @ C.T @ np.linalg.inv(Γ_y)

    X_tilde = K @ Y

    Γ_ε = Γ_x - K @ C @ Γ_x
    X_hat = X_ + K @ Y_tilde
    print('Estimated prameters:\n')
    print(f'X_hat = {X_hat}')
    print(f'Γ_ε = {Γ_ε}')

    # Plot the trochoid
    p1, p2 = X_hat
    t = np.linspace(0, 10, 100)
    x = p1 * t - p2 * np.sin(t)
    y = p1 - p2 * np.cos(t)
    plt.plot(x, y)
    plt.title('Ex 4.9 - Estimated trochoid')
    plt.show()
