""" Kalman 3.2
Exo 3.1 - Representation of a quadratic function
p63
@Author: Agnus Oscar"""


import numpy as np
import matplotlib.pyplot as plt

# === Parameters === #

# Set the area of the graph
x = np.linspace(-10, 10, 100)
y = np.linspace(-10, 10, 100)
X = np.array([x, y])

# ================== #

# === Functions === #

def least_square(A, Y):
    """Solve the least square problem Mx = y"""
    return np.linalg.inv(A.T @ A) @ A.T @ Y

# ================= #

if __name__ == '__main__':
    print('# === Exercice 3.1 === #')

    # Design matrix
    A = np.array([[ 4, 0],
                  [10, 1],
                  [10, 5],
                  [13, 5],
                  [15, 3]])

    # Observation vector
    Y = np.array([5, 10, 8, 14, 17])

    # Unkonwn vector
    X_hat = least_square(A, Y)
    print(X_hat)

    # Filtered observation
    Y_hat = A @ X_hat

    # Residual
    R = Y - Y_hat

    print(Y_hat)
    print(R)
    print("\n")

    estimated_Y = np.array([X_hat[0] * 20 + X_hat[1] * 10])
    print(estimated_Y)