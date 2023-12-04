""" Kalman 3.2
Exo 3.3 - Identifying parameters of a DC Motor
p63
@Author: Agnus Oscar"""


import numpy as np
import matplotlib.pyplot as plt

# === Parameters === #
# Set the area of the graph
x1 = np.linspace(-10, 10, 100)
x2 = np.linspace(-10, 10, 100)

# Points to be passed through
t = np.array([-3, -1, 0, 2, 3, 6])
y = np.array([17, 3, 1, 5, 11, 46])

# Coefficients of the parabola
# p1, p2, p3 = np.polyfit(t, y, 2)

# # Matrixes
# Q = p1
# L = p2
# C = p3
# =================== #

# === Functions === #
def least_square(Y, A):
    """Solve the least square problem Mx = y"""
    return np.linalg.inv(A.T @ A) @ (A.T @ Y)

def parabola(X):
    Z = np.zeros((len(X[0]), len(X[1])))
    for i in range(len(X[0])):
        for j in range(len(X[1])):
            Z[i, j] = X[0]
    return Z # Z must be 2 dimensional
# ================= #


if __name__ == '__main__':
    print('### Exercice 3.3 ###')

    M = np.array([[ 4, 0],
                  [10, 1],
                  [10, 5],
                  [13, 5],
                  [15, 3]])

    Y = np.array([5, 10, 8, 14, 17])

    X_hat = least_square(Y, M)
    print("X_hat : ")
    print(X_hat)
    print("\n")

    # Filtered y
    Y_hat = M @ X_hat
    # Residual vector
    residual = Y - Y_hat

    print("Y_hat : ")
    print(Y_hat)
    print("Residual : ")
    print(residual)
    print("\n")

    estimated_y = np.array([X_hat[0] * 20 + X_hat[1] * 10])
    print("Estimated y : ")
    print(estimated_y)