""" Kalman 3.2
Exo 3.2 - Identification of a parabola
p65
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

# ================= #


if __name__ == '__main__':

    print('#=== Exercice 3.2 === ###')

    # M contains the "x" coordinate of the points to be passed through (t in this case), elevated to ², then to 1, then to 0
    M = np.array([[9,-3, 1],
                  [1,-1, 1],
                  [0, 0, 1],
                  [4, 2, 1],
                  [9, 3, 1],
                [36, 6, 1]])

    # Y contains the "y" coordinate of the points to be passed through (y in this case)
    Y = np.array([17, 3, 1, 5, 11, 46])

    # === Compute the coefficients of the parabola === #
    X_hat = least_square(Y, M)
    p1, p2, p3 = X_hat[0], X_hat[1], X_hat[2]
    print("X_hat : ",X_hat)
    print("\n")
    # ================================================ #

    # == Extra stuff == #
    filtered_y = M @ X_hat
    residual = y - filtered_y

    print("filtered_y : ",filtered_y)
    print("residual : ",residual)
    print("\n")
