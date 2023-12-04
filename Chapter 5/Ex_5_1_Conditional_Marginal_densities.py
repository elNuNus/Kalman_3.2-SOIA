""" Kalman 3.2
Ex 5.1 - Conditional and Marginal densities
p109
@Author: Agnus Oscar"""

# === Imports === #
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
# === Parameters === #
pi_matrix = np.array([[.1, .2, 0],
                      [.1, .3, .1],
                      [0, .1, .1]])
# ================== #
# Marginal densities
def pi(a, b, pi_matrix):
    s = 0
    if b == "x":
        for i in range(len(pi_matrix)):
            s += pi_matrix[a-1, i] * i
    if b == "y":
        for i in range(len(pi_matrix)):
            s += pi_matrix[i, a-1] * i

    return s


if __name__ == '__main__':
    print("pi(1, x) = ", pi(1, "x", pi_matrix))
    print("pi(2, x) = ", pi(2, "x", pi_matrix))
    print("pi(3, x) = ", pi(3, "x", pi_matrix))
    print("\n")
    print("pi(1, y) = ", pi(1, "y", pi_matrix))
    print("pi(2, y) = ", pi(2, "y", pi_matrix))
    print("pi(3, y) = ", pi(3, "y", pi_matrix))