""" Kalman 3.2
Ex 5.2 - Weather Forcast
p109
@Author: Agnus Oscar"""

# === Imports === #
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
# === Parameters === #

# ================== #

if __name__ == '__main__':
    A = np.array([[.9, .5],
                  [.1, .5]])

    print(A**10)