""" Kalman 3.2
Ex 3.5 - Monte Carlo Method
p64
@Author: Agnus Oscar"""


import numpy as np
import matplotlib.pyplot as plt

# === Parameters === #

n = 1000
k_max = 5
epsilon = 1

y_m = [[0, 0, 0, 0, 0, 0] for i in range(n)] # The measured output vector
# y_m = np.zeros((n, k_max))
u = 1

y_real = np.array([0, 1, 2.5, 4.1, 5.8, 7.5]) # The real output vector, that we’re going to compare with the measured output vector

# ================== #

if __name__ == '__main__':
    # Generate a cloud of vectors p = (a, b) using a uniform law

    for i in range(n):
        p = np.random.uniform(0,2, size=2)
        a, b = p[0], p[1]
        A = np.array([[1, 0],
                      [a, 0.3]])
        B = np.array([[b], [1 - b]])
        C = np.array([1, 1])

        x = np.array([[0], [0]])

        for k in range(6):
            y = C @ x
            y_m[i][k] = y[0]
            x = A @ x + B * u

        # Draw the p vectors such that the output vector is close to the measured output vector, at least at an epsilon precision
        residual = np.linalg.norm(y_m[i] - y_real) # Compute the norm of the residual vector
        print(residual)
        if residual <= epsilon:
            plt.plot(p[0], p[1], 'ro')
        else:
            plt.plot(p[0], p[1], 'bo')

    plt.show()