""" Kalman 3.2
Ex 4.12 - Estimating the parameters of an electric motor
p 92
@Author: Agnus Oscar"""

# === Imports === #
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

# === Parameters === #
n = 2 # Number of unknowns

Γ_α = np.array([[1, 0],
                [0, 1]])

Γ_β = np.array([1])

A0 = np.array([[.5, 0],
               [0, 1]])
A1 = np.array([[1, -1],
               [1, 1]])
A2 = np.array([[1, -1],
               [1, 1]])
A = np.array([A0, A1, A2])


C0 = np.array([[1, 1]])
C1 = np.array([[1, 1]])
C2 = np.array([[1, 1]])
C = np.array([C0, C1, C2])


u0 = np.array([8, 16]).reshape(2, 1)
u1 = np.array([-6, -18]).reshape(2, 1)
u2 = np.array([32, -8]).reshape(2, 1)

U = np.array([0, 0 ,0])

Y = np.array([7, 30, -6]).reshape(3, 1)
# ================== #

# === Unknowns === #
Γ_x = 10**2 * np.identity(n)
X_ = np.zeros((n, 1))
# ================== #

def Kalman(Γ_x, X_, Γ_β, A, C, U, Y):
    """ Kalman filter, for one iteration.
    """
    Γ_y = C @ Γ_x @ C.T + Γ_β
    K = Γ_x @ C.T @ np.linalg.inv(Γ_y)

    # === Correction === #
    X_ = X_ + K @ (Y - C @ X_)
    Γ_x = (np.identity(n) - K @ C) @ Γ_x
    Y_tilde = Y - C @ X_

    # === Prediction === #
    X_ = A @ X_ + U
    Γ_x = A @ Γ_x @ A.T + Γ_α

    return Γ_x, X_

def Ellipse(theta, Γ, a, w_moy=np.array([0,0])):
    """ Returns the coordinates of the ellipse associated with the covariance matrix Γ
        :param theta: angle
        :param Γ: covariance matrix
        :param a: a confidence level
        :param w_moy: mean vector

        :return: the coordinates of the point on the ellipse associated with the covariance matrix Γ
        """
    u = np.array([np.cos(theta), np.sin(theta)]).reshape(2, 1) # Unit circle
    return w_moy.reshape(2, 1) + a * sp.linalg.sqrtm(Γ) @ u


if __name__ == '__main__':
    # Solve the system by using a Kalman filter

    # === Parameters === #
    heta = 0.95
    N = 100
    # ================== #


    plt.legend()
    plt.show()