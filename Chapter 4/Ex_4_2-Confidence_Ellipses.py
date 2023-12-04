""" Kalman 3.2
Ex 4.2 - Confidence Ellipses
p
@Author: Agnus Oscar"""

# === Imports === #
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

# === Parameters === #

N = 100 # Number of points on the unit circle
w_moy = 0 # unused here, make sure not to forget it in the future
heta = 0.9

A1 = np.array([[1,0],
               [0,3]])
A2 = np.array([[np.cos(np.pi/4),-np.sin(np.pi/4)],
               [np.sin(np.pi/4), np.cos(np.pi/4)]])

sqrt_A1 = sp.linalg.sqrtm(A1)
sqrt_A2 = sp.linalg.sqrtm(A2)

Γ1 = np.eye(2)
Γ2 = 3 * Γ1
Γ3 = A1 @ Γ2 @ A1.T + Γ1
Γ4 = A2 @ Γ3 @ A2.T
Γ5 = Γ4 + Γ3
Γ6 = A2 @ Γ5 @ A2.T

# ================== #

def Ellipse(theta, Γ, a, w_moy=np.array([0,0])):
    """ Returns the coordinates of the ellipse associated with the covariance matrix Γ
        :param theta: angle
        :param Γ: covariance matrix
        :param a: a confidence level
        :param w_moy: mean vector

        :return: the coordinates of the point on the ellipse associated with the covariance matrix Γ
        """
    u = np.array([np.cos(theta), np.sin(theta)]).reshape(2, 1) # Unit circle
    return w_moy + a * sp.linalg.sqrtm(Γ) @ u

if __name__ == '__main__':

    # 1. Associate each covariance matrix with its confidence ellipse

    # 2. Verify the results by regenerating these ellipses

    θ = np.linspace(0, 2*np.pi, N)
    a = np.sqrt(-2*np.log(1-heta))
    #
    # x = np.cos(θ)
    # y = np.sin(θ)
    #
    # X = np.array([x, y]) # the n points on the unit circle
    # Y = np.zeros_like(X)

    X = np.zeros(N)
    Y = np.zeros_like(X)

    # === 2D plot === #
    plt.figure()
    plt.title('Kalman 3.2 - Ex 4.2 Confidence ellipses')
    plt.xlabel('x')
    plt.ylabel('y')

    k = 0 # Γ counter

    for Γ in [Γ1, Γ2, Γ3, Γ4, Γ5, Γ6]:
        k += 1
        for i in range(len(θ)):
            X[i], Y[i] = Ellipse(θ[i], Γ, a)[:,0]

        plt.plot(X, Y, label=('Confidence ellipse Γ'+str(k)))

    plt.legend()
    plt.show()
