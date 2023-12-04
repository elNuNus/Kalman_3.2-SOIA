""" Kalman 3.2
Ex 4.1 - Gaussian distribution
p87
@Author: Agnus Oscar"""

# === Imports === #
import numpy as np
import matplotlib.pyplot as plt

# === Parameters === #
n = 2   # dimension of the state vector
N = 100 # number of points to plot in one dimension
# ================== #

def pi(x1, x2, x_, Γ):
    """
    Computes the π function for a Gaussian distribution
    :param x1: scalar
    :param x2: scalar
    :param x_: mean vector
    :param Γ: covariance matrix
    :return: scalar
    """
    det_Γ = np.linalg.det(Γ)
    inv_Γ = np.linalg.inv(Γ)
    x = np.array([x1,x2]).reshape(2,1) # x = [x1, x2], where x1 and x2 are scalars
    x_ = x_.reshape(2,1)
    # Uncentered Gaussian distribution
    return 1/(det_Γ * np.sqrt((2*np.pi)**2)) * np.exp(-0.5*(x-x_).reshape(1,2) @ inv_Γ @ (x-x_))
    # # Centered Gaussian distribution
    # return 1/(det_Γ * np.sqrt((2*np.pi)**2)) * np.exp((-1/2)*x.reshape(1,2) @ inv_Γ @ x)


if __name__ == '__main__':

    # === Useful variables === #
    x_moy = np.array([1,2]).reshape(2,1)
    Γx = np.array([[1, 0],
                   [0, 1]])
    A = np.array([[np.cos(np.pi/6), -np.sin(np.pi/6)], [np.sin(np.pi/6), np.cos(np.pi/6)]]) @ np.array([[1, 0], [0, 3]])
    B = np.array([2, -5]).reshape(2, 1)
    # ======================== #


    # ___1. Draw the graph and coutour lines of pi_x, x_moy and Γx; in 3D
    x1 = np.linspace(-5, 5, N)
    x2 = np.linspace(-5, 5, N)
    X1, X2 = np.meshgrid(x1, x2)

    # Evaluate pi_x for each point of the meshgrid
    Z = np.zeros_like(X1)
    for i in range(len(x1)):
        for j in range(len(x2)):
            x = np.array([x1[i], x2[j]]).reshape(2, 1)
            Z[i, j] = pi(x1[i], x2[j], x_moy, Γx)

    # === 3D plot === #
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X1, X2, Z, cmap='viridis')
    ax.set_xlabel('x1',color='r')
    ax.set_ylabel('x2',color='g')
    ax.set_zlabel('z',color='b')
    plt.title('3D plot of π_x')
    # plt.show()
    # =============== #

    # ___2. Now with y
    y_moy = A @ x_moy + B
    print(y_moy)
    Γy = A @ Γx @ A.T #
    # Γy = (A @ B) @ Γx @ (A @ B).T # But in this case Γx = I => Γy = (A @ B)**2

    # Evaluate pi_y for each point of the meshgrid
    W = np.zeros_like(X1)
    for i in range(len(x1)):
        for j in range(len(x2)):
            x = np.array([x1[i], x2[j]]).reshape(2, 1)
            y = A @ x + B
            y1, y2 = y[0, 0], y[1, 0]
            W[i, j] = pi(y1, y2, y_moy, Γy)

    # print(W == Z)

    # === 3D plot === #
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(y1, y2, W, cmap='viridis')
    ax.set_xlabel('y1', color='r')
    ax.set_ylabel('y2', color='g')
    ax.set_zlabel('w', color='b')
    plt.title('3D plot of π_y')
    plt.show()
    # =============== #