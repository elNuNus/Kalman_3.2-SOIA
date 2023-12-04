""" Kalman 3.2
Ex 4.3 - Confidence ellipse prediction
p
@Author: Agnus Oscar"""

# === Imports === #
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import scipy.linalg

# === Parameters === #
N = 1000 # Number of points
x_ = np.array([1,2])
Γ_x = np.array([[4, 3],
                [3, 3]]) # Covariance matrix
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
    return w_moy.reshape(2, 1) + a * sp.linalg.sqrtm(Γ) @ u

if __name__ == '__main__':

    # ___1. Generate a cloud of N points representing a random Gaussian vector centered in R^2 (covariance = Identity)

    X_cloud = np.random.multivariate_normal([0, 0], np.identity(2), N).reshape(2, N)
    for i in range(N):
        X_cloud[:, i] = scipy.linalg.sqrtm(Γ_x) @ X_cloud[:, i] + x_

    # # Using lists
    # X_cloud = [[], []]
    # for i in range(N):
    #     point = np.random.multivariate_normal(x_, Γ_x)
    #     X_cloud[0].append(point[0])
    #     X_cloud[1].append(point[1])

    # # Altenatively, this could work too:
    # X_cloud = np.random.Generator.multivariate_normal(x_, Γ_x, N).reshape(2,N)


    # ___2. Draw the confidence ellipses for different values of heta with the cloud X
    plt.figure()

    θ = np.linspace(0, 2*np.pi, N)

    # Plot of the cloud
    plt.scatter(X_cloud[0], X_cloud[1], alpha=0.6, label='Cloud of random points')

    # Plot of the ellipses
    for heta in [0.9, 0.99, 0.999]:
        # Compute the ellipse
        a = np.sqrt(-2 * np.log(1-heta))
        n = len(θ)
        X1 = np.zeros(N)
        X2 = np.zeros_like(X1)
        for i in range(n):
            X1[i], X2[i] = Ellipse(θ[i], Γ_x, a, x_)[:,0]
        # Plot the ellipse
        plt.plot(X1, X2, label='Confidence ellipse for heta = {}'.format(heta))

    # === 2D plot === #
    plt.title('Kalman 3.2 - Ex 4.3 Confidence ellipses : prediction'
              '\nCloud of points of a random Gaussian vector centered in R^2')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()

    # ___3. Find an estimation of x_ and Γ_x from the cloud X

    # Estimation of x_
    x_est = np.mean(X_cloud, axis=1)
    Γ_x_est = np.cov(X_cloud)

    # Comparison
    print('x_ = {}'.format(x_), end='\n\n')
    print('x_est = {}'.format(x_est), end='\n\n')
    print('Γ_x = {}'.format(Γ_x), end='\n\n')
    print('Γ_x_est = {}'.format(Γ_x_est), end='\n\n')


    # ___4. Write a program that illustrates the evolution of the particle cloud in time for given values of t
    dt = 0.01     # Time step
    n = int(5/dt) # Number of time steps, total duration = 5s

    A = np.array([[ 0, 1],
                  [-1, 0]])
    B = np.array([2, 3]).reshape(2, 1)

    alpha = np.random.multivariate_normal([0, 0], dt * np.identity(2), N).reshape(2, N) # White Gaussian noise

    plt.figure()

    # Euler integration
    # X1 and X2 store the abscissa and ordinate of the cloud at each time step
    X1 = np.zeros((n, N))        # abscissa
    X2 = np.zeros_like(X1)       # ordinate
    X1[0,:], X2[0,:] = X_cloud[0,:], X_cloud[1,:] # Initial position of the cloud

    for k in range(n):                      # For each time step
        X_cloud_k = np.zeros_like(X_cloud)  # Stores the new cloud at time k

        for i in range(N):                  # For each point i in the cloud
            # x1 = X_cloud_k[0, i] + dt * (A @ X_cloud_k[:, i] + B * np.sin(k * dt)) + alpha[0, i]
            # x2 = X_cloud_k[1, i] + dt * (A @ X_cloud_k[:, i] + B * np.sin(k * dt)) + alpha[1, i]

            x = X_cloud_k[:, i] + dt * (A @ X_cloud_k[:, i] + B * np.sin(k * dt)) + alpha[:, i]

            X_cloud_k[0, i] = x[0, 0] # for some reason, x is a (2,2) matrix, so we need to extract the first row
            X_cloud_k[1, i] = x[1, 0]

        X1[k,:] = X_cloud_k[0,:]
        X2[k,:] = X_cloud_k[1,:]

    # Plot of the cloud in time
    for t in [0, 1, 2, 3, 4, 5 - dt]:
        k = int(t/dt)
        plt.scatter(X1[k], X2[k], alpha=0.6, label='Cloud of points of a random moving vector for t = {}'.format(t))

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Kalman 3.2 - Ex 4.3 Confidence ellipses : prediction'
              '\nEvolution of the particle cloud in time')
    plt.legend()
    plt.show()

    # ___5. Represent the evolution using only Kallman predictions and compare the resulting ellipses with the question 4

    # We'll just add the ellipses on the plot of the previous question
    dt = 0.01  # Time step
    n = int(5 / dt)  # Number of time steps, total duration = 5s

    A = np.array([[0, 1],
                  [-1, 0]])
    B = np.array([2, 3]).reshape(2, 1)

    alpha = np.random.multivariate_normal([0, 0], dt * np.identity(2), N).reshape(2, N)  # White Gaussian noise

    plt.figure()

    # Euler integration
    # X1 and X2 store the abscissa and ordinate of the cloud at each time step
    X1 = np.zeros((n, N))  # abscissa
    X2 = np.zeros_like(X1)  # ordinate
    X1[0, :], X2[0, :] = X_cloud[0, :], X_cloud[1, :]  # Initial position of the cloud

    for k in range(n):  # For each time step
        X_cloud_k = np.zeros_like(X_cloud)  # Stores the new cloud at time k

        for i in range(N):  # For each point i in the cloud
            # x1 = X_cloud_k[0, i] + dt * (A @ X_cloud_k[:, i] + B * np.sin(k * dt)) + alpha[0, i]
            # x2 = X_cloud_k[1, i] + dt * (A @ X_cloud_k[:, i] + B * np.sin(k * dt)) + alpha[1, i]

            x = X_cloud_k[:, i] + dt * (A @ X_cloud_k[:, i] + B * np.sin(k * dt)) + alpha[:, i]

            X_cloud_k[0, i] = x[0, 0]  # for some reason, x is a (2,2) matrix, so we need to extract the first row
            X_cloud_k[1, i] = x[1, 0]

        X1[k, :] = X_cloud_k[0, :]
        X2[k, :] = X_cloud_k[1, :]

    # Plot of the cloud in time
    for t in [0, 1, 2, 3, 4, 5 - dt]:
        k = int(t / dt)
        plt.scatter(X1[k], X2[k], alpha=0.6, label='Cloud of points of a random moving vector for t = {}'.format(t))


    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Kalman 3.2 - Ex 4.3 Confidence ellipses : prediction'
              '\nEvolution of the particle cloud in time, with Kalman predictions')
    plt.legend()
    plt.show()
