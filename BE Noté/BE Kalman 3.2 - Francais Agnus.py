""" Kalman 3.2 - BE Noté
pour le
11/12/2023

@Author : Pierre Français
@Author : Oscar Agnus
"""


import numpy as np
import scipy.linalg as sp
import matplotlib.pyplot as plt


# === Parameters ===#
dt = .5
noise = 1  # noise level, set to 0 to remove the noise
variance = .05 * dt  # variance of the noise
std = np.sqrt(variance)  # standard deviation of the noise, which is the square root of the variance
std = variance  # I suspect that the std and the variance are inverted in the subject

smooth_land = 1  # set to 1 to use the smoothing part on the landmarks
smooth_odo = 1  # set to 1 to use the smoothing part on the odometer
read_lidars = 0  # set to 1 to plot the lidars measurements

plot_pred = 1  # set to 1 to plot the predicted trajectory
plot_smooth = 1  # set to 1 to plot the smoothed trajectory
plot_land = 1  # set to 1 to plot the corrected trajectory using the landmarks
plot_odo = 1  # set to 1 to plot the corrected trajectory using the odometer

display_ellipses = 1  # set to 1 to display the ellipses

d = np.array([np.array([ 0,  7.8, 17.5, 33.8, 29.9]),
              np.array([10, 14.1, 10.2, 30.4, 31.9]),
              np.array([20, 19.8, 16.4, 22.7, 25.2]),
              np.array([30, 27.6, 22.5, 14.9, 21.8]),
              np.array([40, 33.9, 31.9,  8.3, 14.1]),
              np.array([50, 30.2, 31.3, 13.7, 10.9]),
              np.array([60, 23.5, 22.8, 18.1, 19.0]),
              np.array([70, 16.4, 17.4, 25.3, 24.6]),
              np.array([80, 13.8, 21.4, 28.7, 23.6])])


# Matrices
G_a = np.identity(3) * std  # process noise
G_b = np.identity(4) * np.sqrt(.02)  # measurement noise


def A(theta,delta,dt):
    return np.array([np.array([1, 0, dt * np.cos(delta) * np.cos(theta)]),
                     np.array([0, 1, dt * np.cos(delta) * np.sin(theta)]),
                     np.array([0, 0, 1])])


def C_k_land(k, X):
    return 2 * np.array([[ 2 - X[k, 0], -5 - X[k, 1], 0],
                         [17 - X[k, 0], -5 - X[k, 1], 0],
                         [17 - X[k, 0], 32 - X[k, 1], 0],
                         [32 - X[k, 0], 32 - X[k, 1], 0]])

def C_k_odo(k, X, odo_only = False):
    if not odo_only:
        return 2 * np.array([[ 2 - X[k, 0], -5 - X[k, 1], 0],
                             [17 - X[k, 0], -5 - X[k, 1], 0],
                             [17 - X[k, 0], 32 - X[k, 1], 0],
                             [32 - X[k, 0], 32 - X[k, 1], 0],
                             [0, 0, 1]])
    else:
        return np.array([[0, 0, 1]])


landmarks = np.array([[-2, 5], [17, -5], [17, 32], [-2, 32]])
# ==================#


# === Functions === #


def correction(x, Gk, y, C, Gb):
    if Gk.shape == (1,):  # if Gk is a scalar
        Gy = Gk*C@C.T + Gb
    else:
        Gy = C@Gk@C.T + Gb
    if Gy.shape == (1,):  # if Gy is a scalar
        K = Gk@C.T/Gy
        if Gk.shape == (1,):  # if Gk is a scalar
            K = Gk*C.T/Gy
    elif Gk.shape == (1,):  # if Gk is a scalar
        K = Gk*C.T@np.linalg.inv(Gy)
    else:
        K = Gk@C.T@np.linalg.inv(Gy)
    if C.shape == (1,): # if C is a scalar
        yTild = y - C*x
    elif y.shape == (1,):  # if y is a scalar
        yTild = y - C@x
    else:
        yTild = y - C@x
    if yTild.shape == ():
        xChap = x + K*yTild
    else :
        xChap = x + K@yTild
    if K.shape == () or C.shape == () or Gk.shape == ():
        G = Gk - K*C*Gk
    elif K.shape == (3,) or C.shape == (3,) or Gk.shape == (3,):
        G = Gk - K*C*Gk
    else:
        G = Gk - K@C@Gk
    return xChap, G


def prediction(xChap1, G1, A, u, Ga):
    xChap2 = A@xChap1 + u
    G2 = A@G1@A.T + Ga
    return xChap2, G2


def Kalman(x,Gk,y,C,GB,A,u,Ga):
    xk1, Gk1 = correction(x, Gk, y, C, GB)
    xk2, Gk2 = prediction(xk1, Gk1, A, u, Ga)
    return xk2, Gk2


def loadcsv(file1):
    fichier = open(file1, 'r')
    D = fichier.read().split("\n")
    fichier.close()
    for i in range(len(D)):
        D[i] = D[i].split(";")
    D = D[:-1]  # si la dernière ligne est vide
    D = np.array([[float(elt) for elt in Ligne] for Ligne in D])
    return D


def f(X, U, noise=0):
    x, y, theta, v, delta = X[0], X[1], X[2], X[3], X[4]
    alpha = np.random.normal(0, std, 3)*noise  # noise on the angles and the speed
    return np.array([v * np.cos(theta) * np.cos(delta),
                     v * np.cos(delta) * np.sin(theta),
                     v * np.sin(delta) / 3 + alpha[0],  # there was an error in the subject, it was sin(theta)
                     U[0] + alpha[1],
                     U[1] + alpha[2]])

def plot_ellipse(Cov,X_mean, c='red'):
    # Cov *= .1
    theta = np.arange(0, np.pi * 2, 0.01)
    x = np.cos(theta)
    y = np.sin(theta)
    X = np.array([x, y])
    X_new = np.array([X_mean[0], X_mean[1]]).reshape(2, 1)
    Cov_new = np.array([Cov[0, 0], Cov[0, 1],
                        Cov[1, 0], Cov[1, 1]]).reshape(2, 2)
    nu = 0.9
    a = np.sqrt(-2 * np.log(1 - nu))
    Y_plot = a*sp.sqrtm(Cov_new) @ X
    plt.plot(Y_plot[0]+X_new[0], Y_plot[1]+X_new[1], color=c, linewidth=.5)


# ================= #


if __name__ == '__main__':
    # === Load data ===#
    batiment = loadcsv("batiment.csv")
    consignes = loadcsv("consigne.csv")
    N = len(consignes)
    # ==================#

    # === Initialisation === #
    X_0 = np.array([1, 2.2, 0, 1, 0]).reshape((5, 1)) # initial state vector
    G_x_0 = np.eye(3) # initial covariance matrix

    # Simulation
    X_simu = np.zeros((N+1, 5))  # stores the simulated states using the Euler method
    X_simu[0] = X_0.flatten()

    # Prediction only, no correction
    X_no_corr = np.zeros((N+1, 3))  # stores the predicted states using the Kalman filter
    X_no_corr[0] = np.array([X_0[0], X_0[1], X_0[3]]).flatten()
    G_x_no_corr = np.zeros((N+1, 3, 3))  # stores the covariance matrices of the predicted states
    G_x_no_corr[0] = G_x_0

    # Smoothing with one point
    X_smooth = np.zeros((N+1, 3))  # stores the smoothed states using the Kalman filter
    G_x_smooth = np.zeros((N+1, 3, 3))  # stores the covariance matrices of the smoothed states

    # Correction and Prediction with landmarks
    X_land = np.zeros((N+1, 3))  # stores the corrected states using the Kalman filter and the landmarks
    X_land[0] = np.array([X_0[0], X_0[1], X_0[3]]).flatten()
    G_x_land = np.zeros((N + 1, 3, 3))  # stores the covariance matrices of the corrected states
    G_x_land[0] = G_x_0

    X_corr = np.zeros((N + 1, 3))  # stores the corrected states using the Kalman filter and the landmarks
    G_x_corr = np.zeros((N + 1, 3, 3))  # stores the covariance matrices of the corrected states

    X_pred = np.zeros((N + 1, 3))  # stores the predicted states using the Kalman filter
    G_x_pred = np.zeros((N + 1, 3, 3))  # stores the covariance matrices of the predicted states

    # Correction and Prediction with landmarks and the odometer
    X_odo = np.zeros((N+1, 3))  # stores the corrected states using the Kalman filter and the landmarks
    X_odo[0] = np.array([X_0[0], X_0[1], X_0[3]]).flatten()
    G_x_odo = np.zeros((N + 1, 3, 3))  # stores the covariance matrices of the corrected states
    G_x_odo[0] = G_x_0

    # ====================== #

    # === Simulation === #
    for k in range(1, N+1):
        X_simu[k] = X_simu[k-1] + dt * f(X_simu[k-1], consignes[k-1], 0)
    # ================== #

    # === Prediction only, no correction === #
    for k in range(1, N+1):
        theta, delta = X_simu[k-1][2], X_simu[k-1][4]  # we suppose that we know exactly the theta and delta angles
        # Prediction
        U = np.array([0, 0, consignes[k-1][0]]).reshape((3, 1))  # input vector, only with the speed input (U=0 here)
        U = U + np.array([0, 0, noise * np.random.normal(0, std)]).reshape((3, 1))  # we add noise to the input vector
        x_no_corr, g_x_no_corr = prediction(X_no_corr[k-1].reshape(3, 1), G_x_no_corr[k-1], A(theta, delta, dt), U, G_a)
        X_no_corr[k] = x_no_corr.flatten()
        G_x_no_corr[k] = g_x_no_corr
    # ======================================== #


    # === Smoothing === #
    # We assume that the final position is the same as the starting position
    # Initialisation

    # X_smooth[-1] = np.array([X_0[0], X_0[1], X_0[3]]).flatten()  # we start from the first point
    x_0 = X_simu[-1][0]
    y_0 = X_simu[-1][1]
    v_0 = X_simu[-2][3]
    X_smooth[-1] = np.array([x_0, y_0, v_0]).flatten()  # we start from the last point

    # We can spot that the simulation doesn't end at the same point as the starting point, but we'll assume that it does

    for k in range(N-1, -1, -1):  # we're going backwards in time
        x_up = X_no_corr[k]
        g_x_up = G_x_no_corr[k]

        theta, delta = X_simu[k][2], X_simu[k][4]  # we suppose that we know exactly the angles theta and delta

        J = g_x_up @ A(theta, delta, dt).T @ np.linalg.inv(G_x_no_corr[k+1])

        x_smooth = x_up + J @ (X_smooth[k+1] - X_no_corr[k+1])
        g_x_smooth = g_x_up - J @ (G_x_no_corr[k+1] - G_x_smooth[k+1]) @ J.T

        X_smooth[k] = x_smooth.flatten()
        G_x_smooth[k] = g_x_smooth
    # ================= #

    # === Correction and Prediction with the landmarks === #
    for k in range(1, N + 1):
        line = 0  # index of the line of the matrix d
        theta, delta = X_simu[k-1][2], X_simu[k-1][4]  # we suppose that we know exactly the theta and delta angles

        if k*dt % 10 == 0 and k*dt <= 80:  # we have a measurement every 10 seconds
            # Correction
            y_land = np.array([d[line][1], d[line][2], d[line][3], d[line][4]]).reshape((4, 1))
            line += 1
            G_zeros = np.zeros((3, 3))  # the uncertainty on the measurement is 0
            x_corr, g_x_corr = correction(X_land[k-1].reshape(3, 1), G_zeros, y_land, C_k_land(k, X_land), G_b)
            X_corr[k] = x_corr.flatten()
            G_x_corr[k] = g_x_corr

            # Plot the corrected position with a large cross
            plt.plot(x_corr[0], x_corr[1], marker='o', color='cyan')
            plt.plot([x_corr[0], x_corr[0]], [x_corr[1], x_corr[1] + 1], color='cyan')
            plt.plot([x_corr[0], x_corr[0]], [x_corr[1], x_corr[1] - 1], color='cyan')
            plt.plot([x_corr[0], x_corr[0]+.5], [x_corr[1], x_corr[1]], color='cyan')
            plt.plot([x_corr[0], x_corr[0]-.5], [x_corr[1], x_corr[1]], color='cyan')

            # Smoothing
            if smooth_land == 1:
                X_land[k] = x_land.flatten()  # we need these values for the smoothing
                G_x_land[k] = g_x_land
                for i in range(k-1, -1, -1):  # we're going backwards in time
                    J = G_x_corr[i] @ A(theta, delta, dt).T @ np.linalg.inv(G_x_land[i+1])
                    x_land = X_corr[i] + J @ (x_land - X_pred[i+1])
                    g_x_land = G_x_corr[i] - J @ (G_x_pred[i+1] - g_x_land) @ J.T

            # Prediction
            U = np.array([0, 0, consignes[k-1][0] + 0 * np.random.normal(0, std)]).reshape((3, 1))
            x_land, g_x_land = prediction(x_corr, g_x_corr, A(theta, delta, dt), U, G_a)
            X_pred[k] = x_land.flatten()
            G_x_pred[k] = g_x_land

            X_land[k] = x_land.flatten()
            G_x_land[k] = g_x_land

        else:
            # Correction
            C = np.array([0, 0, 1]).reshape((3, 1))  # we only measure the speed
            # Prediction
            U = np.array([0, 0, consignes[k-1][0] + noise * np.random.normal(0, std)]).reshape((3, 1))
            x_land, g_x_land = prediction(X_land[k-1].reshape(3, 1), G_x_land[k-1], A(theta, delta, dt), U, G_a)

            X_land[k] = x_land.flatten()
            G_x_land[k] = g_x_land
    # ================================ #
    # print("7. 8. Donner les nouvelles positions et matrices de covariances :")
    # print("X_land = ", X_land)
    # print("G_x_land = ", G_x_land)

    # === Correction and Prediction with landmarks and the odometer === #
    G_b = np.identity(5) * np.sqrt(.06)  # measurement noise
    G_b2 = np.identity(1) * np.sqrt(.06)  # measurement noise
    for k in range(1, N + 1):
        line = 0  # index of the line of the matrix d
        theta, delta = X_simu[k-1][2], X_simu[k-1][4]  # we suppose that we know exactly the theta and delta angles
        v = X_simu[k-1][3]  # we suppose that we know the speed with a variance of 0.06

        if k*dt % 10 == 0 and k*dt <= 80:  # we have a measurement every 10 seconds
            # Correction
            y_odo = np.array([d[line][1], d[line][2], d[line][3], d[line][4], v]).reshape((5, 1))
            line += 1
            G_zeros = np.zeros((3, 3))  # the uncertainty on the measurement is 0
            x_corr, g_x_corr = correction(X_odo[k-1].reshape(3, 1), G_zeros, y_odo, C_k_odo(k, X_odo, False), G_b)
            X_corr[k] = x_corr.flatten()
            G_x_corr[k] = g_x_corr

            # # Plot the corrected position with a large cross
            # plt.plot(x_corr[0], x_corr[1], marker='o', color='cyan')
            # plt.plot([x_corr[0], x_corr[0]], [x_corr[1], x_corr[1] + 1], color='cyan')
            # plt.plot([x_corr[0], x_corr[0]], [x_corr[1], x_corr[1] - 1], color='cyan')
            # plt.plot([x_corr[0], x_corr[0]+.5], [x_corr[1], x_corr[1]], color='cyan')
            # plt.plot([x_corr[0], x_corr[0]-.5], [x_corr[1], x_corr[1]], color='cyan')



            # Smoothing
            if smooth_odo == 1:
                X_odo[k] = x_odo.flatten()  # we need these values for the smoothing
                G_x_odo[k] = g_x_odo
                for i in range(k-1, -1, -1):  # we're going backwards in time
                    J = G_x_corr[i] @ A(theta, delta, dt).T @ np.linalg.inv(G_x_odo[i+1])
                    x_odo = X_corr[i] + J @ (x_odo - X_pred[i+1])
                    g_x_odo = G_x_corr[i] - J @ (G_x_pred[i+1] - g_x_odo) @ J.T

            # Prediction
            U = np.array([0, 0, consignes[k-1][0] + 0 * np.random.normal(0, std)]).reshape((3, 1))
            x_odo, g_x_odo = prediction(x_corr, g_x_corr, A(theta, delta, dt), U, G_a)
            X_pred[k] = x_odo.flatten()
            G_x_pred[k] = g_x_odo

            X_odo[k] = x_odo.flatten()
            G_x_odo[k] = g_x_odo

        else:  # if we don't have a measurement
            # Correction
            y_odo = np.array([v]).reshape((1,))  # we only measure the speed
            x_corr, g_x_corr = correction(X_odo[k-1].reshape(3, 1), G_x_odo[k-1], y_odo,
                                          C_k_odo(k, X_odo, True), G_b2)
            X_corr[k] = x_corr.flatten()
            G_x_corr[k] = g_x_corr

            # Prediction
            U = np.array([0, 0, consignes[k-1][0] + noise * np.random.normal(0, std)]).reshape((3, 1))
            x_odo, g_x_odo = prediction(X_odo[k-1].reshape(3, 1), G_x_odo[k-1], A(theta, delta, dt), U, G_a)

            X_odo[k] = x_odo.flatten()
            G_x_odo[k] = g_x_odo
    # ================================================================= #

    # === LIDARS === #
    lidars = loadcsv("lidar.csv")
    if read_lidars == 1:
        # We plot the lidars measurements, to make sure that they are similar to the building's boundaries
        for k in range(len(lidars[0,:])):  # for each instant
            j = int(k*5/dt)  # the corresponding time, given that the measurements are taken every 5 seconds
            for i in range(len(lidars[:,k])):  # for each lidar, we plot the measurement of the wall, given X_simu
                dist = lidars[i][k]
                angle = i * np.pi/4
                xlidar = dist * np.cos(X_simu[j][2] + angle) + X_simu[j][0]
                ylidar = dist * np.sin(X_simu[j][2] + angle) + X_simu[j][1]
                # Display the lidars measurements, which should overlap the building's boundaries
                plt.scatter(xlidar, ylidar, color='magenta', marker='x')
                # Now using light beams, pewpewpewpewpewpewpewpewpewpewpewpewpewpewpewpewpewpew
                xbeam = np.array([xlidar, X_simu[j][0]])
                ybeam = np.array([ylidar, X_simu[j][1]])
                plt.plot(xbeam, ybeam, color='magenta', linestyle='--', linewidth=.5)
    # ===============#


    # === Plot === #
    # Plot the building
    for i in range(len(batiment[0])): # for each wall
        xbuild = np.array([batiment[0][i], batiment[2][i]])
        ybuild = np.array([batiment[1][i], batiment[3][i]])
        plt.plot(xbuild, ybuild, color='black')

    # Plot the landmarks
    plt.scatter(landmarks[:, 0], landmarks[:, 1], marker='2', s=100, color='c', label='Landmarks')

    # Plot the robot
    # Simulated trajectory
    plt.plot(X_simu[:, 0], X_simu[:, 1], color='red', label='Simulated trajectory')
    for i in range(len(X_simu)):
        if plot_pred == 1:
        # Predicted trajectory, no correction
            xpred = X_no_corr[i][0]
            ypred = X_no_corr[i][1]
            plt.plot(xpred, ypred, marker='x', color='blue')
        # Smoothed trajectory
        if plot_smooth == 1:
            xsmooth = X_smooth[i][0]
            ysmooth = X_smooth[i][1]
            plt.plot(xsmooth, ysmooth, marker='x', color='green')
        if plot_land == 1:
        # Corrected trajectory, using landmarks
            xland = X_land[i][0]
            yland = X_land[i][1]
            plt.plot(xland, yland, marker='+', color='purple')
        if plot_odo == 1:
        # Corrected trajectory, using landmarks and the odometer
            xodo = X_odo[i][0]
            yodo = X_odo[i][1]
            plt.plot(xodo, yodo, marker='+', color='orange')
        # Ellipses
        if display_ellipses == 1:
            if i % 3 == 0:  # plot one in every 3 ellipses
                # plot_ellipse(G_x_no_corr[i]*.01, X_no_corr[i], 'blue')
                # plot_ellipse(G_x_smooth[i]*.01, X_smooth[i], 'green')
                plot_ellipse(G_x_land[i]*.1, X_land[i], 'purple')
                plot_ellipse(G_x_odo[i]*.1, X_odo[i], 'orange')
    # Labels
    plt.plot(xbuild[-1], ybuild[-1], color='black', label='Building')  # to avoid having multiple labels

    plt.plot(X_no_corr[-1][0], X_no_corr[-1][1], 'x', color='blue', label='Predicted trajectory')
    plt.plot(X_smooth[-1][0], X_smooth[-1][1], 'x', color='green', label='Smoothed trajectory, using one point')
    plt.plot(X_land[-1][0], X_land[-1][1], '+', color='purple',
             label='Corrected trajectory, using landmarks \n Smoothing = ' + str(smooth_land))
    plt.plot(X_odo[-1][0], X_odo[-1][1], '+', color='orange',
             label='Corrected trajectory, using landmarks and the odometer \n Smoothing = ' + str(smooth_odo))
    # Measurements
    plt.scatter(x_corr[0], x_corr[1], color='cyan', label='Measurements')

    plt.legend()
    plt.title("Kalman 3.2 - BE Noté")
    plt.show()
    # ============ #