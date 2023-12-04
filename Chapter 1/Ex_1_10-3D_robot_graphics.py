""" Kalman 3.2
Ex 1.10 - 3D robot graphics
p
@Author: Agnus Oscar"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm


state_vector = np.array([0, # px
                         0, # py
                         5, # pz
                         0, # v
                         0, # phi
                         0, # theta
                         0]) # psi

# Useful variables
dt = 0.01 # Time step

# Observation vector
u1 = 1 # Linear, tangential acceleration
u2 = 0 # Angular velocity (roll, Phi)
u3 = 0 # Angular velocity (yaw, Psi)


# Initial position of the robot
AUV0_H = np.array([[0, 0, 10, 0, 0, 10, 0, 0],
                   [-1, 1, 0, -1, -0.2, 0, 0.2, 1],
                   [0, 0, 0, 0, 1, 0, 1, 0],
                   [1, 1, 1, 1, 1, 1, 1, 1]])



def transformation(X):
    """Computes the homogeneous matrix of the robot, after a given transformation.
    X : the state vector of the robot"""

    # Set the position before rotation with the state vector
    px,py,pz = X[0],X[1],X[2]
    # Tangential speed
    v = X[3]
    # Euler angles
    phi,theta,Psi = X[4],X[5],X[6]
    # Vertical vector with the 3 positions
    b = np.array([px, py, pz]).reshape((3, 1))

    # Adjacent matrixes, used to compute the Rotation matrix with their exponential
    # Phi
    Ad_i = np.array([[0, 0, 0],
                     [0, 0,-1],
                     [0, 1, 0]])
    # Theta
    Ad_j = np.array([[ 0, 0, 1],
                     [ 0, 0, 0],
                     [-1, 0, 0]])
    # Psi
    Ad_k = np.array([[0,-1, 0],
                     [1, 0, 0],
                     [0, 0, 0]])

    # Exponential matrix, which is the rotation matrix
    R = expm(Ad_i*phi) @ expm(Ad_j*theta) @ expm(Ad_k*Psi)

    # Transformation matrix
    Transf = np.concatenate((R,b),axis=1) # Add the translation vector
    Transf = np.concatenate((Transf,np.array([[0, 0, 0, 1]])),axis=0) # Add the fourth dimension, so we have a square matrix

    # Homogeneous matrix, which represents the pattern matrix after transformation
    H = Transf @ AUV0_H
    return H


def X_dot(X, u1, u2, u3):
    """Computes the derivative of the state vector at a given time"""
    v, phi, theta, psi = X[3], X[4], X[5], X[6]
    dpx =  v * np.cos(theta) * np.cos(psi)
    dpy =  v * np.cos(theta) * np.sin(psi)
    dpz = -v * np.sin(theta)
    dv = u1
    dphi = -0.1 * np.sin(phi) + np.tan(theta) * v * (np.sin(phi) * u2 + np.cos(phi) * u3)
    dtheta = np.cos(phi) * u2 - np.sin(phi) * u3
    dpsi = (np.sin(phi) / np.cos(theta)) * v * u2 + (np.cos(phi) / np.cos(theta)) * v * u3

    return np.array([dpx, dpy, dpz, dv, dphi, dtheta, dpsi])


def plot_frame():
    """Computes the next frame of the robot.
    state_vector : the state vector of the robot at the previous frame"""

    # Clear the figure to avoid the superposition of the plots
    ax.clear()

    # Plot the pattern in blue
    ax.plot(M[0, :], M[1, :], M[2, :], 'b')

    # Plot the shadow in black
    ax.plot(M[0, :], M[1, :], np.zeros(M.shape[1]), 'black')

    # Name the figure
    ax.set_title('Kalman 3.2 - Ex 1.10 - 3D robot graphics')

    # Name and set the axes
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_zlim(-10, 10)


def draw_rotation_vector(X):
    """Draws the rotation vector of the robot"""

    # Compute the derivative of the angles
    dphi   = X_dot(X,u1,u2,u3)[4]
    dtheta = X_dot(X,u1,u2,u3)[5]
    dpsi   = X_dot(X,u1,u2,u3)[6]
    # Get the angles
    phi   = X[4]
    theta = X[5]
    psi   = X[6]

    # Compute the rotation vector
    omega = np.array([[np.cos(theta)*np.cos(psi), -np.sin(psi), 0],
                      [np.cos(theta)*np.sin(psi), np.cos(psi), 0],
                      [-np.sin(theta), 0, 1]]) @ np.array([dphi, dtheta, dpsi])

    # Plot the 3 different components of the rotation vector
    ax.plot([0, omega[0]], [0, omega[1]], [0, omega[2]], color = 'g',label='Rotation vector')


# def plot_pattern(AUV0_H):
# 
#     """Plots the edges of the 3D pattern along with the shadow on the xy plane
#     AUV0_H : the matrix of the pattern at a given time"""
# 
#     # # Place the dots, which is not mandatory
#     # ax.scatter(AUV0_H[:, 0], AUV0_H[:, 1], AUV0_H[:, 2],color='y')
# 
#     # Connect all of the dots altogether
#     for i in range(len(AUV0_H)):
#         for j in range(i + 1, len(AUV0_H)):
#             ax.plot([AUV0_H[i][0], AUV0_H[j][0]], [AUV0_H[i][1], AUV0_H[j][1]], [AUV0_H[i][2], AUV0_H[j][2]], color='r')
#             # Draw the shadow on the xy plane
#             ax.plot([AUV0_H[i][0], AUV0_H[j][0]], [AUV0_H[i][1], AUV0_H[j][1]], [0, 0], color='black')
# 
#     plt.show() # Show the figure

if __name__ == "__main__" :

    # Animation
    fig = plt.figure()    # Create a new figure
    ax = fig.add_subplot(111, projection='3d')    # Create a new 3D axis

    for t in np.arange(0, 10, dt):

        # Compute the next state vector
        state_vector = state_vector + dt * X_dot(state_vector, u1, u2, u3)

        # Compute the transformed pattern
        M = transformation(state_vector)

        plot_frame()
        draw_rotation_vector(state_vector)
        plt.pause(dt)  # Pause the animation for dt seconds, allowing us to see the animation