""" Kalman 3.2
Ex 3.6 - Localisation by Simulated Annealing
p65
@Author: Agnus Oscar"""

# === Imports === #
import numpy as np
import matplotlib.pyplot as plt


def f(p):
    y = 10000 * np.ones([8, 1]) # y stores the eight distances of the lidars

    for k in p:                     # for each lidar
        for j in range(A.shape[1]): # for each wall
            # Compute the useful vectors
            vect_a_m = np.array([A[0, j] - p[0], A[1, j] - p[1]])
            vect_b_m = np.array([B[0, j] - p[0], B[1, j] - p[1]])
            vect_b_a = np.array([B[0, j] - A[0, j], B[1, j] - A[1, j]])

            angle = p[2] + k * np.pi / 4 # angle of the lidar

            vect_u = np.array([np.cos(angle), np.sin(angle)]) # Unity vector of the lidar’s direction

            # Compute the determinant of the vectors
            det_a_m_u = np.dot(vect_a_m.reshape(1, 2), vect_u.reshape(2, 1))
            det_b_m_u = np.dot(vect_b_m.reshape(1, 2), vect_u.reshape(2, 1))

            det_a_m_b_a = np.dot(vect_a_m.reshape(1, 2), vect_b_a.reshape(2, 1))
            det_u_b_a = np.dot(vect_u.reshape(1, 2), vect_b_a.reshape(2, 1))

            # print("det_a_m_u",k,j," : ",(det_a_m_u * det_b_m_u <= 0), (det_a_m_b_a * det_u_b_a >= 0))

            if ((det_a_m_u * det_b_m_u <= 0) and (det_a_m_b_a * det_u_b_a >= 0)): # Check if the wall intersects with the lidar
                if (det_a_m_u * det_b_m_u != 0): # Check if the wall is parallel to the lidar
                    dist = det_a_m_b_a / det_u_b_a # Compute the distance between the lidar and the wall

            else :
                dist = 10000 # If the wall is parallel to the lidar, the distance is infinite
            y[int(k)] = dist #error

    return y

def draw_room():
    for j in range(A.shape[1]):
        plt.plot(np.array([A[0, j], B[0, j]]), np.array([A[1, j], B[1, j]]), color='blue')

def draw(p, y, col):
    p = p.flatten()
    y = y.flatten()
    for i in np.arange(0, 8):
        plt.plot(p[0] + np.array([0, y[i] * np.cos(p[2] + i * np.pi / 4)]),
                 p[1] + np.array([0, y[i] * np.sin(p[2] + i * np.pi / 4)]), color=col)

# === Parameters === #

A = np.array([[0, 7, 7, 9, 9, 7, 7, 4, 2, 0, 5, 6, 6, 5],
              [0, 0, 2, 2, 4, 4, 7, 7, 5, 5, 2, 2, 3, 3]])
B = np.array([[7, 7, 9, 9, 7, 7, 4, 2, 0, 0, 6, 6, 5, 5],
              [0, 2, 2, 4, 4, 7, 7, 5, 5, 0, 2, 3, 3, 2]])
y = np.array([[6.4], [3.6], [2.3], [2.1], [1.7], [1.6], [3.0], [3.1]])

# ================== #


if __name__ == '__main__':
    # === Plot === #
    fig = plt.figure()

    # === Initialisation === #
    i = 0
    dt = 0.01
    p0 = np.array([[4], [4], [0]])  # Arbitrary initial position
    j0 = np.linalg.norm(y - f(p0))  # Loss function

    # === Simulated Annealing === #
    T = 5  # Initial temperature
    while j0 > 5 or i < 500:
        # === Computation === #
        d = np.random.rand(3, 1) * 0.1  # Random direction
        q = p0 + d*T  # New position
        j0 = np.linalg.norm(y - f(p0))  # Compute new loss function objective
        print(j0)
        if np.linalg.norm(y - f(q)) < j0:  # If the new position is better
            p0 = q  # Update the position

        T = 0.99 * T  # Decrease the temperature
        i += 1

        # === Plot === #
        draw_room()
        draw(p0, y, 'red')
        # plt.plot(np.array([p0[0, 0], q[0, 0]]), np.array([p0[1, 0], q[1, 0]]), color='red')
        # plt.plot(np.array([q[0, 0] + np.cos(q[2, 0]), q[0, 0]]), np.array([[q[1, 0]] + np.sin(q[2, 0]), q[1, 0]]), color='blue')
        plt.xlim(-2, 10)
        plt.ylim(-2, 10)
        plt.pause(dt)

    print(f(np.array([[4], [4], [0]])))











    # yeah, i’m a simp
    # S I M P
    # Squirrels In My Pants
    # S I M P
    # Sauce In My Panini
    # S I M P
    # SOIA Intern Making Pancakes