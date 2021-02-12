import matplotlib.pyplot as plt
import numpy as np
from numerical_solver_3d import ParaSolver


def plot_ellipse_3d(lengths, trans_mat, ax):
    (a, b, c) = tuple(lengths)
    theta = np.linspace(0, np.pi, 100)
    phi = np.linspace(0, 2 * np.pi, 100)
    list_x = a * np.outer(np.sin(theta), np.sin(phi))
    list_y = b * np.outer(np.sin(theta), np.cos(phi))
    list_z = c * np.outer(np.cos(theta), np.ones_like(phi))
    list_cor_old = [[x, y, z] for (x, y, z) in zip(list_x, list_y, list_z)]
    list_cor_new = np.array([np.dot(trans_mat, cor) for cor in list_cor_old])
    plt.cla()
    ax.plot_surface(list_cor_new[:, 0], list_cor_new[:, 1], list_cor_new[:, 2], color='b')
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_zlim(-5, 5)

    plt.pause(0.01)


def simulation_3d(list_length, list_vectors):
    print("Plotting...")
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plt.ion()
    for (lengths, trans_mat) in zip(list_length, list_vectors):
        plot_ellipse_3d(lengths, trans_mat, ax)
    print("Finished!")
    plt.ioff()
    plt.show()


def make_grad_tensor(s_1, s_2, w_z):
    A = np.array([[s_1, -w_z, 0],
                  [w_z, s_2, 0],
                  [0, 0, -(s_1+s_2)]])
    return A


def test():
    # initialization
    A = np.array([[1, -0.1, 5],
                  [0, 2, 0],
                  [0, 0, -3]])
    B = make_grad_tensor(s_1=2, s_2=1, w_z=0.75)
    list_A = [A for t in range(100)]

    # data processing
    solver = ParaSolver(list_A=list_A, list_time=np.linspace(0, 1, 100))
    list_eig_values, list_eig_vectors = solver.calc_eig_para()
    list_length = np.sqrt(1/np.array(list_eig_values))
    print("Calculation Done!")

    # animation
    simulation_3d(list_length, list_eig_vectors)


def make_grad_tensor_2d(s, w, beta):
    sym = np.array([[s*np.cos(2*beta), s*np.sin(2*beta), 0],
                    [s*np.sin(2*beta), -s*np.cos(2*beta), 0],
                    [0, 0, 0]])
    anti_sym = np.array([[0, -w, 0],
                         [w, 0, 0],
                         [0, 0, 0]])
    return sym + anti_sym


def result():
    A = make_grad_tensor_2d(s=.5, w=1, beta=0)
    list_time = np.linspace(0, 5, 500)
    list_A = [A for t in list_time]

    # data processing
    solver = ParaSolver(list_A=list_A, list_time=list_time)
    list_eig_values, list_eig_vectors = solver.calc_eig_para()
    list_length = np.sqrt(1 / np.array(list_eig_values))
    print("Calculation Done!")

    # animation
    simulation_3d(list_length, list_eig_vectors)


result()