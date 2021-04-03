import matplotlib.pyplot as plt
import numpy as np


def _plot_ellipse_3d(lengths, trans_mat, ax):
    (a, b, c) = tuple(lengths)
    theta = np.linspace(0, np.pi, 20)
    phi = np.linspace(0, 2 * np.pi, 20)
    list_x = a * np.outer(np.sin(theta), np.sin(phi))
    list_y = b * np.outer(np.sin(theta), np.cos(phi))
    list_z = c * np.outer(np.cos(theta), np.ones_like(phi))
    list_cor_old = [[x, y, z] for (x, y, z) in zip(list_x, list_y, list_z)]
    list_cor_new = np.array([np.dot(trans_mat, cor) for cor in list_cor_old])
    plt.cla()
    # ax.plot_surface(list_cor_new[:, 0], list_cor_new[:, 1], list_cor_new[:, 2], color='b')
    ax.plot_wireframe(list_cor_new[:, 0], list_cor_new[:, 1], list_cor_new[:, 2], color='r')
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_zlim(-2, 2)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    plt.pause(0.05)


def simulation_3d(list_length, list_vectors):
    print("Plotting...")
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    plt.ion()
    time = 0
    for (lengths, trans_mat) in zip(list_length, list_vectors):
        print("At time %d" % time)
        _plot_ellipse_3d(lengths, trans_mat, ax)
        time += 1
    print("Finished!")
    plt.ioff()
    plt.show()


# plot the trajectory of a particle
def plot_position(list_position):
    print("Plotting trajectory...")
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.scatter(list_position[:, 0], list_position[:, 1], list_position[:, 2],
               c='b', marker='o', s=1)
    ax.set(xlabel='X', ylabel='Y', zlabel='Z')
    print('Finished')
    plt.show()
