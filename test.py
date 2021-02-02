import matplotlib.pyplot as plt
import numpy as np


def plot_ellipse_2d(a, b, angle):
    t = np.linspace(0, 2 * np.pi, 1000)
    list_x = a * np.cos(t)
    list_y = b * np.sin(t)
    cor_old = [[x, y] for (x, y) in zip(list_x, list_y)]
    rot = np.array([[np.cos(angle), np.sin(angle)],
                    [-np.sin(angle), np.cos(angle)]])

    list_cor_new = np.array([np.dot(rot, cor) for cor in cor_old])

    plt.plot(list_cor_new[:, 0], list_cor_new[:, 1])
    plt.show()


a_1 = 2
a_2 = 3
theta = 1


def plot_ellipse_3d_psedo_2d(a, b, c, angle):
    theta = np.linspace(0, np.pi, 100)
    phi = np.linspace(0, 2*np.pi, 100)
    list_x = a*np.outer(np.sin(theta), np.sin(phi))
    list_y = b*np.outer(np.sin(theta), np.cos(phi))
    list_z = c*np.outer(np.cos(theta), np.ones_like(phi))
    list_cor_xy = [[x, y] for (x, y) in zip(list_x, list_y)]
    rot = np.array([[np.cos(angle), np.sin(angle)],
                    [-np.sin(angle), np.cos(angle)]])
    list_cor_xy_new = np.array([np.dot(rot, cor) for cor in list_cor_xy])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(list_cor_xy_new[:, 0], list_cor_xy_new[:, 1], list_z)
    plt.show()


plot_ellipse_3d_psedo_2d(1, 2, 3, 1)
