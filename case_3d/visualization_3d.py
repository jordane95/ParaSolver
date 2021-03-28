import matplotlib.pyplot as plt
import numpy as np
from case_3d.numerical_3d import NumericalSolver3D


class NumSol3D4Vis(NumericalSolver3D):
    def __init__(self, s_1, s_2, w_z, list_time):
        super().__init__(s_1, s_2, w_z, list_time)
        self.list_length = None

    def calc_length(self):
        self.calc_eig_para()
        self.list_length = np.sqrt(1/np.array(self.list_eig_values))
        return self.list_length


def plot_ellipse_3d_psedo_2d(a, b, c, angle, ax):
    theta = np.linspace(0, np.pi, 100)
    phi = np.linspace(0, 2 * np.pi, 100)
    list_x = a * np.outer(np.sin(theta), np.sin(phi))
    list_y = b * np.outer(np.sin(theta), np.cos(phi))
    list_z = c * np.outer(np.cos(theta), np.ones_like(phi))
    list_cor_xy = [[x, y] for (x, y) in zip(list_x, list_y)]
    rot = np.array([[np.cos(angle), np.sin(angle)],
                    [-np.sin(angle), np.cos(angle)]])
    list_cor_xy_new = np.array([np.dot(rot, cor) for cor in list_cor_xy])
    plt.cla()
    ax.plot_surface(list_cor_xy_new[:, 0], list_cor_xy_new[:, 1], list_z, color='b')
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_zlim(-5, 5)

    plt.pause(0.01)


def simulation_3d(list_length, list_angles):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    plt.ion()
    for (length, angle) in zip(list_length, list_angles):
        (a, b, c) = tuple(length)
        plot_ellipse_3d_psedo_2d(a, b, c, angle, ax)
    print("Finished")
    plt.ioff()
    plt.show()


def test():
    # get data
    solver = NumSol3D4Vis(s_1=2, s_2=1, w_z=0.75, list_time=np.linspace(0, 1, 100))
    list_length = solver.calc_length()
    list_angle = solver.calc_angle()
    print("Calculation Done")
    # plot
    simulation_3d(list_length, list_angle)


if __name__ == '__main__':
    test()