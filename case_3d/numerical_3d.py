import numpy as np
import matplotlib.pyplot as plt
from common.paraSolver import ParaSolver


class NumericalSolver3D(ParaSolver):
    def __init__(self, s_1, s_2, w_z, list_time):
        A = np.array([[s_1, -w_z, 0],
                      [w_z, s_2, 0],
                      [0, 0, -(s_1 + s_2)]])
        list_A = [A for _ in list_time]
        super().__init__(list_A, list_time)

    @staticmethod
    def decompose(E, sort=True):
        eig_values, eig_vectors = np.linalg.eig(E)
        if sort:
            sorted_index = eig_values.argsort()
            eig_values = np.array([eig_values[idx] for idx in sorted_index])
        return eig_values, eig_vectors


class DummySolver(ParaSolver):
    def __init__(self, s_1, s_2, w_z, list_time):
        A = np.array([[s_1, -w_z, 0],
                      [w_z, s_2, 0],
                      [0, 0, -(s_1 + s_2)]])
        list_A = [A for _ in list_time]
        super().__init__(list_A, list_time)

    def plot_ratio_right(self):
        list_ratios = self.list_ratios[:, 0]
        plt.plot(self.list_time, np.log(list_ratios), label='ratio_num')
        plt.xlabel('time')
        plt.ylabel('log(a_1/a_2)')
        plt.legend()
        plt.show()


def test():
    # numSolver3d = NumericalSolver3D(s_1=2, s_2=1, w_z=0.75, list_time=np.linspace(0, 10, 1000))
    # numSolver3d.calc_geo_para()
    # numSolver3d.plot_angle()
    dummySolver = DummySolver(s_1=2, s_2=1, w_z=0.75, list_time=np.linspace(0, 30, 3000))
    dummySolver.calc_geo_para()
    dummySolver.plot_ratio_right()
    # dummySolver.plot_angle()


if __name__ == '__main__':
    test()
