import numpy as np
from common.paraSolver import ParaSolver


class NumericalSolver3D(ParaSolver):
    def __init__(self, s_1, s_2, w_z, list_time):
        A = np.array([[s_1, -w_z, 0],
                      [w_z, s_2, 0],
                      [0, 0, -(s_1 + s_2)]])
        list_A = [A for _ in list_time]
        super().__init__(list_A, list_time)


def test():
    numSolver3d = NumericalSolver3D(s_1=2, s_2=1, w_z=0.1, list_time=np.linspace(0, 6, 6000))
    numSolver3d.calc_geo_para()
    numSolver3d.plot_ratio()
    numSolver3d.plot_angle()


if __name__ == '__main__':
    test()
