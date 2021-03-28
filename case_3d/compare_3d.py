import numpy as np
import matplotlib.pyplot as plt
from case_3d.numerical_3d import NumericalSolver3D
from case_3d.analytical_3d import AnalyticalSolver3D


class Comparator:
    def __init__(self, s_1, s_2, w_z, list_time):
        self.s_1 = s_1
        self.s_2 = s_2
        self.w_z = w_z
        self.list_time = list_time

    def compare_ratio(self):
        anaSolver = AnalyticalSolver3D(self.s_1, self.s_2, self.w_z, self.list_time)
        numSolver = NumericalSolver3D(self.s_1, self.s_2, self.w_z, self.list_time)
        (list_ratio_xy_ana, list_ratio_xz_ana, list_ratio_yz_ana, list_angle_ana) = anaSolver.calc_para_ana()
        (list_ratio_xy_num, list_ratio_xz_num, list_ratio_yz_num) = numSolver.calc_ratio()
        plt.subplot(1, 3, 1)
        plt.plot(self.list_time, list_ratio_xy_ana, color='r', label='ratio_xy_ana')
        plt.plot(self.list_time, list_ratio_xy_num, color='b', label='ratio_xy_num')
        plt.legend()
        plt.xlabel('t')
        plt.ylabel('a_1/a_2')
        plt.subplot(1, 3, 2)
        plt.plot(self.list_time, list_ratio_xz_ana, color='r', label='ratio_xz_ana')
        plt.plot(self.list_time, list_ratio_xz_num, color='b', label='ratio_xz_num')
        plt.legend()
        plt.xlabel('t')
        plt.ylabel('a_1/a_3')
        plt.subplot(1, 3, 3)
        plt.plot(self.list_time, list_ratio_yz_ana, color='r', label='ratio_yz_ana')
        plt.plot(self.list_time, list_ratio_yz_num, color='b', label='ratio_yz_num')
        plt.legend()
        plt.xlabel('t')
        plt.ylabel('a_2/a_3')
        plt.show()
        return None

    def compare_angle(self):
        anaSolver = AnalyticalSolver3D(self.s_1, self.s_2, self.w_z, self.list_time)
        numSolver = NumericalSolver3D(self.s_1, self.s_2, self.w_z, self.list_time)
        (list_ratio_xy_ana, list_ratio_xz_ana, list_ratio_yz_ana, list_angle_ana) = anaSolver.calc_para_ana()
        list_angle_num = numSolver.calc_angle()
        plt.plot(self.list_time, list_angle_ana, color='r', label='angle_ana')
        plt.plot(self.list_time, list_angle_num, color='b', label='angle_num')
        plt.legend()
        plt.xlabel('t')
        plt.show()
        return None


def test():
    # At most 7s, otherwise, beyond the biggest number that the computer can represent.
    compartor = Comparator(s_1=2, s_2=1, w_z=0.75, list_time=np.linspace(0, 5, 300))
    compartor.compare_ratio()
    compartor.compare_angle()


if __name__ == '__main__':
    test()
