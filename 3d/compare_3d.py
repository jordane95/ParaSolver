import numpy as np
import matplotlib.pyplot as plt
import numerical_3d as num
import analytical_3d as ana


class Comparator:
    def __init__(self, s_1, s_2, w_z, list_time):
        self.s_1 = s_1
        self.s_2 = s_2
        self.w_z = w_z
        self.list_time = list_time

    def plot_compare_ratio(self):
        anaSolver = ana.AnalyticalSolver3D(self.s_1, self.s_2, self.w_z, self.list_time)
        numSolver = num.NumericalSolver3D(self.s_1, self.s_2, self.w_z, self.list_time)
        (list_ratio_xy_ana, list_ratio_yz_ana, list_ratio_xz_ana, list_angle_ana) = anaSolver.calc_para_ana()
        (list_ratio_xy_num, list_ratio_yz_num, list_ratio_xz_num) = numSolver.calc_ratio()
        plt.subplot(1, 3, 1)
        plt.plot(self.list_time, list_ratio_xy_ana, color='r', label='ratio_xy_ana')
        plt.plot(self.list_time, list_ratio_xy_num, color='b', label='ratio_xy_num')
        plt.legend()
        plt.subplot(1, 3, 2)
        plt.plot(self.list_time, list_ratio_yz_ana, color='r', label='ratio_yz_ana')
        plt.plot(self.list_time, list_ratio_yz_num, color='b', label='ratio_yz_num')
        plt.legend()
        plt.subplot(1, 3, 3)
        plt.plot(self.list_time, list_ratio_xz_ana, color='r', label='ratio_xz_ana')
        plt.plot(self.list_time, list_ratio_xz_num, color='b', label='ratio_xz_num')
        plt.legend()
        plt.show()
        return None

    def plot_compare_angle(self):
        anaSolver = ana.AnalyticalSolver3D(self.s_1, self.s_2, self.w_z, self.list_time)
        numSolver = num.NumericalSolver3D(self.s_1, self.s_2, self.w_z, self.list_time)
        (list_ratio_xy_ana, list_ratio_yz_ana, list_ratio_xz_ana, list_angle_ana) = anaSolver.calc_para_ana()
        list_angle_num = numSolver.calc_angle()
        plt.plot(self.list_time, list_angle_ana, color='r', label='angle_ana')
        plt.plot(self.list_time, list_angle_num, color='b', label='angle_num')
        plt.legend()
        plt.show()
        return None


compartor = Comparator(s_1=2, s_2=1, w_z=0.75, list_time=np.linspace(0, 5, 500))
compartor.plot_compare_ratio()
compartor.plot_compare_angle()