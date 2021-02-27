import numpy as np
import matplotlib.pyplot as plt


# analytical solution of two-dimensional time-independent flow
class AnalyticalSolver:
    def __init__(self, s, w, beta, list_time):
        self.s = s
        self.w = w
        self.beta = beta
        self.list_time = list_time
        self.list_ratio_ana = None
        self.list_angle_ana = None

    # when s^2 > w^2
    def s_sup_w(self):
        list_time = self.list_time
        s = self.s
        w = self.w
        beta = self.beta
        tmp = np.sqrt(s ** 2 - w ** 2)
        list_ratio = []
        list_angle = []
        for t in list_time:
            # analytical solution of ratio
            gamma = np.cosh(tmp * t) ** 2 + (s ** 2 + w ** 2) * np.sinh(tmp * t) ** 2 / (s ** 2 - w ** 2)
            ratio = gamma + np.sqrt(gamma ** 2 - 1)
            list_ratio.append(ratio)
            # analytical solution of angle
            top = tmp * np.sin(2 * beta) - w * np.cos(2 * beta) * np.tanh(tmp * t)
            below = tmp * np.cos(2 * beta) + w * np.sin(2 * beta) * np.tanh(tmp * t)
            theta = np.arctan(top / below) / 2
            list_angle.append(theta)
        return list_ratio, list_angle

    # when s^2 = w^2
    def s_equ_w(self):
        list_time = self.list_time
        s = self.s
        w = self.w
        beta = self.beta
        list_ratio = []
        list_angle = []
        for t in list_time:
            # ratio
            gamma = 1+(s**2+w**2)*t**2
            ratio = gamma + np.sqrt(gamma**2 - 1) ######??????
            list_ratio.append(ratio)
            # angle
            top = -w*np.cos(2*beta)*t+np.sin(2*beta)
            below = w*np.sin(2*beta)+np.cos(2*beta)
            theta = np.arctan(top/below) / 2
            list_angle.append(theta)
        return list_ratio, list_angle

    # when s^2 < w^2
    def s_inf_w(self):
        list_time = self.list_time
        s = self.s
        w = self.w
        beta = self.beta
        tmp = np.sqrt(w**2 - s**2)
        list_ratio = []
        list_angle = []
        for t in list_time:
            # ratio
            gamma = np.cos(tmp*t)**2 + (w**2+s**2) * np.sin(tmp*t)**2 / (w**2-s**2)
            ratio = gamma + np.sqrt(gamma**2-1)
            list_ratio.append(ratio)
            # angle
            top = tmp*np.sin(2*beta)-w*np.cos(2*beta)*np.tan(tmp*t)
            below = tmp*np.cos(2*beta)+w*np.sin(2*beta)*np.tan(tmp*t)
            theta = np.arctan(top/below) / 2
            list_angle.append(theta)
        return list_ratio, list_angle

    def calc_para_ana(self):
        if self.s**2 > self.w**2:
            (list_ratio, list_angle) = self.s_sup_w()
        elif self.s**2 == self.w**2:
            (list_ratio, list_angle) = self.s_equ_w()
        else:
            (list_ratio, list_angle) = self.s_inf_w()
        self.list_ratio_ana = list_ratio
        self.list_angle_ana = list_angle
        return list_ratio, list_angle

    def plot_ratio_ana(self):
        plt.plot(self.list_time, self.list_ratio_ana, label='ratio')
        plt.legend()
        plt.xlabel('time')
        plt.ylabel('a2/a1')
        plt.show()

    def plot_angle_ana(self):
        plt.plot(self.list_time, self.list_angle_ana, label='angle')
        plt.legend()
        plt.xlabel('time')
        plt.ylabel('angle')
        plt.show()
