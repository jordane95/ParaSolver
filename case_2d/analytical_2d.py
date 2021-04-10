import numpy as np
import matplotlib.pyplot as plt


# analytical solution of two-dimensional time-independent flow
class AnalyticalSolver2D:
    def __init__(self, s, w, beta, list_time):
        self.s = s
        self.w = w
        self.beta = beta

        self.list_time = list_time

        self.list_ratio = None
        self.list_angle = None

    @staticmethod
    def s_sup_w(s, w, beta, list_time):
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

    @staticmethod
    def s_equ_w(s, w, beta, list_time):
        list_ratio = []
        list_angle = []
        for t in list_time:
            # ratio
            gamma = 1+(s**2+w**2)*t**2
            ratio = gamma + np.sqrt(gamma**2 - 1)
            list_ratio.append(ratio)
            # angle
            top = -w*np.cos(2*beta)*t+np.sin(2*beta)
            below = w*np.sin(2*beta)+np.cos(2*beta)
            theta = np.arctan(top/below) / 2
            list_angle.append(theta)
        return list_ratio, list_angle

    @staticmethod
    def s_inf_w(s, w, beta, list_time):
        tmp = np.sqrt(w**2 - s**2)
        list_ratio = []
        list_angle = []
        list_angle_ = []
        for t in list_time:
            # ratio
            gamma = np.cos(tmp*t)**2 + (w**2+s**2) * np.sin(tmp*t)**2 / (w**2-s**2)
            ratio = gamma + np.sqrt(gamma**2-1)
            list_ratio.append(ratio)
            # angle
            top = tmp*np.sin(2*beta)-w*np.cos(2*beta)*np.tan(tmp*t)
            below = tmp*np.cos(2*beta)+w*np.sin(2*beta)*np.tan(tmp*t)
            tmp_theta = np.arctan(top/below)
            list_angle.append(tmp_theta/2)
            if tmp_theta > 0: list_angle_.append((tmp_theta-np.pi)/2)
            if tmp_theta < 0: list_angle_.append((tmp_theta+np.pi)/2)
            if tmp_theta == 0: list_angle_.append(0)
        return list_ratio, list_angle, list_angle_

    def calc_geo_para(self):
        if self.s**2 > self.w**2:
            self.list_ratio, self.list_angle = self.s_sup_w(self.s, self.w, self.beta, self.list_time)
            self.list_angle_ = None
        elif self.s**2 == self.w**2:
            self.list_ratio, self.list_angle = self.s_equ_w(self.s, self.w, self.beta, self.list_time)
            self.list_angle_ = None
        else:
            self.list_ratio, self.list_angle, self.list_angle_ = self.s_inf_w(self.s, self.w, self.beta, self.list_time)
        return self.list_ratio, self.list_angle, self.list_angle_
