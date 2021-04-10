import numpy as np


class AnalyticalSolver3D:
    def __init__(self, s_1, s_2, w_z, list_time):
        self.s_1 = s_1
        self.s_2 = s_2
        self.w_z = w_z
        self.list_time = list_time

        self.list_ratios = None
        self.list_angle = None

    @staticmethod
    def delta_sup_zero(s_1, s_2, w_z, list_time):
        tmp = (s_1-s_2)**2
        b = np.sqrt(np.abs(tmp-4*w_z**2)) / 2
        list_ratios = []
        list_angle = []
        for t in list_time:
            # calculate ratio
            p = 2*(np.cosh(b*t)**2 + ((tmp+4*w_z**2) * np.sinh(b*t)**2 / (tmp-4*w_z**2)))
            q = np.sqrt(4 * tmp / b**2 * np.sinh(b*t)**2 * (np.cosh(b*t)**2 + w_z**2 / b**2 * np.sinh(b*t)**2))
            # print('p=', p, 'q=', q, 'p-q=', p-q)
            ratio_xy = np.sqrt((p+q)/(p-q))
            ratio_xz = np.sqrt(2*np.exp(3*(s_1+s_2)*t)/(p-q))
            ratio_yz = np.sqrt(2*np.exp(3*(s_1+s_2)*t)/(p+q))
            list_ratios.append([ratio_xy, ratio_xz, ratio_yz])
            # calculate angle
            tan = -w_z*np.tanh(b*t)/b
            theta = np.arctan(tan)/2
            list_angle.append(theta)
        return list_ratios, list_angle

    @staticmethod
    def delta_equ_zero(s_1, s_2, w_z, list_time):
        list_ratios = []
        list_angle = []
        for t in list_time:
            # calculate ratio
            det = 1 - ((s_1-s_2)*t/2)**2 + (w_z*t)**2
            p = 2*(1+((s_1-s_2)*t/2)**2+(w_z*t)**2)/det
            q = 2*np.abs(s_1-s_2)*t*np.sqrt(1+(w_z*t)**2)/det
            ratio_xy = np.sqrt((p+q)/(p-q))
            ratio_xz = np.sqrt(2 * np.exp(3*(s_1+s_2)*t) * det / (p-q))
            ratio_yz = np.sqrt(2 * np.exp(3*(s_1+s_2)*t) * det / (p+q))
            list_ratios.append([ratio_xy, ratio_xz, ratio_yz])
            # compute angle
            theta = np.arctan(-w_z*t)/2
            list_angle.append(theta)
        return list_ratios, list_angle

    # when delta < 0
    @staticmethod
    def delta_inf_zero(s_1, s_2, w_z, list_time):
        tmp = (s_1-s_2)**2
        b = np.sqrt(np.abs(tmp - 4 * w_z ** 2)) / 2
        list_ratios = []
        list_angle = []
        for t in list_time:
            # calculate ratio
            p = 2*(np.cos(b*t)**2 - (tmp+4*w_z**2) * np.sin(b*t)**2 / (tmp-4*w_z**2))
            q = np.sqrt(4 * tmp * np.sin(b*t)**2 * (np.cos(b*t)**2 + w_z**2 * np.sin(b*t)**2 / b**2))/b
            ratio_xy = np.sqrt((p+q)/(p-q))
            ratio_xz = np.sqrt(2*np.exp(3*(s_1+s_2)*t)/(p-q))
            ratio_yz = np.sqrt(2*np.exp(3*(s_1+s_2)*t)/(p+q))
            list_ratios.append([ratio_xy, ratio_xz, ratio_yz])
            # calculate angle
            theta = np.arctan(-w_z/b*np.tan(b*t))/2
            list_angle.append(theta)
        return list_ratios, list_angle

    def calc_geo_para(self):
        delta = (self.s_1-self.s_2)**2-4*self.w_z**2
        if delta > 0:
            self.list_ratios, self.list_angle = self.delta_sup_zero(self.s_1, self.s_2, self.w_z, self.list_time)
        elif delta == 0:
            self.list_ratios, self.list_angle = self.delta_equ_zero(self.s_1, self.s_2, self.w_z, self.list_time)
        else:
            self.list_ratios, self.list_angle = self.delta_inf_zero(self.s_1, self.s_2, self.w_z, self.list_time)
        return self.list_ratios, self.list_angle
