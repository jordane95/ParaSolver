import numpy as np
import matplotlib.pyplot as plt

# Documentation
# gradient tensor : A
# transformation tensor : F
# deformation tensor : G
# strain tensor : S = (A+A.T)/2
# The procedure of the solver:
# input: list_grad, list_time
# -> compute transformation tensors, by _calc_list_F(), need list_A
# -> compute deformations tensors, by decompose()
# -> compute eigen parameters, by calc_eig_para(), need list_F
# -> compute geo parameters, by calc_geo_para()
#                                   - calc_ratio(), need list_eig_values
#                                   - calc_angle(), need list_eig_vectors


class ParaSolver:
    def __init__(self, list_A, list_time):
        self.list_A = list_A
        self.list_time = list_time

        # assume equal-time space in list_time
        self.delta_t = list_time[1]-list_time[0]

        self.list_F = None
        self.list_eig_values = None
        self.list_eig_vectors = None
        self.list_ratios = None
        self.list_angles = None

        self.list_S = None
        self.list_eig_values_s = None
        self.list_eig_vectors_s = None
        self.list_inner_product = None

        self.list_omega = None

    @staticmethod
    def _calc_F_next(A, delta_t, F_prec, dim=3):
        D = np.eye(dim) - (delta_t / 2) * A
        E = np.eye(dim) + (delta_t / 2) * A
        F = np.dot(np.dot(np.linalg.pinv(D), E), F_prec)
        return F

    def _calc_list_F(self, dim=3):
        # calculate transformation tensor at each time
        F = np.eye(dim)
        self.list_F = []
        for A in self.list_A:
            F = self._calc_F_next(A, self.delta_t, F, dim)
            self.list_F.append(F)
        return self.list_F

    @staticmethod
    def decompose(F):
        # decompose F to get eigen parameters
        F_inv = np.linalg.pinv(F)
        G = np.dot(F_inv.transpose(), F_inv)
        params = np.linalg.eig(G)
        return params

    def calc_eig_para(self, sort=True):
        # calculate the eig values and vectors at each time
        if self.list_F is None:
            self._calc_list_F()
        self.list_eig_values = []
        self.list_eig_vectors = []
        for F in self.list_F:
            eig_values, eig_vectors = self.decompose(F)
            for i, a in enumerate(eig_values):
                if type(a) == complex:
                    eig_values[i] = a.__abs__()
            if sort:
                sorted_index = eig_values.argsort()
                eig_values = np.array([eig_values[idx] for idx in sorted_index])
                eig_vectors = np.array([eig_vectors[:, idx] for idx in sorted_index]).transpose()
            self.list_eig_values.append(eig_values)
            self.list_eig_vectors.append(eig_vectors)
        return self.list_eig_values, self.list_eig_vectors

    @staticmethod
    def calc_ratio(list_eig_values):
        """
        calculate the ratios out of from the eig values
        :return:
        """
        eps = 1e-7
        list_ratios = []
        for eig_values in list_eig_values:
            ratio_xy = np.sqrt(eig_values[1]/(eig_values[0]+eps))
            ratio_yz = np.sqrt(eig_values[2]/(eig_values[1]+eps))
            ratio_xz = np.sqrt(eig_values[2]/(eig_values[0]+eps))
            ratios = [ratio_xy, ratio_yz, ratio_xz]
            list_ratios.append(ratios)
        return list_ratios

    @staticmethod
    def calc_angle(list_eig_vectors, normalize=False):
        """
        calculate the angles out of from the eig vectors
        :param list_eig_vectors:
        :param normalize:
        :return:
        """
        list_angles = []
        for eig_vectors in list_eig_vectors:
            # inner product -> matrix product
            angles = np.arccos(eig_vectors)
            for i in range(len(angles)):
                for j in range(len(angles[i])):
                    if angles[i][j] > np.pi/2:
                        angles[i][j] -= np.pi
            if normalize:
                angles = np.abs(angles)
            list_angles.append(angles)
        return list_angles

    def calc_geo_para(self, sort=False, normalize=False):
        """

        :param sort:
        :param normalize:
        :return:
        """
        self.calc_eig_para(sort)
        self.list_ratios = self.calc_ratio(self.list_eig_values)
        self.list_angles = self.calc_angle(self.list_eig_vectors, normalize)
        return self.list_ratios, self.list_angles

    def plot_ratio(self, log=False):
        list_ratios = np.array(self.list_ratios)
        list_time = self.list_time
        # if max_time:
        #     list_time = [time for time in self.list_time if time < max_time]
        #     list_ratios = list_ratios[:len(list_time), :]
        if log:
            list_ratios = np.log(list_ratios)
        labels = ['a_1/a_2', 'a_1/a_3', 'a_2/a_3']
        for i in range(3):
            plt.subplot(1, 3, i+1)
            plt.plot(list_time, list_ratios[:, i], label=labels[i])
            plt.xlabel('t')
            plt.ylabel('ratio')
            plt.legend()
        plt.show()
        return None

    def plot_angle(self):
        list_angles = np.array(self.list_angles)
        num = 0
        for i in range(3):
            for j in range(3):
                num += 1
                plt.subplot(3, 3, num)
                plt.plot(self.list_time, list_angles[:, i, j], label='angle(e_'+str(j+1)+', x_'+str(i+1)+')')
                plt.legend()
                plt.xlabel('t')
                plt.ylabel('theta')
        plt.show()
        return None

    def calc_strain_tensor(self):
        self.list_S = [(A+A.T)/2 for A in self.list_A]
        return self.list_S

    def calc_eig_strain(self):
        self.list_eig_values_s = []
        self.list_eig_vectors_s = []
        for S in self.list_S:
            eig_value, eig_vector = np.linalg.eig(S)
            index_sorted = eig_value.argsort()
            eig_value_sorted = np.array([eig_value[i] for i in index_sorted])
            eig_vector_sorted = np.array([eig_vector[:, i] for i in index_sorted]).transpose()
            self.list_eig_values_s.append(eig_value_sorted)
            self.list_eig_vectors_s.append(eig_vector_sorted)
        return self.list_eig_values_s, self.list_eig_vectors_s

    def calc_coli(self, abs=False):
        list_eig_vectors_e = self.list_eig_vectors
        list_eig_vectors_s = self.list_eig_vectors_s
        list_inner_product = [np.dot(e.transpose(), s) for (e, s) in zip(list_eig_vectors_e, list_eig_vectors_s)]
        if abs:
            list_inner_product = np.abs(list_inner_product)
        self.list_inner_product = list_inner_product
        return list_inner_product

    def plot_coli(self, abs=False):
        self.calc_strain_tensor()
        self.calc_eig_strain()
        self.calc_coli(abs)
        list_time = self.list_time
        list_coli = self.list_inner_product
        num = 0
        for i in range(3):
            for j in range(3):
                num += 1
                plt.subplot(3, 3, num)
                plt.plot(list_time, [coli[i, j] for coli in list_coli], label='<e_'+str(i+1)+', s_'+str(j+1)+'>')
                plt.legend()
                plt.xlabel('t')
                plt.ylabel('inner product')
        plt.show()
        return None

    # calculate rot omega
    def calc_omega(self):
        list_W = np.array([(A-A.T)/2 for A in self.list_A])
        self.list_omega = [np.array([W[1, 0], W[2, 0], W[2, 1]]) for W in list_W]
        return self.list_omega


def make_grad_tensor(s_1, s_2, w_z):
    A = np.array([[s_1, -w_z, 0],
                  [w_z, s_2, 0],
                  [0, 0, -(s_1+s_2)]])
    return A


def test():
    A = make_grad_tensor(s_1=2, s_2=1, w_z=0.25)

    A = np.array([[1, -0.1, 5],
                  [0, 2, 0],
                  [0, 0, -3]])
    steps = 5000
    list_A = [A for _ in range(steps)]
    paraSolver = ParaSolver(list_A=list_A, list_time=np.linspace(0, 5, steps))
    paraSolver.calc_geo_para(sort=True, normalize=True)
    paraSolver.plot_ratio()
    paraSolver.plot_angle()
    paraSolver.plot_coli(abs=True)


if __name__ == '__main__':
    test()
