import numpy as np
import matplotlib.pyplot as plt


class ParaSolver:
    def __init__(self, list_A, list_time):
        self.list_A = list_A
        self.list_time = list_time
        self.list_F = None
        self.list_eig_values = None
        self.list_eig_vectors = None
        self.list_ratio_xy = None
        self.list_ratio_yz = None
        self.list_ratio_xz = None
        self.list_ratios = None
        self.list_angles = None

        self.list_S = None
        self.list_eig_values_s = None
        self.list_eig_vectors_s = None
        self.list_inner_product = None

    def reset(self, list_A, list_time):
        self.list_A = list_A
        self.list_time = list_time

    def calc_trans_mat(self, A, delta_t, F_prec, dim):
        D = np.eye(dim) - (delta_t / 2) * A
        E = np.eye(dim) + (delta_t / 2) * A
        F = np.dot(np.dot(np.linalg.pinv(D), E), F_prec)
        return F

    def calc_F(self):
        # calculate the transformation tensor at each time
        dim = 3
        list_A = self.list_A
        delta_t = self.list_time[1]-self.list_time[0]
        steps = len(list_A)
        trans = np.eye(dim)
        list_F = []
        for A in list_A:
            trans = self.calc_trans_mat(A, delta_t, trans, dim)
            list_F.append(trans)
        self.list_F = list_F
        return list_F

    def decompose(self, F):
        # decompose of F according to the algorithm
        F_inv = np.linalg.pinv(F)
        G = np.dot(F_inv.transpose(), F_inv)
        parameters = np.linalg.eig(G)
        return parameters

    def calc_eig_para(self):
        # calculate the eig values and vectors at each time
        self.calc_F()
        list_eig_values = []
        list_eig_vectors = []
        for F in self.list_F:
            (eig_value, eig_vector) = self.decompose(F)
            sorted_index = eig_value.argsort()
            eig_value_sorted = np.array([eig_value[i] for i in sorted_index])
            eig_vector_sorted = np.array([eig_vector[:,i] for i in sorted_index]).transpose()
            list_eig_values.append(eig_value_sorted)
            list_eig_vectors.append(eig_vector_sorted)
        self.list_eig_values = list_eig_values
        self.list_eig_vectors = list_eig_vectors
        return list_eig_values, list_eig_vectors

    def calc_ratio(self):
        self.calc_eig_para()
        # calculate the ratios out of from the eig values
        list_eig_values = self.list_eig_values
        list_eig_vectors = self.list_eig_vectors
        # ratio
        list_ratio_xy = []
        list_ratio_yz = []
        list_ratio_xz = []
        list_ratios = []
        for eig_values in list_eig_values:
            ratio_xy = np.sqrt(eig_values[1]/eig_values[0])
            ratio_yz = np.sqrt(eig_values[2]/eig_values[1])
            ratio_xz = np.sqrt(eig_values[2]/eig_values[0])
            list_ratio_xy.append(ratio_xy)
            list_ratio_yz.append(ratio_yz)
            list_ratio_xz.append(ratio_xz)
            ratios = [ratio_xy, ratio_yz, ratio_xz]
            list_ratios.append(ratios)
        self.list_ratio_xy = list_ratio_xy
        self.list_ratio_yz = list_ratio_yz
        self.list_ratio_xz = list_ratio_xz
        self.list_ratios = list_ratios
        return list_ratio_xy, list_ratio_yz, list_ratio_xz

    def calc_angle(self):
        self.calc_eig_para()
        # calculate the angles out of from the eig vectors
        list_eig_vectors = self.list_eig_vectors
        # angle
        list_angles = []
        for eig_vectors in list_eig_vectors:
            # inner product -> matrix product
            angles = np.arccos(eig_vectors)

            for i in range(len(angles)):
                for j in range(len(angles[i])):
                    if angles[i][j] > np.pi/2:
                        angles[i][j] -= np.pi

            list_angles.append(angles)
        self.list_angles = list_angles
        return list_angles

    def calc_geo_para(self):
        (self.list_ratio_xy, self.list_ratio_yz, self.list_ratio_xz) = self.calc_ratio()
        self.list_angles = self.calc_angle()
        return None

    def plot_ratio(self):
        self.calc_geo_para()
        list_ratios = self.list_ratios
        list_ratios= np.array(list_ratios)
        list_time = self.list_time
        plt.subplot(1, 3, 1)
        plt.plot(list_time, list_ratios[:, 0], label='ratio_xy')
        plt.xlabel('t')
        plt.ylabel('a_1/a_2')
        plt.legend()
        plt.subplot(1, 3, 2)
        plt.plot(list_time, list_ratios[:, 1], label='ratio_yz')
        plt.xlabel('t')
        plt.ylabel('a_2/a_3')
        plt.legend()
        plt.subplot(1, 3, 3)
        plt.plot(list_time, list_ratios[:, 2], label='ratio_xz')
        plt.xlabel('t')
        plt.ylabel('a_1/a_3')
        plt.legend()
        plt.show()
        return None

    def plot_angle(self):
        self.calc_geo_para()
        list_angles = self.list_angles
        list_angles = np.array(list_angles)
        list_time = self.list_time
        # self.para_check()
        num = 0
        for i in range(3):
            for j in range(3):
                num += 1
                plt.subplot(3, 3, num)
                plt.plot(list_time, list_angles[:, i, j], label='angle')
                # print(len(list_angles[:, i, j]))
                plt.legend()
                plt.xlabel('t')
                plt.ylabel('angle')
        plt.show()
        return None

    def para_check(self):
        for item in self.list_angles:
            print(item)

    # colinearity
    def calc_strain_tensor(self):
        list_S = [(A+A.T)/2 for A in self.list_A]
        self.list_S = list_S
        return list_S

    def calc_eig_strain(self):
        self.list_eig_values_s = []
        self.list_eig_vectors_s = []
        for S in self.list_S:
            (eig_value, eig_vector) = np.linalg.eig(S)
            index_sorted = eig_value.argsort()
            eig_value_sorted = np.array([eig_value[i] for i in index_sorted])
            eig_vector_sorted = np.array([eig_vector[:, i] for i in index_sorted]).transpose()
            self.list_eig_values_s.append(eig_value_sorted)
            self.list_eig_vectors_s.append(eig_vector_sorted)
        return self.list_eig_values_s, self.list_eig_vectors_s

    def calc_coli(self):
        list_eig_vectors_e = self.list_eig_vectors
        list_eig_vectors_s = self.list_eig_vectors_s
        list_inner_product = []
        for (e, s) in zip(list_eig_vectors_e, list_eig_vectors_s):
            inner_product = e.transpose()*s
            list_inner_product.append(inner_product)
        self.list_inner_product = list_inner_product
        return list_inner_product

    def plot_coli(self):
        self.calc_strain_tensor()
        self.calc_eig_strain()
        self.calc_coli()
        list_time = self.list_time
        list_coli = self.list_inner_product
        num = 0
        for i in range(3):
            for j in range(3):
                num += 1
                plt.subplot(3, 3, num)
                plt.plot(list_time, [coli[i, j] for coli in list_coli], label='<e*s>')
                plt.legend()
                plt.xlabel('t')
                plt.ylabel('<e*s>')
        plt.show()
        return None


def make_grad_tensor(s_1, s_2, w_z):
    A = np.array([[s_1, -w_z, 0],
                  [w_z, s_2, 0],
                  [0, 0, -(s_1+s_2)]])
    return A


A = make_grad_tensor(s_1=2, s_2=1, w_z=0.25)

B = np.array([[2, -0.1, 0],
              [0.1, 2, -0.1],
              [0.1, 0, -4]])

list_A = []
for i in range(5000):
    list_A.append(A)

paraSolver = ParaSolver(list_A=list_A, list_time=np.linspace(0, 5, 5000))
paraSolver.plot_ratio()
paraSolver.plot_angle()
paraSolver.plot_coli()






