import numpy as np


# numerical solution
def calc_trans_mat(A, delta_t, F_prec, dim):
    D = np.eye(dim)-(delta_t/2)*A
    E = np.eye(dim)+(delta_t/2)*A
    F = np.dot(np.dot(np.linalg.pinv(D), E), F_prec)
    return F


def calc_geo_parameters(F):
    F_inv = np.linalg.pinv(F)
    G = np.dot(F_inv.transpose(), F_inv)
    parameters = np.linalg.eig(G)
    return parameters


def para_solver(list_A, delta_t, dim):
    steps = len(list_A)
    # calculate tensor of deformation at each time
    trans = np.eye(dim)
    list_F = [trans]
    for i in range(steps):
        trans = calc_trans_mat(list_A[i], delta_t, list_F[i], dim)
        list_F.append(trans)
    # calculate the deformation parameters at each time
    list_value = []
    list_vector = []
    for i in range(len(list_F)):
        (eig_value, eig_vector) = calc_geo_parameters(list_F[i])
        list_value.append(eig_value)
        list_vector.append(eig_vector)
    return list_value, list_vector


A = np.array([[1, 0, 1],
              [0, 2, 0],
              [0, 0, -3]])

list_value, list_vector = para_solver(list_A=[A, A, A], delta_t=0.01, dim=3)
print(list_value)
print(list_vector)