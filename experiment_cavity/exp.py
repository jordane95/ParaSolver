import numpy as np
from common.file_io import read_grad
from common.paraSolver import ParaSolver
from common.visualization import simulation_3d

path = 'gradU-0.001.txt'
deltaT, list_A = read_grad(path)
list_time = [deltaT*i for i in range(len(list_A))]
solver = ParaSolver(list_A, list_time)


def visualization(solver):
    list_eig_values, list_eig_vectors = solver.calc_eig_para()
    list_length = np.sqrt(1/(np.array(list_eig_values)+1e-8))
    print('calculation done')
    simulation_3d(list_length, list_eig_vectors)


def data(solver):
    # data for analyse
    solver.calc_geo_para(sort=True, abs=True)
    solver.plot_ratio()
    solver.plot_angle()
    solver.plot_coli(abs=True)
    # solver.plot_ratio_angle()


# visualization(solver)
data(solver)
