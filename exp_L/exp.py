import numpy as np
from common.file_io import read_grad, read_position
from common.paraSolver import ParaSolver
from common.visualization import simulation_3d, plot_position


# solve deformation of particle in the flow
path = 'gradU.txt'
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
    solver.calc_geo_para(sort=True, normalize=True)
    solver.plot_ratio(log=True)
    solver.plot_angle()
    solver.plot_coli(abs=True)


# plot trajectory of the particle in the flow
position_file = 'Utr.txt'
delta, list_position, list_velocity = read_position(position_file)


# run the program
# visualization(solver=solver)
# data(solver=solver)
plot_trajectory(list_position)

