from common.fileIO import read_grad, get_position
from common.paraSolver import ParaSolver
from common.visualization import simulation_3d, plot_position


def run():
    # calculation
    path = 'gradU.txt'
    delta_t, list_A = read_grad(path)
    list_time = [delta_t * i for i in range(len(list_A))]
    solver = ParaSolver(list_A, list_time)

    # position
    filename = 'Utr.txt'
    delta_t, list_position = get_position(filename)
    plot_position(list_position, delta=0.0005, max_time=2.9, shape='l')

    # visualization
    # list_eig_values, list_eig_vectors = solver.calc_eig_para(sort=True)
    # list_length = np.sqrt(1 / (np.array(list_eig_values) + 1e-8))
    # simulation_3d(list_length, list_eig_vectors)

    # data
    solver.calc_geo_para(normalize=True)
    solver.calc_coli(normalize=True)
    max_time = 3
    solver.plot_ratio(log=False, max_time=3)
    solver.plot_angle(max_time)
    solver.plot_coli(max_time)


if __name__ == '__main__':
    run()
