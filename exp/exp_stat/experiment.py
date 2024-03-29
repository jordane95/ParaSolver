from common.statSolver import StatSolver
import numpy as np


def run(nums, delta_t, steps, calc_para=False, plot_traj=False, save_traj=False, shape=None, save_path="./img/position_"):
    stat_solver = StatSolver(path="./data/", num=np.arange(nums), delta_t=delta_t, steps=steps)
    if calc_para:
        stat_solver.solve()
        stat_solver.calc_avg()
        stat_solver.plot_ratio_avg()
        stat_solver.plot_angle_avg()
        stat_solver.plot_coli_avg()
    if plot_traj:
        stat_solver.plot_trajectory(shape=shape)
    if save_traj:
        stat_solver.save_trajectory(shape=shape, save_path=save_path)


if __name__ == '__main__':
    run()
