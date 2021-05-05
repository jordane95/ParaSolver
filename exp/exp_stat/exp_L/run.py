from common.statSolver import StatSolver


def run():
    stat_solver = StatSolver(path="./data/", num=49, delta_t=0.0001, steps=10000)
    stat_solver.solve()
    stat_solver.calc_avg()
    stat_solver.plot_ratio_avg()
    stat_solver.plot_angle_avg()
    stat_solver.plot_coli_avg()

    stat_solver.save_trajectory()


if __name__ == '__main__':
    run()
