from exp.exp_stat.experiment import run

nums = 49
delta_t = 0.0001
steps = 10000
calc_para = False
plot_traj = True

if __name__ == '__main__':
    run(nums=nums, delta_t=delta_t, steps=steps, calc_para=calc_para, plot_traj=plot_traj)
