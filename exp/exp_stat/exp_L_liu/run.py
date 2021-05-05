from exp.exp_stat.experiment import run

nums = 10
delta_t = 0.001
steps = 2000
calc_para = True
plot_traj = True

if __name__ == '__main__':
    run(nums=nums, delta_t=delta_t, steps=steps, calc_para=calc_para, plot_traj=plot_traj)
