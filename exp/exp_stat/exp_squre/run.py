from exp.exp_stat.experiment import run

nums = 5
delta_t = 0.0008
steps = 4100
calc_para = True
plot_traj = True

if __name__ == '__main__':
    run(nums=nums, delta_t=delta_t, steps=steps, calc_para=calc_para, plot_traj=plot_traj)
