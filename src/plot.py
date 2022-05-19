import pickle
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

def calculate_data(plot_dict):
    prc_25 = list()
    prc_50 = list()
    prc_75 = list()

    # Buffers for plots
    current_step = list()
    current_25 = list()
    current_50 = list()
    current_75 = list()
    steps = list()

    # plot_dict = tester.results['testing_steps']

    for step in plot_dict.keys():
        if len(current_step) < 10:
            current_25.append(np.percentile(np.array(plot_dict[step]), 25))
            current_50.append(np.percentile(np.array(plot_dict[step]), 50))
            current_75.append(np.percentile(np.array(plot_dict[step]), 75))
            current_step.append(sum(plot_dict[step]) / len(plot_dict[step]))
        else:
            current_step.pop(0)
            current_25.pop(0)
            current_50.pop(0)
            current_75.pop(0)
            current_25.append(np.percentile(np.array(plot_dict[step]), 25))
            current_50.append(np.percentile(np.array(plot_dict[step]), 50))
            current_75.append(np.percentile(np.array(plot_dict[step]), 75))
            current_step.append(sum(plot_dict[step]) / len(plot_dict[step]))

        prc_25.append(sum(current_25) / len(current_25))
        prc_50.append(sum(current_50) / len(current_50))
        prc_75.append(sum(current_75) / len(current_75))
        steps.append(step)
    return prc_25,prc_50,prc_75



colors = {
    'dqprm': '#EDB732',
    'iqrm': '#C565C7',
    'modular': '#1f77b4',
    'hie_iqrm2': '#FF6347',
    'hie_iqrm_3L': '#FF6347',
    # 'sac': '#DA4C4C',
}


info_list = [
    ('minecraft2','nav_map2','navigation', "Navigation 2 Agents"),
    ('minecraft2','nav_map1','navigation', "Navigation 3 Agents"),
    ('minecraft2','nav_map5','navigation', "Navigation 5 Agents"),
]
algorithms = ['dqprm','iqrm','modular','hie_iqrm2']
alg_names = ['DQPRM','IQRM','MODULAR','MAHRM(ours)']
ncols = 3
bbox_to_anchor=(0.5,-0.38)

# info_list = [
#     ('minecraft2','3A_map_0','task3', "MineCraft"),
#     ('pass_room','4button3agent','pass3', "Pass"),
# ]
# algorithms = ['dqprm','iqrm','modular','hie_iqrm2']
# alg_names = ['DQPRM','IQRM','MODULAR','MAHRM(ours)']
# ncols = 2
# bbox_to_anchor=(-0.1,-0.38)


alg_num = 4
fig, ax = plt.subplots(nrows=1, ncols=ncols, sharex=False, sharey=False, gridspec_kw={'height_ratios':[1]}, figsize=(18, 5))
# fig, ax = plt.subplots(nrows=1, ncols=ncols, sharex=False, sharey=False,  figsize=(18, 8))
for i, info in enumerate(info_list):
    env_name, map_name, task, title = info
    col = i % ncols
    row = 0 if i < ncols else 1
    # _, ax = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=False, 
    #     gridspec_kw={'height_ratios':[1, 1]})
    ax_i = ax[col]
    # fig_j = fig[row][col]
    # _, ax = plt.subplots()
    # fig = plt.figure()
    # ax = fig.add_subplot(sub_num)
    for j, alg in enumerate(algorithms):
        print(task, alg)
        ########### load data ############################
        parentDir = os.path.abspath(os.path.join(os.getcwd(), os.pardir, os.pardir))
        data_path = os.path.join(parentDir, 'rm_cooperative_marl_data', '20220515', env_name + map_name)
        file_name = os.path.join(data_path, task + alg + ".npy")
        plot_dict = np.load(file_name, allow_pickle=True).item()
        prc_25, prc_50, prc_75 = calculate_data(plot_dict)
        ########### calculate data ###########
        # mean = np.load('./{}/{}_mean.npy'.format(task, alg))
        # std = np.load('./{}/{}_std.npy'.format(task, alg))
        x = np.arange(len(prc_50))
        ## plot trial mean
        ax_i.plot(x, prc_50, linewidth=2, label=alg_names[j], c=colors[alg])
        ## plot error bars
        ax_i.fill_between(x, prc_25, prc_75, color=colors[alg], alpha=0.25)
    if i == 1:
        ax_i.legend(loc='lower center', bbox_to_anchor=bbox_to_anchor,
        fancybox=True, shadow=True, ncol=alg_num, frameon=False, prop={'size': 20})
    # else:
    #     ax_i.legend_.remove()
    ax_i.set_title(title, fontsize=20)
    ax_i.set_xticks(ticks=[0, 499, 999,1499,1999])
    ax_i.set_xticklabels(['0', '500k', '1000k', '1500k', '2000k'], fontsize=15)
    ax_i.set_yticks(ticks=[0, 200,400,600,800, 1000])
    ax_i.set_yticklabels(['0', '200','400','600','800','1000'], fontsize=15)
    ax_i.set_xlabel('Training Steps', fontdict={'weight': 'normal', 'size': 15})
    ax_i.set_ylabel('Testing Steps', fontdict={'weight': 'normal', 'size': 15})
    ax_i.set_xlim(0, 1999)
    ax_i.set_ylim(0, 1020)
    # if task == 'hopper':
    #     ax_i.set_title('Hopper', fontsize=25)
    #     ax_i.set_xticks(ticks=[0, 19, 39])
    #     ax_i.set_xticklabels(['0', '20k', '40k'], fontsize=15)
    #     ax_i.set_yticks(ticks=[0, 2000, 4000])
    #     ax_i.set_yticklabels(['0', '2000', '4000'], fontsize=15)
    #     ax_i.set_xlabel('Steps', fontdict={'weight':'normal', 'size':15})
    #     ax_i.set_ylabel('Average Return', fontdict={'weight':'normal', 'size':15})
    #     ax_i.set_xlim(0, 39)
    #     ax_i.set_ylim(0, 4200)
    # elif task == 'inverteddoublependulum':
    #     ax_i.set_title('InvertedDoublePendulum', fontsize=25)
    #     ax_i.set_xticks(ticks=[0, 19, 39])
    #     ax_i.set_xticklabels(['0', '5k', '10k'], fontsize=15)
    #     ax_i.set_yticks(ticks=[0, 8000, 16000])
    #     ax_i.set_yticklabels(['0', '8000', '16000'], fontsize=15)
    #     ax_i.set_xlabel('Steps', fontdict={'weight':'normal', 'size':15})
    #     ax_i.set_ylabel('Average Return', fontdict={'weight':'normal', 'size':15})
    #     ax_i.set_xlim(0, 39)
    #     ax_i.set_ylim(0, 18000)
    # elif task == 'walker2d':
    #     ax_i.set_title('Walker2d', fontsize=25)
    #     ax_i.set_xticks(ticks=[0, 98, 197, 295])
    #     ax_i.set_xticklabels(['0', '100k', '200k', '300k'], fontsize=15)
    #     ax_i.set_yticks(ticks=[0, 2500, 5000])
    #     ax_i.set_yticklabels(['0', '2500', '5000'], fontsize=15)
    #     ax_i.set_xlabel('Steps', fontdict={'weight':'normal', 'size':15})
    #     ax_i.set_ylabel('Average Return', fontdict={'weight':'normal', 'size':15})
    #     ax_i.set_xlim(0, 295)
    #     ax_i.set_ylim(0, 5200)
    # elif task == 'halfcheetah':
    #     ax_i.set_title('HalfCheetah', fontsize=25)
    #     ax_i.set_xticks(ticks=[0, 34, 69])
    #     ax_i.set_xticklabels(['0', '35k', '70k'], fontsize=15)
    #     ax_i.set_yticks(ticks=[0, 5000, 10000])
    #     ax_i.set_yticklabels(['0', '5000', '10000'], fontsize=15)
    #     ax_i.set_xlabel('Steps', fontdict={'weight':'normal', 'size':15})
    #     ax_i.set_ylabel('Average Return', fontdict={'weight':'normal', 'size':15})
    #     ax_i.set_xlim(0, 69)
    #     ax_i.set_ylim(0, 11000)
    # elif task == 'humanoid':
    #     ax_i.set_title('Humanoid', fontsize=25)
    #     ax_i.set_xticks(ticks=[0, 39, 79, 119])
    #     ax_i.set_xticklabels(['0', '40k', '80k', '120k'], fontsize=15)
    #     ax_i.set_yticks(ticks=[0, 3000, 6000])
    #     ax_i.set_yticklabels(['0', '3000', '6000'], fontsize=15)
    #     ax_i.set_xlabel('Steps', fontdict={'weight':'normal', 'size':15})
    #     ax_i.set_ylabel('Average Return', fontdict={'weight':'normal', 'size':15})
    #     ax_i.set_xlim(0, 119)
    #     ax_i.set_ylim(0, 6400)
    # elif task == 'ant':
    #     ax_i.set_title('Ant', fontsize=25)
    #     ax_i.set_xticks(ticks=[0, 49, 99, 149])
    #     ax_i.set_xticklabels(['0', '50k', '100k', '150k'], fontsize=15)
    #     ax_i.set_yticks(ticks=[0, 2500, 5000])
    #     ax_i.set_yticklabels(['0', '2500', '5000'], fontsize=15)
    #     ax_i.set_xlabel('Steps', fontdict={'weight':'normal', 'size':15})
    #     ax_i.set_ylabel('Average Return', fontdict={'weight':'normal', 'size':15})
    #     ax_i.set_xlim(0, 149)
    #     ax_i.set_ylim(0, 5500)
    ax_i.spines['top'].set_visible(False)
    ax_i.spines['right'].set_visible(False)
    ax_i.spines['bottom'].set_visible(False)
    ax_i.spines['left'].set_visible(False)
    ax_i.set_facecolor('#EBF0F2')
    ax_i.grid(color='#FFFFFF', linewidth=2)
    ax_i.tick_params(bottom=False, left=False)

plt.subplots_adjust(left=0.06,
                    bottom=0.25,
                    right=0.98, 
                    top=0.90,
                    wspace=0.25,
                    hspace=0.36)
# fig.tight_layout()
# plt.savefig('mujoco_results.pdf')
plt.show()