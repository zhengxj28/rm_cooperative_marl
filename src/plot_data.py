import os
import matplotlib.pyplot as plt
import numpy as np

def plot_multi_agent_results(plot_dict, color, label = ''):
    # def plot_multi_agent_results(tester, num_agents):
    """
    Plot the results stored in tester.results for each of the agents.
    """

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

    plt.plot(steps, prc_25, alpha=0)
    plt.plot(steps, prc_50, color=color, label=label)
    plt.plot(steps, prc_75, alpha=0)
    plt.grid()
    plt.fill_between(steps, prc_50, prc_25, color=color, alpha=0.25)
    plt.fill_between(steps, prc_50, prc_75, color=color, alpha=0.25)
    plt.ylabel('Testing Steps to Task Completion', fontsize=15)
    plt.xlabel('Training Steps', fontsize=15)
    plt.locator_params(axis='x', nbins=5)


def plot_data(date, env_name, map_name, task_name, alg_name, color):
    parentDir = os.path.abspath(os.path.join(os.getcwd(), os.pardir, os.pardir))
    data_path = os.path.join(parentDir, 'rm_cooperative_marl_data', date, env_name+map_name)
    file_name = os.path.join(data_path , task_name + alg_name + ".npy")
    plot_dict = np.load(file_name, allow_pickle=True).item()
    plot_multi_agent_results(plot_dict, color, label=task_name+alg_name)


if __name__ == "__main__":
    # plot_data('20211110', "buttons", "cqrm", 'purple')
    # plot_data('20211110',"buttons", "dqprm", 'blue')
    # plot_data('20211110',"buttons", "ihrl", 'red')
    # plot_data('20211110',"buttons", "iql", 'green')

    save_fig = True

    # env_name = 'minecraft'
    # map_name = 'multiA_map_0'
    # task_name = 'task2'
    # plot_data(date, env_name, map_name, task_name, 'cqrm', 'purple')
    # plot_data(date, env_name, map_name, task_name, 'iqrm', 'red')

    # env_name = 'minecraft2'
    # map_name = 'nav_map3'
    # title = '{}_{}'.format(env_name,map_name)
    # plt.title(title)
    # plot_data('20211125', env_name, map_name, 'navigation', 'iqrm', 'purple')
    # plot_data('20211125', env_name, map_name, 'navigation_same_rm', 'iqrm', 'red')
    # plot_data('20211125', env_name, map_name, 'navigation_same_rm', 'cqrm', 'blue')

    # env_name = 'minecraft2'
    # map_name = 'nav_map1'
    # title = '{}_{}'.format(env_name, map_name)
    # plt.title(title)
    # plot_data('20211125', env_name, map_name, 'navigation', 'dqprm', 'green')
    # plot_data('20211125', env_name, map_name, 'navigation_complex_rm', 'iqrm', 'purple')
    # # plot_data(date, env_name, map_name, 'navigation_complex_rm_rs', 'iqrm', 'blue')
    # # plot_data('20211125', env_name, map_name, 'navigation_complex_rm', 'cqrm', 'blue')
    # plot_data('20211126', env_name, map_name, 'navigation_good_p', 'dqprm', 'red')
    # plot_data('20211206', env_name, map_name, 'navigation', 'hie_iqrm', 'yellow')
    # plot_data('20220410', env_name, map_name, 'navigation', 'hie_iqrm2', 'blue')

    # env_name = 'minecraft2'
    # map_name = '3A_map_0'
    # title = '{}_{}'.format(env_name, map_name)+'task5'
    # plt.title(title)
    # plot_data('20211210', env_name, map_name, 'task3', 'dqprm', 'green')
    # plot_data('20211210', env_name, map_name, 'task3', 'iqrm', 'purple')
    # plot_data('20211210', env_name, map_name, 'task3', 'hie_iqrm', 'yellow')

    # plot_data('20211210', env_name, map_name, 'task5', 'dqprm', 'green')
    # plot_data('20211210', env_name, map_name, 'task5', 'iqrm', 'purple')
    # plot_data('20211210', env_name, map_name, 'task5', 'hie_iqrm', 'yellow')
    # plot_data('20220410', env_name, map_name, 'task5', 'hie_iqrm2', 'blue')

    # plot_data('20211210', env_name, map_name, 'task6', 'dqprm', 'green')
    # plot_data('20211210', env_name, map_name, 'task6', 'iqrm', 'purple')
    # plot_data('20211210', env_name, map_name, 'task6', 'hie_iqrm', 'yellow')

    # env_name = 'minecraft2'
    # map_name = 'multiA_map_0'
    # title = '{}_{}'.format(env_name, map_name)+'task2'
    # plt.title(title)
    # plot_data('20211213', env_name, map_name, 'task2', 'dqprm', 'green')
    # plot_data('20211213', env_name, map_name, 'task2', 'iqrm', 'purple')
    # plot_data('20211213', env_name, map_name, 'task2', 'hie_iqrm', 'yellow')

    # env_name = 'minecraft2'
    # map_name = 'nav_map5'
    # title = '{}_{}'.format(env_name,map_name)
    # plt.title(title)
    # # plot_data('20211216', env_name, map_name, 'navigation', 'dqprm', 'green')
    # plot_data('20211216', env_name, map_name, 'navigation_good_p', 'dqprm', 'red')
    # plot_data('20211217', env_name, map_name, 'navigation', 'hie_iqrm', 'yellow')
    # plot_data('20220410', env_name, map_name, 'navigation', 'hie_iqrm2', 'blue')

    # env_name = 'pass_room'
    # map_name = '4button3agent'
    # title = '{}_{}'.format(env_name,map_name)
    # plt.title(title)
    # plot_data('20220412', env_name, map_name, 'pass', 'iqrm', 'green')
    # plot_data('20220412', env_name, map_name, 'pass', 'hie_iqrm2', 'red')
    # plot_data('20220412', env_name, map_name, 'pass_rs', 'iqrm', 'yellow')
    # plot_data('20220412', env_name, map_name, 'pass_rs', 'hie_iqrm2', 'blue')

    env_name = 'pass_room'
    map_name = '4button3agent'
    title = '{}_{}'.format(env_name, map_name)
    plt.title(title)
    plot_data('20220416', env_name, map_name, 'pass2', 'hie_iqrm2', 'green')
    plot_data('20220416', env_name, map_name, 'pass2_rs', 'hie_iqrm2', 'lime')
    plot_data('20220416', env_name, map_name, 'pass3', 'hie_iqrm2', 'red')
    plot_data('20220416', env_name, map_name, 'pass3_rs', 'hie_iqrm2', 'orange')
    plot_data('20220419', env_name, map_name, 'pass4', 'hie_iqrm2', 'blue')
    plot_data('20220419', env_name, map_name, 'pass4_rs', 'hie_iqrm2', 'purple')
    plot_data('20220502', env_name, map_name, 'pass', 'hie_iqrm_3L', 'yellow')
    plot_data('20220505', env_name, map_name, 'pass', 'hie_iqrm_3L', 'black')



    plt.legend(loc='best')
    if save_fig:
        base_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
        fig_path = os.path.join(base_path, 'figures', title+'.pdf')
        plt.savefig(fig_path)
    plt.show()
