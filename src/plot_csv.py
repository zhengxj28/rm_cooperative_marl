import os
import matplotlib.pyplot as plt
import numpy as np
import csv
from matplotlib.ticker import FuncFormatter

colors = {
    'DQPRM': '#EDB732',
    'IQRM': '#C565C7',
    'MODULAR': '#1f77b4',
    'MAHRM': '#FF6347'
}

def read_csv_file(filename):
    data = dict()
    data_list = transpose_csv(filename)
    num_legends = (len(data_list)-1)//3

    for i in range(num_legends):
        legend_name = data_list[3*i+1][0]
        avg = list(map(float, data_list[3*i+1][1:]))
        low = list(map(float, data_list[3*i+2][1:]))
        high = list(map(float, data_list[3*i+3][1:]))
        data[legend_name] = [avg, low, high]
    return data

def transpose_csv(filename):
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        csv_data = list(reader)
    transposed_csv_data = list(map(list, zip(*csv_data)))

    return transposed_csv_data

def smooth_data(data, weight=0.8):
    s_array = [0 for _ in range(len(data))]
    s_array[0] = data[0]
    for i in range(1, len(data)):
        s_array[i] = weight * s_array[i - 1] + (1 - weight) * data[i]
    return s_array

# 绘制曲线
def plot_curves(filename, **kwargs):
    base_path = os.path.join(os.path.dirname(__file__), '..')
    filename = os.path.join(base_path, 'data', filename)
    data = read_csv_file(filename)
    plt.figure(figsize=kwargs['figsize'])
    plt.clf()
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.title(kwargs['title'], fontsize=30)

    plt.grid(color='#FFFFFF', linewidth=2)
    y_max = 0
    for curve_name in kwargs['curve_names']:
        avg, low, high = data[curve_name]
        avg = smooth_data(avg, kwargs['smooth_weight'])
        low = smooth_data(low, kwargs['smooth_weight'])
        high = smooth_data(high, kwargs['smooth_weight'])
        y_max = max(y_max, max(high))
        num_units = len(avg)
        x = [i for i in range(num_units)]
        display_name = "MAHRM（本文）" if curve_name=="MAHRM" else curve_name
        plt.plot(x, avg, linewidth=2, label=display_name, c=colors[curve_name])
        plt.fill_between(x, low, high, color=colors[curve_name], alpha=0.25)


    y_tick_interval = kwargs['y_tick_interval']
    num_y_ticks = int(y_max/y_tick_interval)+1
    plt.xlim(-5, 1999)
    plt.ylim(-0.001, y_max+y_tick_interval)
    fontsize = 15
    plt.xticks(ticks=[0, 499, 999, 1499, 1999], labels=['0', '500k', '1000k', '1500k', '2000k'], fontsize=fontsize)
    plt.yticks(ticks=[y_tick_interval*i for i in range(num_y_ticks)],fontsize=fontsize)
    plt.xlabel('训练步数', fontsize=fontsize)
    plt.ylabel('单步平均奖励', fontsize=fontsize)
    plt.locator_params(axis='x', nbins=5)
    plt.gca().set_facecolor('#EBF0F2')
    if kwargs['show_legend']:
        plt.legend(loc='upper right', fontsize=fontsize)

    if kwargs['savefig']:
        plt.savefig(kwargs['savefig'])
    else:
        plt.show()

def plot_navigation():
    plot_curves(filename='navigation2agents.csv',
                figsize=(9,7),
                smooth_weight=0.5,
                show_legend=False,
                curve_names=["DQPRM", "IQRM", "MODULAR", "MAHRM"],
                y_tick_interval=0.04,
                title='N=2导航任务场景',
                savefig='../figures_cn/navigation2agents.pdf')

    plot_curves(filename='navigation3agents.csv',
                figsize=(9,7),
                smooth_weight=0.8,
                show_legend=False,
                curve_names=["DQPRM", "IQRM", "MODULAR", "MAHRM"],
                y_tick_interval=0.04,
                title='N=3导航任务场景',
                savefig='../figures_cn/navigation3agents.pdf')

    plot_curves(filename='navigation5agents.csv',
                figsize=(9,7),
                smooth_weight=0.8,
                show_legend=True,
                curve_names=["DQPRM", "IQRM", "MODULAR", "MAHRM"],
                y_tick_interval=0.04,
                title='N=5导航任务场景',
                savefig='../figures_cn/navigation5agents.pdf')

def plot_minecraft():
    # plot_curves(filename='navigation2agents.csv',
    #             figsize=(9,7),
    #             smooth_weight=0.5,
    #             show_legend=False,
    #             curve_names=["DQPRM", "IQRM", "MODULAR", "MAHRM"],
    #             title='N=2导航任务场景',
    #             savefig='../figures_cn/navigation2agents.pdf')

    plot_curves(filename='3A_map_0.csv',
                figsize=(9,7),
                smooth_weight=0.9,
                show_legend=False,
                curve_names=["DQPRM", "IQRM", "MODULAR", "MAHRM"],
                y_tick_interval=0.01,
                title='任务二',
                savefig='')

def plot_pass():
    plot_curves(filename='4button3agents.csv',
                figsize=(9,7),
                smooth_weight=0.9,
                show_legend=True,
                curve_names=["DQPRM", "IQRM", "MODULAR", "MAHRM"],
                y_tick_interval=0.005,
                title='任务二',
                savefig='')

if __name__=="__main__":
    # plot_navigation()
    # plot_minecraft()
    plot_pass()
