from src.tester.tester import Tester
from src.tester.tester_params import TestingParameters
from src.tester.learning_params import LearningParameters
import os


def minecraft_config(num_times, task_name):
    """
    Function setting the experiment parameters and environment.

    Output
    ------
    Tester : tester object
        Object containing the information necessary to run this experiment.
    """

    max_num_agents = 5
    base_file_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir))

    joint_rm_file = os.path.join(base_file_path, 'reward_machines', 'minecraft', task_name+'team.txt')

    local_rm_files = []
    for i in range(max_num_agents):
        local_rm_string = os.path.join(base_file_path, 'reward_machines', 'minecraft',
                                       task_name+'agent%d.txt'%(i + 1))
        local_rm_files.append(local_rm_string)

    step_unit = 1000

    # configuration of testing params
    testing_params = TestingParameters()
    testing_params.test = True
    testing_params.test_freq = 1 * step_unit
    testing_params.num_steps = step_unit

    # configuration of learning params
    learning_params = LearningParameters()
    learning_params.gamma = 0.9  # 0.9
    learning_params.alpha = 0.1
    learning_params.T = 50
    learning_params.initial_epsilon = 0.1  # Set epsilon to zero to turn off epsilon-greedy exploration (only using boltzmann)
    learning_params.max_timesteps_per_task = testing_params.num_steps

    tester = Tester(learning_params, testing_params)
    tester.step_unit = step_unit
    tester.total_steps = 500 * step_unit
    # tester.total_steps = 10 * step_unit
    tester.min_steps = 1

    tester.num_times = num_times

    tester.rm_test_file = joint_rm_file
    tester.rm_learning_file_list = local_rm_files

    # Set the environment settings for the experiment
    env_settings = dict()

    parentDir = os.path.abspath(os.path.join(os.getcwd()))
    env_settings['file_map'] = os.path.join(parentDir, 'Environments', 'minecraft', 'maps', r'multiA_map_0.txt')
    env_settings['consider_night'] = False
    env_settings['p'] = 0.98

    tester.env_settings = env_settings

    tester.experiment = 'minecraft'

    return tester
