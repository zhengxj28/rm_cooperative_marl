from src.tester.tester import Tester
from src.tester.tester_params import TestingParameters
from src.tester.learning_params import LearningParameters
import os


def minecraft2_config(num_times, task_name, map_name):
    """
    Function setting the experiment parameters and environment.

    Output
    ------
    Tester : tester object
        Object containing the information necessary to run this experiment.
    """

    max_num_agents = 5
    base_file_path = os.path.join(os.path.dirname(__file__), '..', '..')

    joint_rm_file = os.path.join(base_file_path, 'reward_machines', 'minecraft2', map_name, task_name + 'team.txt')
    rm_env_file = os.path.join(base_file_path, 'reward_machines', 'minecraft2', map_name, task_name + 'env.txt')

    local_rm_files = []
    for i in range(max_num_agents):
        local_rm_string = os.path.join(base_file_path, 'reward_machines', 'minecraft2', map_name,
                                       task_name + 'agent%d.txt' % (i + 1))
        local_rm_files.append(local_rm_string)

    local_rm_files_name = []  # remove '.txt'
    for i in range(max_num_agents):
        local_rm_string = os.path.join(base_file_path, 'reward_machines', 'minecraft2', map_name,
                                       task_name + 'agent%d' % (i + 1))
        local_rm_files_name.append(local_rm_string)

    step_unit = 1000

    # configuration of testing params
    testing_params = TestingParameters()
    testing_params.test = True
    testing_params.test_freq = 1 * step_unit
    testing_params.num_steps = step_unit

    # configuration of learning params
    learning_params = LearningParameters()
    learning_params.gamma = 0.9
    learning_params.gamma_controller = 0.9
    learning_params.alpha = 0.1
    learning_params.alpha_controller = 0.1
    learning_params.T = 50
    learning_params.T_controller = 50
    learning_params.initial_epsilon = 0.1  # Set epsilon to zero to turn off epsilon-greedy exploration (only using boltzmann)
    learning_params.max_timesteps_per_task = testing_params.num_steps

    ####### for deep learning ###############
    learning_params.lr = 0.001
    learning_params.hidden_dim = 64
    learning_params.embedding_size = 64

    learning_params.buffer_size = 64
    learning_params.batch_size = 8
    learning_params.target_network_update_freq = 50

    tester = Tester(learning_params, testing_params)
    tester.step_unit = step_unit
    tester.total_steps = 2000 * step_unit
    # tester.total_steps = 10 * step_unit  # for debug only
    tester.min_steps = 1
    tester.max_option_length = 50

    tester.num_times = num_times

    tester.rm_test_file = joint_rm_file
    tester.rm_env_file = rm_env_file
    tester.rm_learning_file_list = local_rm_files
    tester.rm_learning_file_name_list = local_rm_files_name

    # Set the environment settings for the experiment
    env_settings = dict()

    parentDir = os.path.join(os.path.dirname(__file__), '..')
    env_settings['file_map'] = os.path.join(parentDir, 'Environments', 'minecraft2', 'maps', map_name + '.txt')

    env_settings['consider_night'] = False
    env_settings['p'] = 0.98

    tester.env_settings = env_settings

    tester.experiment = 'minecraft2'

    return tester
