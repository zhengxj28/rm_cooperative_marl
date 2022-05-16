from datetime import datetime
import os, time, sys, random
sys.path.append("..")
sys.path.append("../..")
import numpy as np

show_trajectory = False

def print_params(tester, alg_name):
    learning_params = tester.learning_params
    print('epsilon:', tester.learning_params.initial_epsilon)
    print('decay_factor gamma:', tester.learning_params.gamma)
    print('learning_rate alpha:', tester.learning_params.alpha)
    print('temperature T:', tester.learning_params.T)

    ##### for HRL methods ##########
    if alg_name in ['hie_iqrm','hie_iqrm2','hie_iqrm_3L','mul_hie_iqrm','modular']:
        print("decay_factor of controller gamma_controller:", tester.learning_params.gamma_controller)
        print("temperature of controller T_controller:", tester.learning_params.T_controller)
        print("max_option_length", tester.max_option_length)

    #### for deep learning #########
    if alg_name in ['modular']:
        print("lr of network:", learning_params.lr)
        print("hidden_dim:", learning_params.hidden_dim)
        print("embedding_size:", learning_params.embedding_size)
        print("buffer_size", learning_params.buffer_size)
        print("batch_size", learning_params.batch_size)
        print("target_network_update_freq:", learning_params.target_network_update_freq)

if __name__ == "__main__":
    random.seed(0)
    start_time = time.time()
    independent_trail_times = 10  # Number of separate trials to run the algorithm for
    # num_agents = 3  # This will be automatically set to 3 for buttons experiment (max 10)

    # env_name='buttons'
    # env_name = 'minecraft'
    env_name = 'minecraft2'

    map_name = '3A_map_0'  # 3 agents, traditional minecraft map
    task_name = 'task3'
    # task_name = 'task4'
    # task_name = 'task5'
    # task_name = 'task6'

    # map_name = 'multiA_map_0'  # 2 agents, traditional minecraft map
    # task_name = 'task1'
    # task_name = 'task2'

    # map_name = 'nav_map0'  # 3 agents navigate to a,b,c, no misdirection
    # map_name = 'nav_map1'  # 3 agents navigate to a,b,c, placed with misdirection
    # map_name = 'nav_map2'  # 2 agents navigate to a,b, placed with misdirection
    # map_name = 'nav_map3'  # 2 agents navigate to a,b, no misdirection
    # map_name = 'nav_map5'  # 5 agents navigate to a,b,c,d,e, placed with misdirection

    # task_name = 'navigation'  # simple team rm
    # task_name = 'navigation_complex_rm'  # complex team rm, no rm projection
    # task_name = 'navigation_complex_rs'  # complex team rm, use handcrafted reward shaping, no rm projection
    # task_name = 'navigation_good_p'  # using good rm projection, telling each agent what to do

    # env_name = 'pass_room'
    # map_name = '4button3agent'
    # map_name = '8button5agent'
    # task_name = 'pass'
    # task_name = 'pass_rs'  # 4 states RM, worst performance
    # task_name = 'pass_rs2'
    # task_name = 'pass2'  # 3 states RM, with back transitions
    # task_name = 'pass2_rs'
    # task_name = 'pass3'  # 31 states RM, best performance
    # task_name = 'pass3_rs'
    # task_name = 'pass4'  # 3 states RM, without back transitions
    # task_name = 'pass4_rs'

    ################## decide which algorithm to run ###############################
    # alg_name= 'cqrm'  # centralized qrm
    # alg_name = 'dqprm_s'  # decentralized q-learning for projected rm, source code
    # alg_name = 'ihrl'
    # alg_name = 'iql'
    # alg_name = 'dqprm'  # decentralized q-learning for projected rm, modified code
    # alg_name = 'iqrm'  # independent qrm
    # alg_name = 'hie_iqrm'  # hierarchical iqrm
    alg_name = 'hie_iqrm2'  # hierarchical iqrm with option elimination & sub-rm generation
    # alg_name = 'mul_hie_iqrm'  # multi-level hierarchical iqrm with option elimination & sub-rm generation
    # alg_name = 'hie_iqrm_3L'  # 3-level hierarchical iqrm with option elimination & sub-rm generation
    # alg_name = 'modular'  # HRL baseline

    print('num_times:', independent_trail_times)
    print('env_name:', env_name)
    print('map_name:', map_name)
    print('task_name:', task_name)
    print('alg_name:', alg_name)

    if env_name == 'rendezvous':
        from src.config.rendezvous_config import rendezvous_config

        tester = rendezvous_config(independent_trail_times)  # Get test object from config script
    elif env_name == 'buttons':
        from src.config.buttons_config import buttons_config

        tester = buttons_config(independent_trail_times)  # Get test object from config script
    elif env_name == 'minecraft':
        from src.config.minecraft_config import minecraft_config

        tester = minecraft_config(independent_trail_times, task_name)  # Get test object from config script
    elif env_name == 'minecraft2':
        from src.config.minecraft2_config import minecraft2_config

        tester = minecraft2_config(independent_trail_times, task_name, map_name)  # Get test object from config script
    elif env_name == 'pass_room':
        from src.config.pass_room_config import pass_room_config

        tester = pass_room_config(independent_trail_times, task_name, map_name)  # Get test object from config script
    else:
        raise ValueError('No such environment: ' + env_name)

    print_params(tester, alg_name)

    if alg_name == 'cqrm':
        from algorithms.cqrm import run_cqrm_experiment

        run_cqrm_experiment(tester, independent_trail_times, show_print=True)
    elif alg_name == 'ihrl':
        from algorithms.ihrl import run_ihrl_experiment

        run_ihrl_experiment(tester, independent_trail_times, show_print=True)
    elif alg_name == 'iql':
        from algorithms.iql import run_iql_experiment

        run_iql_experiment(tester, independent_trail_times, show_print=True)
    elif alg_name == 'dqprm_s':
        from algorithms.dqprm_s import run_dqprm_experiment

        run_dqprm_experiment(tester, independent_trail_times, show_print=True)
    elif alg_name == 'dqprm':
        from algorithms.dqprm import run_dqprm_experiment

        run_dqprm_experiment(tester, independent_trail_times, show_print=True)
    elif alg_name == 'iqrm':
        from algorithms.iqrm import run_iqrm_experiment

        run_iqrm_experiment(tester, independent_trail_times, show_print=True)
    elif alg_name == 'hie_iqrm':

        from algorithms.hie_iqrm import run_hie_iqrm_experiment

        run_hie_iqrm_experiment(tester, independent_trail_times, show_print=True)
    elif alg_name == 'hie_iqrm2':
        from algorithms.hie_iqrm2 import run_hie_iqrm2_experiment

        run_hie_iqrm2_experiment(tester, independent_trail_times, show_print=True)
    elif alg_name == 'mul_hie_iqrm':
        from algorithms.mul_hie_iqrm import run_mul_hie_iqrm_experiment

        run_mul_hie_iqrm_experiment(tester, independent_trail_times, show_print=True)
    elif alg_name == 'hie_iqrm_3L':
        from algorithms.hie_iqrm_3L import run_hie_iqrm_3L_experiment

        run_hie_iqrm_3L_experiment(tester, independent_trail_times, show_print=True)
    elif alg_name == 'modular':
        from algorithms.modular import run_modular_experiment

        run_modular_experiment(tester, independent_trail_times, show_print=True)
    else:
        raise ValueError('No such algorithm: ' + alg_name)

    end_time = time.time()
    total_min = (end_time - start_time) / 60
    print('Total time: %.2f min' % total_min)


    # Save the results
    parentDir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    now = datetime.now()

    data_path = os.path.join(parentDir, os.pardir, 'rm_cooperative_marl_data')
    if not os.path.isdir(data_path):
        os.mkdir(data_path)

    experiment_data_path = os.path.join(data_path, now.strftime("%Y%m%d"))

    if not os.path.isdir(experiment_data_path):
        os.mkdir(experiment_data_path)

    env_path = os.path.join(experiment_data_path, env_name + map_name)

    if not os.path.isdir(env_path):
        os.mkdir(env_path)


    # save_file_str = r'\{}_'.format(now.strftime("%Y-%m-%d_%H-%M-%S"))
    # save_file_str = r'\{}_'.format(now.strftime("%Y-%m-%d_%H-%M"))

    ### store as pickle
    # save_file_str = save_file_str + experiment + '.p'
    # save_file = open(experiment_data_path + save_file_str, "wb")
    # pickle.dump(tester, save_file)

    ### store as numpy dict
    save_file_str = os.path.join(env_path, task_name + alg_name) + '.npy'
    result_dict = tester.results['testing_steps']
    np.save(save_file_str, result_dict)
