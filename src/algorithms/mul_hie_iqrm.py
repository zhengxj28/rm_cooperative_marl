import numpy as np
import random, time, os, math
from src.Agent.mul_hie_iqrm_agent import *
from src.tester.tester import Tester
from src.Environments.load_env import *
import matplotlib.pyplot as plt


def run_level(env, level, tester, high_levels, update_q_function):
    """

    Parameters
    ----------
    env: environment
    level: int, current level
    tester: Tester, storing information of hyper parameters
    high_levels: High_Levels, storing information of steps, tau, structure of task (represented as multi-level RM)
    update_q_function: bool, True in run_qlearning, False in run_test

    Returns
    -------

    """
    learning_params = tester.learning_params
    testing_params = tester.testing_params
    max_episode_length = tester.learning_params.max_timesteps_per_task
    max_option_length = tester.max_option_length
    if level == 0:
        while (high_levels.steps < max_episode_length) \
                and (high_levels.tau < max_option_length) \
                and (not high_levels.is_rm_state_changed(level + 1)):
            s = env.get_state()
            a = high_levels.get_action(level)
            _, l, s_new = env.environment_step(a)

            high_levels.update_rm_states(level)
            if update_q_function:
                high_levels.update_Q_functions(level, l)
            high_levels.calculate_G(level + 1)
            high_levels.get_label(level + 1)
            high_levels.update_rm_states(level + 1)
            high_levels.add_step()
            s = s_new
            if tester.start_test():
                break
    else:
        while (high_levels.steps < max_episode_length) \
                and (high_levels.tau < max_option_length) \
                and (not high_levels.is_rm_state_changed(level + 1)):
            high_levels.get_action(level)
            run_level(env, level - 1, tester, high_levels, update_q_function)
            high_levels.update_rm_states(level)
            if update_q_function:
                high_levels.update_Q_functions(level)
            high_levels.calculate_G(level + 1)
            high_levels.get_label(level + 1)
            high_levels.update_rm_states(level + 1)
            high_levels.add_step()
            tester.add_step()
            if tester.start_test():
                break
    return


def run_qlearning_task(tester,
                       high_levels,
                       show_print=True):
    """
    This code runs one q-learning episode. q-functions, and accumulated reward values of agents
    are updated accordingly. If the appropriate number of steps have elapsed, this function will
    additionally run a test episode.

    Parameters
    ----------
    epsilon : float
        Numerical value in (0,1) representing likelihood of choosing a random action.
    tester : Tester object
        Object containing necessary information for current experiment.
    agent_list : list of Agent objects
        Agent objects to be trained and tested.
    show_print : bool
        Optional flag indicating whether or not to print output statements to terminal.
    """
    # Initializing parameters and the game
    learning_params = tester.learning_params
    testing_params = tester.testing_params

    high_levels.initialize()
    num_levels = high_levels.num_levels

    env = load_testing_env(tester)

    run_level(env, num_levels - 1, tester, high_levels, update_q_function=True)

    if tester.start_test():
        step = tester.get_current_step()
        high_levels_copy = High_Levels(high_levels.task_name)
        testing_reward, trajectory, testing_steps = run_test(tester,
                                                             high_levels_copy,
                                                             show_print=show_print)
        # Save the testing reward
        if 0 not in tester.results.keys():
            tester.results[0] = {}
        if step not in tester.results[0]:
            tester.results[0][step] = []
        tester.results[0][step].append(testing_reward)

        # Save the testing trace
        if 'trajectories' not in tester.results.keys():
            tester.results['trajectories'] = {}
        if step not in tester.results['trajectories']:
            tester.results['trajectories'][step] = []
        tester.results['trajectories'][step].append(trajectory)

        # Save how many steps it took to complete the task
        if 'testing_steps' not in tester.results.keys():
            tester.results['testing_steps'] = {}
        if step not in tester.results['testing_steps']:
            tester.results['testing_steps'][step] = []
        tester.results['testing_steps'][step].append(testing_steps)

        # Keep track of the steps taken
        if len(tester.steps) == 0 or tester.steps[-1] < step:
            tester.steps.append(step)


def run_test(tester,
             high_levels,
             show_print=True):
    """
    Run a test of the q-learning with reward machine method with the current q-function.

    Parameters
    ----------
    controller: high_level controller
    agent_list : list of Agent objects
        Agent objects to be trained and tested.
    learning_params : LearningParameters object
        Object storing parameters to be used in learning.
    testing_params : TestingParameters object
        Object storing parameters to be used in testing.

    Ouputs
    ------
    testing_reard : float
        Reward achieved by agent during this test episode.
    trajectory : list
        List of dictionaries containing information on current step of test.
    step : int
        Number of testing steps required to complete the task.
    """
    testing_env = load_testing_env(tester)
    num_levels = high_levels.num_levels

    run_level(testing_env, num_levels, tester, high_levels, update_q_function=False)

    if show_print:
        print('Reward of {} achieved in {} steps. Current step: {} of {}'.format(testing_reward, steps,
                                                                                 tester.current_step,
                                                                                 tester.total_steps))
    from src.run import show_trajectory
    if show_trajectory:
        testing_env.show()
    return testing_reward, trajectory, steps


def run_mul_hie_iqrm_experiment(tester,
                                independent_trail_times,
                                show_print=True):
    """
    Run the entire q-learning with reward machines experiment a number of times specified by num_times.

    Inputs
    ------
    tester : Tester object
        Test object holding true reward machine and all information relating
        to the particular tasks, world, learning parameters, and experimental results.
    num_agents : int
        Number of agents in this experiment.
    num_times : int
        Number of times to run the entire experiment (restarting training from scratch).
    show_print : bool
        Flag indicating whether or not to output text to the terminal.
    """

    learning_params = tester.learning_params

    for trail_id in range(independent_trail_times):
        start_time = time.time()
        # Reseting default step values
        tester.restart()

        rm_test_file = tester.rm_test_file  # rm file of team task
        testing_env = load_testing_env(tester)
        num_agents = testing_env.num_agents

        # Create the a list of agents for this experiment
        agent_list = []
        num_rm_list = []  # num of available rm of each agent

        for i in range(num_agents):
            actions = testing_env.get_actions(i)
            local_event_set = testing_env.event_set_of_agents[i]  # local events of agent i
            agent_i = Agent(local_event_set, testing_env.num_states, actions, i)
            agent_list.append(agent_i)
            num_rm_list.append(agent_i.num_rms)

        high_levels = High_Levels(tester, agent_list)

        num_episodes = 0

        # Task loop
        epsilon = learning_params.initial_epsilon

        while not tester.stop_learning():
            num_episodes += 1
            run_qlearning_task(tester,
                               high_levels,
                               show_print=show_print)

        # Backing up the results
        end_time = time.time()
        step_time = (end_time - start_time) / (tester.total_steps / tester.step_unit)
        print('Finished iteration ', trail_id, 'Running time %.4f s per %d steps' % (step_time, tester.step_unit))

    # # plot_multi_agent_results(tester, num_agents)
    # plot_multi_agent_results(tester.results['testing_steps'], num_agents)


def plot_multi_agent_results(plot_dict, num_agents):
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
    plt.plot(steps, prc_50, color='red')
    plt.plot(steps, prc_75, alpha=0)
    plt.grid()
    plt.fill_between(steps, prc_50, prc_25, color='red', alpha=0.25)
    plt.fill_between(steps, prc_50, prc_75, color='red', alpha=0.25)
    plt.ylabel('Testing Steps to Task Completion', fontsize=15)
    plt.xlabel('Training Steps', fontsize=15)
    plt.locator_params(axis='x', nbins=5)

    plt.show()
