import numpy as np
import random, time, os, math
from src.Agent.modular_agent import Agent, High_Controller
from src.tester.tester import Tester
from src.Environments.load_env import *
import matplotlib.pyplot as plt


def run_qlearning_task(epsilon,
                       tester,
                       controller,
                       agent_list,
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

    num_agents = len(agent_list)

    # controller.initialize_reward_machine()
    for i in range(num_agents):
        agent_list[i].initialize_reward_machine()

    max_episode_length = learning_params.max_timesteps_per_task

    env = load_testing_env(tester)
    steps = 0
    while steps < max_episode_length and not env.reward_machine.is_terminal_state(env.u):
        s_start = env.get_state()
        # choose the best rm for each agent by controller
        o = controller.get_next_option(s_start, epsilon, learning_params)
        for ag_id in range(num_agents):
            agent_list[ag_id].set_rm(rm_id=o[ag_id])

        G = 0  # cumulative discounted team reward
        tau = 0
        while (tau < tester.max_option_length) and not env.reward_machine.is_terminal_state(env.u):
            # Perform a q-learning step.
            s = env.get_state()
            a = np.array([agent_list[i].get_next_action(s[i], epsilon, learning_params) for i in range(num_agents)])
            r_team, l, s_new = env.environment_step(a)
            G = math.pow(tester.learning_params.gamma_controller, tau) * r_team + G

            for ag_id in range(num_agents):
                agent = agent_list[ag_id]
                u1 = agent.u
                u2 = agent.rm.get_next_state(u1, l)
                r = agent.rm.get_reward(u1, u2)
                agent.update_q_function(rm_id=o[ag_id],
                                        s=s[ag_id],
                                        s_new=s_new[ag_id],
                                        u=u1,
                                        u_new=u2,
                                        a=a[ag_id],
                                        reward=r,
                                        learning_params=learning_params)
                agent.update_agent(label=l)  # update RM state of this agent
                # update all the Q-functions of all RMs of this agent
                # for rm_id in range(agent.num_rms):
                #     rm = agent.avail_rms[rm_id]
                #     for u1_ in rm.U:
                #         u2_ = rm.get_next_state(u1_, l)
                #         r = rm.get_reward(u1_, u2_)
                #         agent.update_q_function(rm_id=rm_id,
                #                                 s=s[ag_id],
                #                                 s_new=s_new[ag_id],
                #                                 u=u1_,
                #                                 u_new=u2_,
                #                                 a=a[ag_id],
                #                                 reward=r,
                #                                 learning_params=learning_params)
            # Update step count
            tau += 1
            steps += 1
            tester.add_step()
            # update the high-level controller after executing the option
            is_option_terminal = controller.is_option_terminal(o, l)
            if (steps >= max_episode_length) or tester.start_test() or is_option_terminal:
                break
            if env.reward_machine.is_terminal_state(env.u):
                pass

        ################ option has been completed ######################
        # update Q-function of the controller
        if not tester.start_test():
            controller.update_q_function(s_start, o, G, tau, s, learning_params)

        # If enough steps have elapsed, test and save the performance of the agents.
        if tester.start_test():
            t_init = time.time()
            step = tester.get_current_step()

            agent_list_copy = []

            # Need to create a copy of the agent for testing. If we pass the agent directly
            # mid-episode to the test function, the test will reset the world-state and reward machine
            # state before the training episode has been completed.
            for i in range(num_agents):
                actions = agent_list[i].actions
                agent_id = agent_list[i].agent_id
                num_states = agent_list[i].num_states
                local_event_set = env.event_set_of_agents[i]
                agent_copy = Agent(local_event_set, num_states, actions, agent_id)
                # Pass only the q-function by reference so that the testing updates the original agent's q-function.
                agent_copy.q = agent_list[i].q
                agent_list_copy.append(agent_copy)
            controller_copy = High_Controller(controller.dim_option, agent_list_copy)
            controller_copy.q = controller.q
            # Run a test of the performance of the agents
            testing_reward, trajectory, testing_steps = run_test(controller_copy,
                                                                 agent_list_copy,
                                                                 tester,
                                                                 learning_params,
                                                                 testing_params,
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

        # If each agent has completed its task, reset it to its initial state.
        if all(agent.is_task_complete for agent in agent_list):
            for i in range(num_agents):
                agent_list[i].initialize_reward_machine()

            # Make sure we've run at least the minimum number of training steps before breaking the loop
            if tester.stop_task(steps):
                break

        # checking the steps time-out
        if tester.stop_learning():
            break


def run_test(controller,
             agent_list,
             tester,
             learning_params,
             testing_params,
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
    num_agents = len(agent_list)

    testing_env = load_testing_env(tester)

    for i in range(num_agents):
        agent_list[i].initialize_reward_machine()

    a_team = np.full(num_agents, -1, dtype=int)
    testing_reward = 0

    trajectory = []
    steps = 0
    num_steps = testing_params.num_steps

    # Starting interaction with the environment
    while (steps < num_steps) and not testing_env.is_terminal():
        s_start = testing_env.get_state()
        o = controller.get_next_option(s_start, -1.0, learning_params)
        print([agent_list[ag_id].avail_rms[o[ag_id]].tag for ag_id in range(len(o))])  # show which sub-rm is executed
        for ag_id in range(num_agents):
            agent_list[ag_id].set_rm(rm_id=o[ag_id])
        tau = 0
        while (tau < tester.max_option_length) and not testing_env.is_terminal():
            # Perform a team step
            s_team = testing_env.get_state()
            for i in range(num_agents):
                a_team[i] = agent_list[i].get_next_action(s_team[i], -1.0, learning_params)
            # trajectory.append({'s' : np.array(s_team, dtype=int), 'a' : np.array(a_team, dtype=int), 'u_team': np.array(u_team, dtype=int), 'u': int(testing_env.u)})
            r, l, s_team_next = testing_env.environment_step(a_team)
            testing_reward = testing_reward + r
            tau += 1
            steps += 1
            for i in range(num_agents):
                agent_list[i].update_agent(l)
            is_option_terminal = controller.is_option_terminal(o, l)
            if (steps >= num_steps) or is_option_terminal:
                break


    if show_print:
        print('Reward of {} achieved in {} steps. Current step: {} of {}'.format(testing_reward, steps,
                                                                                 tester.current_step,
                                                                                 tester.total_steps))
    from src.run import show_trajectory
    if show_trajectory:
        testing_env.show()
    return testing_reward, trajectory, steps


def run_modular_experiment(tester,
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

        # rm_test_file = tester.rm_test_file  # rm file of team task
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
        controller = High_Controller(num_rm_list, agent_list)
        num_episodes = 0

        # Task loop
        epsilon = learning_params.initial_epsilon

        while not tester.stop_learning():
            num_episodes += 1
            # epsilon = epsilon*0.99
            run_qlearning_task(epsilon,
                               tester,
                               controller,
                               agent_list,
                               show_print=show_print)

        # Backing up the results
        end_time = time.time()
        step_time = (end_time - start_time) / (tester.total_steps / tester.step_unit)
        print('Finished iteration ', trail_id, 'Running time %.4f s per %d steps' % (step_time, tester.step_unit))

    # tester.agent_list = agent_list
    #
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
