import numpy as np
import random, time

from src.tester.tester import Tester
from src.Agent.iqrm_agent import Agent
from src.Environments.load_env import *
import matplotlib.pyplot as plt
import wandb

def run_qlearning_task(epsilon,
                       tester,
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

    for i in range(num_agents):
        agent_list[i].initialize_reward_machine()

    num_steps = learning_params.max_timesteps_per_task

    # training_envs = load_agent_envs(tester, agent_list)  # training_envs: list
    env = load_testing_env(tester)

    for t in range(num_steps):
        # Update step count
        tester.add_step()

        # independent q-learning
        # Perform a q-learning step.
        if not all(agent.is_task_complete for agent in agent_list):
            s = env.get_state()
            a = np.array([agent_list[i].get_next_action(s[i], epsilon, learning_params) for i in range(num_agents)])
            _, l, s_new = env.environment_step(a)  # do not use reward from env

            for i in range(num_agents):
                agent_i = agent_list[i]
                u1 = agent_i.u
                u2 = agent_i.rm.get_next_state(u1, l)
                r_i = agent_i.rm.get_reward(u1, u2)  # use reward from rm

                agent_i.update_agent(s=s[i],
                                     a=a[i],
                                     reward=r_i,
                                     s_new=s_new[i],
                                     label=l,
                                     learning_params=learning_params)

                # update Q-functions of other RM states
                for u_other in agent_i.rm.U:
                    if not (u_other == u1) and not (u_other in agent_i.rm.T):
                        l = env.get_mdp_label(s, s_new, u_other)  # counterfactual label
                        u2_other = agent_i.rm.get_next_state(u_other, l)
                        r = agent_i.rm.get_reward(u_other, u2_other)
                        agent_i.update_q_function(s[i], s_new[i], u_other, u2_other, a[i], r, learning_params)

        # If enough steps have elapsed, test and save the performance of the agents.
        if tester.start_test():
            t_init = time.time()
            step = tester.get_current_step()

            agent_list_copy = []

            # Need to create a copy of the agent for testing. If we pass the agent directly
            # mid-episode to the test function, the test will reset the world-state and reward machine
            # state before the training episode has been completed.
            for i in range(num_agents):
                rm_file = agent_list[i].rm_file
                actions = agent_list[i].actions
                agent_id = agent_list[i].agent_id
                num_states = agent_list[i].num_states
                agent_copy = Agent(rm_file, num_states, actions, agent_id)
                # Pass only the q-function by reference so that the testing updates the original agent's q-function.
                agent_copy.q = agent_list[i].q

                agent_list_copy.append(agent_copy)

            # Run a test of the performance of the agents
            testing_reward, trajectory, testing_steps = run_test(agent_list_copy,
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
            if tester.stop_task(t):
                break

        # checking the steps time-out
        if tester.stop_learning():
            break


def run_test(agent_list,
             tester,
             learning_params,
             testing_params,
             show_print=True):
    """
    Run a test of the q-learning with reward machine method with the current q-function.

    Parameters
    ----------
    agent_list : list of Agent objects
        Agent objects to be trained and tested.
    learning_params : LearningParameters object
        Object storing parameters to be used in learning.
    Testing_params : TestingParameters object
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

    s_team = testing_env.get_state()
    a_team = np.full(num_agents, -1, dtype=int)
    u_team = np.full(num_agents, -1, dtype=int)
    for i in range(num_agents):
        u_team[i] = agent_list[i].u
    testing_reward = 0

    trajectory = []

    steps = 0

    # Starting interaction with the environment
    for t in range(testing_params.num_steps):
        steps = steps + 1
        s_team = testing_env.get_state()
        # Perform a team step
        for i in range(num_agents):
            a = agent_list[i].get_next_action(s_team[i], -1.0, learning_params)
            a_team[i] = a
            u_team[i] = agent_list[i].u

        # trajectory.append({'s' : np.array(s_team, dtype=int), 'a' : np.array(a_team, dtype=int), 'u_team': np.array(u_team, dtype=int), 'u': int(testing_env.u)})

        r, l, s_team_next = testing_env.environment_step(a_team)

        testing_reward = testing_reward + r

        for i in range(num_agents):
            agent_list[i].update_agent(s=s_team[i],
                                       a=a_team[i],
                                       reward=r,
                                       s_new=s_team_next[i],
                                       label=l,
                                       learning_params=learning_params,
                                       update_q_function=False)

        if testing_env.reward_machine.is_terminal_state(testing_env.u):
            break

    if show_print:
        print('Reward of {} achieved in {} steps. Current step: {} of {}'.format(testing_reward, steps,
                                                                                 tester.current_step,
                                                                                 tester.total_steps))
    if tester.use_wandb:
        wandb.log({"reward": testing_reward,
                   "testing steps": steps,
                   "avg_reward": testing_reward/steps})
    from src.run import show_trajectory
    if show_trajectory:
        testing_env.show()
    return testing_reward, trajectory, steps


def run_iqrm_experiment(tester,
                        num_times,
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

    for t in range(num_times):
        start_time = time.time()
        # Reseting default step values
        tester.restart()

        rm_test_file = tester.rm_test_file
        rm_learning_file_list = tester.rm_learning_file_list

        testing_env = load_testing_env(tester)
        num_agents = testing_env.num_agents

        # Verify that the number of local reward machines matches the number of agents in the experiment.
        # assertion_string = "Number of specified local reward machines must match specified number of agents."
        # assert (len(tester.rm_learning_file_list) == num_agents), assertion_string

        # Create the a list of agents for this experiment
        agent_list = []
        for i in range(num_agents):
            actions = testing_env.get_actions(i)
            agent_list.append(Agent(rm_test_file, testing_env.num_states, actions, i))
            # all agents use the same rm

        num_episodes = 0

        # Task loop
        epsilon = learning_params.initial_epsilon

        while not tester.stop_learning():
            num_episodes += 1
            # epsilon = epsilon*0.99
            run_qlearning_task(epsilon,
                               tester,
                               agent_list,
                               show_print=show_print)

        # Backing up the results
        end_time = time.time()
        step_time = (end_time - start_time) / (tester.total_steps / tester.step_unit)
        print('Finished iteration ', t, 'Running time %.4f s per %d steps' % (step_time, tester.step_unit))

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
