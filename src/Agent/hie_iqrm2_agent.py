from src.reward_machines.sparse_reward_machine import SparseRewardMachine
from src.tester.tester import Tester
import numpy as np
import random, time, os, math
import matplotlib.pyplot as plt
import torch


class Agent:
    """
    Class meant to represent an individual RM-based learning agent.
    The agent maintains a representation of its own q-function and accumulated reward
    which are updated across training episodes.
    The agent also has a representation of its own local reward machine, which it uses
    for learning, and of its state in the world/reward machine.

    Note: Users of this class must manually reset the world state and the reward machine
    state when starting a new episode by calling self.initialize_world() and
    self.initialize_reward_machine().
    """

    def __init__(self, local_event_set, num_states, actions, agent_id):
        """
        Initialize agent object.

        Parameters
        ----------
        local_event_set : set
            The set of possible events of this agent.
        s_i : int
            Index of initial state.
        actions : list
            List of actions available to the agent.
        agent_id : int
            Index of this agent.
        """

        self.agent_id = agent_id
        self.actions = actions
        self.num_actions = len(actions)
        self.num_states = num_states
        self.local_event_set = local_event_set
        self.local_propositions = set()

        self.avail_rms = []  # available rms of this agent
        self.event2rm_id = dict()  # dict: convert event to rm_id
        for event in self.local_event_set:
            self.local_propositions = self.local_propositions.union(set(event))

        # the sub-rm is automatically constructed
        rm_id = 0
        for event in self.local_event_set:
            rm = SparseRewardMachine()
            rm.build_atom_rm(event, self.local_propositions, self.local_event_set)
            self.avail_rms.append(rm)
            self.event2rm_id[event] = rm_id
            rm_id += 1
        self.rm_id2event = dict()  # dict: convert rm_id to event
        for event, rm_id in self.event2rm_id.items():
            self.rm_id2event[rm_id] = event

        self.num_rms = len(self.avail_rms)

        self.rm_id = 0  # id of current chosen rm
        self.rm = self.avail_rms[self.rm_id]  # current chosen rm
        self.u = self.rm.get_initial_state()

        num_states_of_avail_rm = np.array([len(rm_.U) for rm_ in self.avail_rms])
        max_rm_states = num_states_of_avail_rm.max()
        self.q = np.zeros([self.num_rms, num_states, max_rm_states, len(self.actions)])
        self.total_local_reward = 0
        self.is_task_complete = 0

    def set_rm(self, rm_id):
        self.rm = self.avail_rms[rm_id]
        self.u = self.avail_rms[rm_id].u0
        self.is_task_complete = self.rm.is_terminal_state(self.u)
        self.rm_id = rm_id

    def initialize_reward_machine(self):
        """
        Reset the state of the reward machine to the initial state and reset task status.
        """
        self.u = self.rm.get_initial_state()
        self.is_task_complete = 0

    def is_local_event_available(self, label):
        if label:  # Only try accessing the first event in label if it exists
            event = label[0]
            return self.rm.is_event_available(self.u, event)
        else:
            return False

    def get_next_action(self, s, epsilon, learning_params):
        """
        Return the action next action selected by the agent.

        Outputs
        -------
        a : int
            Selected next action for this agent.
        """

        T = learning_params.T

        if random.random() < epsilon:
            # a = random.choice(self.actions)
            weight = np.ones([self.num_actions])
        else:
            weight = np.exp(self.q[self.rm_id, s, self.u, :] * T)
        pr = weight / np.sum(weight)  # pr[a] is probability of taking action a

        # If any q-values are so large that the softmax function returns infinity,
        # make the corresponding actions equally likely
        if any(np.isnan(pr)):
            print('BOLTZMANN CONSTANT TOO LARGE IN ACTION-SELECTION SOFTMAX.')
            temp = np.array(np.isnan(pr), dtype=float)
            pr = temp / np.sum(temp)

        pr = torch.tensor(pr)
        dist = torch.distributions.Categorical(pr)
        a = dist.sample()

        return a

    def update_agent(self, label):
        """
        Update the agent's state, q-function, and reward machine after
        interacting with the environment.

        Parameters
        ----------
        s_new : int
            Index of the agent's next state.
        a : int
            Action the agent took from the last state.
        reward : float
            Reward the agent achieved during this step.
        label : string
            Label returned by the MDP this step.
        learning_params : LearningParameters object
            Object storing parameters to be used in learning.
        """

        # Keep track of the RM location at the start of the
        u2 = self.rm.get_next_state(self.u, label)
        self.u = u2

        if self.rm.is_terminal_state(self.u):
            # Completed task. Set flag.
            self.is_task_complete = 1
        else:
            self.is_task_complete = 0

    def update_q_function(self, rm_id, s, s_new, u, u_new, a, reward, learning_params):
        """
        Update the q function using the action, states, and reward value.

        Parameters
        ----------
        s : int
            Index of the agent's previous state
        s_new : int
            Index of the agent's updated state
        u : int
            Index of the agent's previous RM state
        U_new : int
            Index of the agent's updated RM state
        a : int
            Action the agent took from state s
        reward : float
            Reward the agent achieved during this step
        learning_params : LearningParameters object
            Object storing parameters to be used in learning.
        """
        alpha = learning_params.alpha
        gamma = learning_params.gamma

        if self.rm.is_terminal_state(u_new):
            self.q[rm_id][s][u][a] = (1 - alpha) * self.q[rm_id][s][u][a] + alpha * reward
        else:
            # Bellman update
            self.q[rm_id][s][u][a] = (1 - alpha) * self.q[rm_id][s][u][a] \
                                     + alpha * (reward + gamma * np.amax(self.q[rm_id][s_new][u_new]))


class High_Controller:
    """
    The high-level controller helps the agents choose their own
    subtask (rm) properly to complete the whole team task efficiently.
    """

    def __init__(self, rm_file_name, num_rm_list, agent_list):
        """
        Initialize agent object.

        Parameters
        ----------
        rm_file_name : str
            File path pointing to the reward machine of team task.
        num_rm_list : list
            Num of available rm of each agent,
        """
        self.rm_file_name = rm_file_name  # team task
        self.rm = SparseRewardMachine(self.rm_file_name)

        self.u = self.rm.get_initial_state()
        self.local_propositions = self.rm.get_propositions()

        self.num_agents = len(num_rm_list)
        self.num_rm_list = num_rm_list

        """
        Create a list of the dimension of high-level q-function
        Let N be num_agents, O_i is num_rm of agent i, then the shape is UxO1X...XON
        each option is the rm tuple of agents: o=(rm1,rm2,...,rmN)
        """
        q_shape = [len(self.rm.U), ] + num_rm_list
        self.q = np.zeros(q_shape)  # for softmax selection
        # self.q = np.ones(q_shape)  # for epsilon-greedy
        self.num_options = self.q[0].size  # number of all possible options
        self.is_task_complete = 0

        # action_mask_matrix[u][o]=1 iff option o causes rm transit to another state u'!=u
        # action_mask_matrix[u][o]=1 iff rm does not transit to another state
        self.action_mask_matrix = np.ones(q_shape, dtype=int)
        for u in self.rm.U:
            for o_index in range(self.num_options):
                o = np.unravel_index(o_index, self.num_rm_list)
                events = set()  # events of all agent under option o
                for ag_id in range(self.num_agents):
                    local_event = set(agent_list[ag_id].rm_id2event[o[ag_id]])
                    events = events.union(local_event)
                events = tuple(sorted(list(events)))
                if self.rm.get_next_state(u, events) == u:
                    self.action_mask_matrix[u][o] = 0

    def initialize_reward_machine(self):
        """
        Reset the state of the reward machine to the initial state and reset task status.
        """
        self.u = self.rm.get_initial_state()
        self.is_task_complete = 0

    def is_local_event_available(self, label):
        if label:  # Only try accessing the first event in label if it exists
            event = label[0]
            return self.rm.is_event_available(self.u, event)
        else:
            return False

    def get_next_option(self, epsilon, learning_params):
        """
        Return the action next action selected by the agent.

        Outputs
        -------
        s : int
            Index of the agent's current state.
        a : int
            Selected next action for this agent.
        """

        T = learning_params.T_controller  # temperature of softmax
        if random.random() < epsilon:
            weight = np.ones(self.num_rm_list)
            # o = []
            # for i in range(self.num_agents):
            #     o.append(random.choice([id for id in range(self.num_rm_list[i])]))
        else:
            # epsilon-greedy implementation
            # o = np.unravel_index(self.q[self.u].argmax(), self.q[self.u].shape)

            # softmax implementation
            weight = np.exp(self.q[self.u, :] * T)

        # action mask, eliminate forbidden option
        weight = np.multiply(weight, self.action_mask_matrix[self.u, :])

        pr = weight / np.sum(weight)  # pr[a] is probability of taking action a
        pr = np.reshape(pr, [pr.size])  # reshape to 1d array
        # If any q-values are so large that the softmax function returns infinity,
        # make the corresponding actions equally likely
        if any(np.isnan(pr)):
            print('GET OPTION: BOLTZMANN CONSTANT TOO LARGE IN ACTION-SELECTION SOFTMAX.')
            temp = np.array(np.isnan(np.reshape(pr, [pr.size])), dtype=float)
            pr = temp / np.sum(temp)

        pr = torch.tensor(pr)
        dist = torch.distributions.Categorical(pr)
        o = dist.sample()
        o = np.unravel_index(o, self.num_rm_list)

        return list(o)

    def update_controller(self, label):
        """
        Update the state of reward machine after
        interacting with the environment.

        Parameters
        ----------
        label : string
            Label returned by the MDP this step.
        """

        # Keep track of the RM location at the start of the
        u2 = self.rm.get_next_state(self.u, label)
        is_state_changed = (self.u != u2)
        self.u = u2

        # Moving to the next state
        # Completed task. Set flag.
        self.is_task_complete = self.rm.is_terminal_state(self.u)
        return is_state_changed

    def update_q_function(self, u_start, o, G, tau, learning_params):
        """
        Update the q function using the action, states, and reward value.

        Parameters
        ----------
        u_start : int
            Index of the agent's RM state when starting the option
        o : int
            Option, i.e. chosen RM
        G : float
            Cumulative discounted team reward during executing the option
        tau: int
            Total steps of executing the option
        learning_params : LearningParameters object
            Object storing parameters to be used in learning.
        """
        alpha = learning_params.alpha_controller
        gamma = learning_params.gamma_controller

        o = tuple(o)
        u_new = self.u
        if self.rm.is_terminal_state(u_new):
            self.q[u_start][o] = (1 - alpha) * self.q[u_start][o] + alpha * G
        else:
            # Bellman update
            self.q[u_start][o] = (1 - alpha) * self.q[u_start][o] \
                                 + alpha * (G + math.pow(gamma, tau) * self.q[u_new].max())


if __name__ == '__main__':  # for debug only
    local_event_set = {'', 'a', 'b', 'c', 'd', 'r', ('c', 'r'), ('d', 'r')}
    test_agent = Agent(local_event_set=local_event_set,
                       num_states=100,
                       actions=[0, 1, 2, 3, 4],
                       agent_id=0)
    print()
