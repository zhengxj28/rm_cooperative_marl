from src.reward_machines.sparse_reward_machine import SparseRewardMachine
from src.tester.tester import Tester
import numpy as np
import random, time, os, math
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from src.Agent.networks import QNet
from buffer import ReplayBuffer


class Agent:
    """
    An Agent can be considered as level-0 controller.
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

    def __init__(self, num_option_list, agent_list, learning_params):
        """
        Initialize agent object.

        Parameters
        ----------
        rm_file_name : str
            File path pointing to the reward machine of team task.
        num_option_list : list
            Num of available rm of each agent,
        """

        self.num_agents = len(num_option_list)
        self.dim_option = num_option_list
        state_size = agent_list[0].num_states
        """
        Create a list of the dimension of high-level q-function
        Let N be num_agents, O_i is num_rm of agent i, then the shape is UxO1X...XON
        each option is the rm tuple of agents: o=(rm1,rm2,...,rmN)
        """
        self.num_options = np.zeros(num_option_list).size  # dimension of option
        # self.q = None  # Done: use dqn, input joint state:s output: option
        self.q = QNet(input_dim=self.num_agents,
                      hidden_dim=learning_params.hidden_dim,
                      output_dim=self.num_options,
                      state_size=state_size,
                      embedding_size=learning_params.embedding_size)  # TODO: init parameters @wzfyyds
        self.target_q = QNet()  # target network
        self.learn_step = 0
        # self.q = np.zeros(num_option_list)  # for debug only
        self.is_task_complete = 0
        self.buffer = ReplayBuffer(learning_params.buffer_size)  # TODO: capacity
        self.loss_fn = nn.MSELoss()
        self.optim = optim.Adam(self.q.parameters(), lr=learning_params.lr)  # TODO: lr

        self.option2event = dict()
        for o_index in range(self.num_options):
            o = np.unravel_index(o_index, self.dim_option)
            events = set()  # events of all agent under option o
            for ag_id in range(self.num_agents):
                local_event = set(agent_list[ag_id].rm_id2event[o[ag_id]])
                events = events.union(local_event)
            events = tuple(sorted(list(events)))
            self.option2event[o] = events


        # action_mask_matrix[o]=1 iff event of option o is possible
        # action_mask_matrix[o]=0 iff event impossible
        self.action_mask_matrix = np.ones(self.dim_option, dtype=int)
        # for o_index in range(self.num_options):
        #     o = np.unravel_index(o_index, self.dim_option)
        #     events = set()  # events of all agent under option o
        #     for ag_id in range(self.num_agents):
        #         local_event = set(agent_list[ag_id].rm_id2event[o[ag_id]])
        #         events = events.union(local_event)
        #     events = tuple(sorted(list(events)))
        #     if events in :
        #         self.action_mask_matrix[o] = 0

    def get_next_option(self, s, epsilon, learning_params):
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
            weight = np.ones(self.dim_option)
        else:
            weight = np.exp(self.q(s) * T)  # TODO: q(s) confirm

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
        o = np.unravel_index(o, self.dim_option)

        return list(o)

    def is_option_terminal(self, o, label):
        """
        Update the state of reward machine after
        interacting with the environment.

        Parameters
        ----------
        label : list
            Label returned by the MDP this step.
        """
        o = tuple(o)
        return self.option2event[o] == label

    def update_q_function(self, s_start, o, G, tau, s_new, learning_params):
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
        self.buffer.push(s_start, o, G, s_new)
        if self.buffer.len() >= learning_params.buffer_size:
            # TODO: epsilon greedy?
            
            # TODO: hard update target step
            if self.learn_step % learning_params.target_network_update_freq == 0:
                self.target_q.load_state_dict(self.q.state_dict())
            self.learn_step += 1
            
            # sample from replay buffer
            s_start, o, G, s_new = self.buffer.sample(args.batch_size)
            # DQN tau-step update
            q_eval = self.q(s_start).gather(-1, o.unsqueeze(-1)).squeeze(-1)
            q_next = self.target_q(s_new).detach()
            q_target = G + math.pow(gamma, tau) * torch.max(q_next, dim=-1)[0]
            loss = self.loss_fn(q_eval, q_target)
            self.optim.zero_grad()
            self.backward()
            self.optim.step()

        # alpha = learning_params.alpha_controller
        # gamma = learning_params.gamma_controller
        #
        # o = tuple(o)
        # if self.rm.is_terminal_state(u_new):
        #     self.q[s_start,o] = (1 - alpha) * self.q[s_start][o] + alpha * G
        # else:
        #     # Bellman update
        #     self.q[s_start,o] = (1 - alpha) * self.q[s_start][o] \
        #                          + alpha * (G + math.pow(gamma, tau) * self.q[s_new].max())


if __name__ == '__main__':  # for debug only
    local_event_set = {'', 'a', 'b', 'c', 'd', 'r', ('c', 'r'), ('d', 'r')}
    test_agent = Agent(local_event_set=local_event_set,
                       num_states=100,
                       actions=[0, 1, 2, 3, 4],
                       agent_id=0)
    print()
