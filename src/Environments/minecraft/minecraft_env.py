import random, math, os
import numpy as np
from enum import Enum

import sys, copy

# sys.path.append('../')
# sys.path.append('../../')
from src.reward_machines.sparse_reward_machine import SparseRewardMachine

"""
Enum with the actions that the agent can execute
"""


############# game object #######################
class Entity:
    def __init__(self, i, j):  # row and column
        self.i = i
        self.j = j

    def change_position(self, i, j):
        self.i = i
        self.j = j

    def idem_position(self, i, j):
        return self.i == i and self.j == j

    def interact(self, agent):
        return True


class Agent(Entity):
    def __init__(self, i, j, actions):
        super().__init__(i, j)
        self.num_keys = 0
        self.reward = 0
        self.actions = actions

    def get_actions(self):
        return self.actions

    def interact(self, agent):
        return False

    def update_reward(self, r):
        self.reward += r

    def __str__(self):
        return "A"


class Obstacle(Entity):
    def __init__(self, i, j):
        super().__init__(i, j)

    def interact(self, agent):
        return False

    def __str__(self):
        return "X"


class Empty(Entity):
    def __init__(self, i, j, label=" "):
        super().__init__(i, j)
        self.label = label

    def __str__(self):
        return self.label


class Actions(Enum):
    up = 0  # move up
    right = 1  # move right
    down = 2  # move down
    left = 3  # move left
    none = 4  # none


########### env definition ###############
class MineCraftParams:
    def __init__(self, file_map, consider_night):
        self.file_map = file_map
        self.consider_night = consider_night


class MultiAgentMineCraftEnv:

    def __init__(self, rm_file, env_settings):
        """
        Initialize environment.

        Parameters
        ----------
        rm_file : string
            File path leading to the text file containing the reward machine
            encoding this environment's reward function.
        num_agents : int
            Number of agents in the environment.
        env_settings : dict
            Dictionary of environment settings
        """
        self.env_settings = env_settings  # settings from tester

        self._load_map(self.env_settings['file_map'])
        self.s = self.s_i  # current states are initialized
        self.reward_machine = SparseRewardMachine(rm_file)

        self.u = self.reward_machine.get_initial_state()
        self.last_action = np.full(self.num_agents, -1, dtype=int)  # Initialize last action with garbage values

        self.consider_night = self.env_settings['consider_night']
        self.nSteps = 0
        self.hour = 12
        if self.consider_night:
            self.sunrise = 5
            self.sunset = 21

        self.p = self.env_settings['p']  # slip probability, noise action

    def _load_map(self, file_map):
        """
        This method adds the following attributes to the game:
            - self.map_array: array containing all the static objects in the map
                - e.g. self.map_array[i][j]: contains the object located on row
                        'i' and column 'j'
            - self.agents: are the agents
            - self.map_height: number of rows in every room
            - self.map_width: number of columns in every room
        The inputs:
            - file_map: path to the map file
        """
        # Initial position of all agents, load from map
        ag_init_pos = []
        # contains all the actions that the agent can perform
        actions = np.array(
            [Actions.up.value, Actions.right.value, Actions.left.value, Actions.down.value, Actions.none.value],
            dtype=int)
        # loading the map
        self.map_array = []
        # I use the lower case letters to define the features
        self.class_ids = {}
        self.agents = {}
        f = open(file_map)
        i, j = 0, 0
        ag = 0  # num of agents
        for l in f:
            # I don't consider empty lines!
            if (len(l.rstrip()) == 0): continue
            # this is not an empty line!
            row = []
            b_row = []
            j = 0
            for e in l.rstrip():
                if e in "abcdefghijklmnopqrstuvwxyzH":
                    entity = Empty(i, j, label=e)
                    if e not in self.class_ids:
                        self.class_ids[e] = len(self.class_ids)
                # we need to declare the initial positions of agents
                # to be potentially empty espaces (after they moved)
                elif e in " A":
                    entity = Empty(i, j)
                elif e == "X":
                    entity = Obstacle(i, j)
                else:
                    raise ValueError('Unkown entity ', e)
                if e == "A":
                    self.agents[ag] = Agent(i, j, actions)
                    ag_init_pos.append((i, j))
                    if e not in self.class_ids:
                        self.class_ids[e] = len(self.class_ids)
                    ag += 1
                row.append(entity)
                j += 1
            self.map_array.append(row)
            i += 1
        """
        We use this back map to check what was there when an agent leaves a 
        position
        """
        self.back_map = copy.deepcopy(self.map_array)
        for agent in self.agents.values():
            i, j = agent.i, agent.j
            self.map_array[i][j] = agent
        f.close()
        # height width
        self.map_height, self.map_width = len(self.map_array), \
                                          len(self.map_array[0])
        self.num_agents = len(ag_init_pos)
        self.num_states = self.map_height * self.map_width
        # print("There are", self.n_agents, "agents")
        # contains all the actions that the agent can perform

        self.actions = np.full((self.num_agents, len(actions)), -2, dtype=int)
        for i in range(self.num_agents):
            self.actions[i] = actions
        # convert position to initial states for all agents
        self.s_i = np.zeros([self.num_agents], dtype=int)
        for i in range(self.num_agents):
            self.s_i[i] = self.pos2state(ag_init_pos[i][0], ag_init_pos[i][1])

    def environment_step(self, s, a):
        """
        Execute collective action a from collective state s. Return the resulting reward,
        mdp label, and next state. Update the last action taken by each agent.

        Parameters
        ----------
        s : numpy integer array
            Array of integers representing the environment states of the various agents.
            s[id] represents the state of the agent indexed by index "id".
        a : numpy integer array
            Array of integers representing the actions selected by the various agents.
            a[id] represents the desired action to be taken by the agent indexed by "id.

        Outputs
        -------
        r : float
            Reward achieved by taking action a from state s.
        l : string
            MDP label emitted this step.
        s_next : numpy integer array
            Array of indeces of next team state.
        """

        self.hour = (self.hour + 1) % 24
        s_next = np.full(self.num_agents, -1, dtype=int)

        for i in range(self.num_agents):
            s_next[i], actual_action = self.get_next_state(s[i], a[i], i)
            self.last_action[i] = actual_action

        l = self.get_mdp_label(s, s_next, self.u)
        r = 0

        self.s = s_next  # update env states
        # for e in l:
        #     # Get the new reward machine state and the reward of this step
        #     u2 = self.reward_machine.get_next_state(self.u, e)
        #     r = r + self.reward_machine.get_reward(self.u, u2)
        #     # Update the reward machine state
        #     self.u = u2

        u2 = self.reward_machine.get_next_state(self.u, l)
        r = r + self.reward_machine.get_reward(self.u, u2)
        self.u = u2

        return r, l, s_next

    def get_next_state(self, s, a, agent_id):  # get next state for single agent
        """
         Get the next state in the environment given action a is taken from state s.
         Update the last action that was truly taken due to MDP slip.

         Parameters
         ----------
         s : int
             Index of the current state.
         a : int
             Action to be taken from state s.

         Outputs
         -------
         s_next : int
             Index of the next state.
         last_action : int
             Last action the agent truly took because of slip probability.
         """

        action = Actions(a)  # action = up, right, down, left, none
        slip_p = [self.p, (1 - self.p) / 2, (1 - self.p) / 2]
        check = random.random()

        row, col = self.state2pos(s)
        row_old, col_old = row, col

        # up    = 0
        # right = 1
        # down  = 2
        # left  = 3
        # stay  = 4

        a_ = a  # a_ : actual executed action
        if (check <= slip_p[0]) or (a == Actions.none.value):
            a_ = a
        elif (check > slip_p[0]) & (check <= (slip_p[0] + slip_p[1])):
            if a == 0:
                a_ = 3
            elif a == 2:
                a_ = 1
            elif a == 3:
                a_ = 2
            elif a == 1:
                a_ = 0
        else:
            if a == 0:
                a_ = 1
            elif a == 2:
                a_ = 3
            elif a == 3:
                a_ = 0
            elif a == 1:
                a_ = 2

        action_ = Actions(a_)
        if action_ == Actions.up:
            row -= 1
        if action_ == Actions.down:
            row += 1
        if action_ == Actions.left:
            col -= 1
        if action_ == Actions.right:
            col += 1

        if self.map_array[row][col].interact(agent_id):  # not obstacle
            s_next = self.pos2state(row, col)
            self.agents[agent_id].change_position(row, col)
            self.map_array[row_old][col_old] = self.back_map[row_old][col_old]
            self.map_array[row][col] = self.agents[agent_id]
        else:
            s_next = s

        actual_action = a_
        return s_next, actual_action

    def pos2state(self, row, col):
        return self.map_width * row + col

    def state2pos(self, s):
        """
        Return the row and column indeces of state s in the gridworld.

        Parameters
        ----------
        s : int
            Index of the gridworld state.

        Outputs
        -------
        row : int
            The row index of state s in the gridworld.
        col : int
            The column index of state s in the gridworld.
        """

        """
        width=5 example
        01234
        56789
        .....
        """
        row = np.floor_divide(s, self.map_width)
        col = np.mod(s, self.map_width)

        return row, col

    def _is_night(self):
        return not (self.sunrise <= self.hour <= self.sunset)

    def get_actions(self, id):
        """
        Returns the list with the actions that a particular agent can perform.

        Parameters
        ----------
        id : int
            Index of the agent whose initial state is being queried.
        """
        return np.copy(self.actions[id])

    def get_last_action(self, id):
        """
        Returns a particular agent's last action.

        Parameters
        ----------
        id : int
            Index of the agent whose initial state is being queried.
        """
        return self.last_action[id]

    def get_team_action_array(self):
        """
        Returns the available actions of the entire team.

        Outputs
        -------
        actions : (num_agents x num_actions) numpy integer array
        """
        return np.copy(self.actions)

    def get_initial_state(self, id):
        """
        Returns the initial state of a particular agent.

        Parameters
        ----------
        id : int
            Index of the agent whose initial state is being queried.
        """
        return self.s_i[id]

    def get_initial_team_state(self):
        """
        Return the intial state of the collective multi-agent team.

        Outputs
        -------
        s_i : numpy integer array
            Array of initial state indices for the agents in the experiment.
        """
        return np.copy(self.s_i)

    ############## DQPRM-RELATED METHODS ########################################
    def get_mdp_label(self, s, s_next, u=0):
        """
        Get the mdp label resulting from transitioning from state s to state s_next.

        Parameters
        ----------
        s : numpy integer array
            Array of integers representing the environment states of the various agents.
            s[id] represents the state of the agent indexed by index "id".
        s_next : numpy integer array
            Array of integers representing the next environment states of the various agents.
            s_next[id] represents the next state of the agent indexed by index "id".
        u : int
            Index of the reward machine state

        Outputs
        -------
        l : list
            MDP label resulting from the state transition.
        """

        event_set = set()
        for i in range(self.num_agents):
            row_i, col_i = self.state2pos(s_next[i])  # position of agent i
            event = str(self.back_map[row_i][col_i])
            if event == 'A': event = ' '
            event_set.add(event)
        if self.consider_night and self._is_night():
            event_set.add('n')
        return list(event_set)

    ################## HRL-RELATED METHODS ######################################
    # def get_options_list(self, agent_id):
    #     """
    #     Return a list of strings representing the possible options for each agent.
    #
    #     Input
    #     -----
    #     agent_id : int
    #         The id of the agent whose option list is to be returned.
    #
    #     Output
    #     ------
    #     options_list : list
    #         list of strings representing the options avaialble to the agent.
    #     """
    #
    #     agent1 = 0
    #     agent2 = 1
    #     agent3 = 2
    #
    #     options_list = []
    #
    #     return options_list
    #
    # def get_avail_options(self, agent_id):
    #     """
    #     Given the current metastate, get the available options. Some options are unavailable if
    #     they are not possible to complete at the current stage of the task. In such circumstances
    #     we don't want the agents to update the corresponding option q-functions.
    #     """
    #     agent1 = 0
    #     agent2 = 1
    #     agent3 = 2
    #
    #     avail_options = []
    #
    #     return avail_options
    #
    # def get_avail_meta_action_indeces(self, agent_id):
    #     """
    #     Get a list of the indeces corresponding to the currently available meta-action/option
    #     """
    #     avail_options = self.get_avail_options(agent_id)
    #     all_options_list = self.get_options_list(agent_id)
    #     avail_meta_action_indeces = []
    #     for option in avail_options:
    #         avail_meta_action_indeces.append(all_options_list.index(option))
    #     return avail_meta_action_indeces
    #
    # def get_completed_options(self, s):
    #     """
    #     Get a list of strings corresponding to options that are deemed complete in the team state described by s.
    #
    #     Parameters
    #     ----------
    #     s : numpy integer array
    #         Array of integers representing the environment states of the various agents.
    #         s[id] represents the state of the agent indexed by index "id".
    #
    #     Outputs
    #     -------
    #     completed_options : list
    #         list of strings corresponding to the completed options.
    #     """
    #     agent1 = 0
    #     agent2 = 1
    #     agent3 = 2
    #
    #     completed_options = []
    #
    #     for i in range(self.num_agents):
    #         row, col = self.state2pos(s[i])
    #
    #         if i == agent1:
    #             if (row, col) == self.env_settings['yellow_button']:
    #                 completed_options.append('by')
    #             if (row, col) == self.env_settings['goal_location']:
    #                 completed_options.append('g')
    #             if s[i] == self.env_settings['initial_states'][i]:
    #                 completed_options.append('w1')
    #
    #         elif i == agent2:
    #             if (row, col) == self.env_settings['green_button']:
    #                 completed_options.append('bg')
    #             if (row, col) == self.env_settings['red_button']:
    #                 completed_options.append('a2br')
    #             if s[i] == self.env_settings['initial_states'][i]:
    #                 completed_options.append('w2')
    #
    #         elif i == agent3:
    #             if (row, col) == self.env_settings['red_button']:
    #                 completed_options.append('a3br')
    #             if s[i] == self.env_settings['initial_states'][i]:
    #                 completed_options.append('w3')
    #
    #     return completed_options
    #
    # def get_meta_state(self, agent_id):
    #     """
    #     Return the meta-state that the agent should use for it's meta controller.
    #
    #     Input
    #     -----
    #     s_team : numpy array
    #         s_team[i] is the state of agent i.
    #     agent_id : int
    #         Index of agent whose meta-state is to be returned.
    #
    #     Output
    #     ------
    #     meta_state : int
    #         Index of the meta-state.
    #     """
    #     # Convert the Truth values of which buttons have been pushed to an int
    #     meta_state = int(
    #         '{}{}{}'.format(int(self.red_button_pushed), int(self.green_button_pushed), int(self.yellow_button_pushed)),
    #         2)
    #
    #     # # if the task has been failed, return 8
    #     # if self.u == 6:
    #     #     meta_state = 8
    #
    #     # meta_state = self.u
    #
    #     return meta_state
    #
    # def get_num_meta_states(self, agent_id):
    #     """
    #     Return the number of meta states for the agent specified by agent_id.
    #     """
    #     return int(8)  # how many different combinations of button presses are there

    ######################### TROUBLESHOOTING METHODS ################################
    def _get_map_str(self):
        r = ""
        agent = self.agents[0]
        for i in range(self.map_height):
            s = ""
            for j in range(self.map_width):
                if agent.idem_position(i, j):
                    s += str(agent)
                else:
                    s += str(self.map_array[i][j])
            if (i > 0):
                r += "\n"
            r += s
        return r

    def __str__(self):
        return self._get_map_str()

    def show(self):
        print(self.__str__())



def play():
    parentDir = os.path.abspath(os.path.join(os.getcwd(), os.pardir, os.pardir, os.pardir))
    rm_string = os.path.join(parentDir, 'reward_machines', 'minecraft', 'team_minecraft_rm.txt')
    env_settings = {'p': 0.98,
                    'file_map': os.path.abspath(os.path.join(os.getcwd(), 'maps', 'multiA_map_0.txt')),
                    'consider_night': False}
    game = MultiAgentMineCraftEnv(rm_string, env_settings)
    n = game.num_agents
    # User inputs
    str_to_action = {"w": Actions.up.value, "d": Actions.right.value, "s": Actions.down.value, "a": Actions.left.value,
                     "x": Actions.none.value}

    s = game.get_initial_team_state()
    print(s)

    while True:
        # Showing game
        game.show()

        # Getting action
        a = np.full(n, -1, dtype=int)

        for i in range(n):
            print('\nAction{}?'.format(i + 1), end='')
            usr_inp = input()

            if not (usr_inp in str_to_action):
                print('forbidden action')
                a[i] = str_to_action['x']
            else:
                # print(str_to_action[usr_inp])
                a[i] = str_to_action[usr_inp]

        r, l, s = game.environment_step(s, a)

        print("---------------------")
        print("Next States: ", s)
        print("Label: ", l)
        print("Reward: ", r)
        print("RM state: ", game.u)
        # print('Meta state: ', game.get_meta_state(0))
        print("---------------------")

        if game.reward_machine.is_terminal_state(game.u):  # Game Over
            break
    game.show()


# This code allow to play a game (for debugging purposes)
if __name__ == '__main__':
    play()
