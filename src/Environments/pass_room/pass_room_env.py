import random, math, os
import numpy as np
from enum import Enum

import sys, copy

# sys.path.append('../')
# sys.path.append('../../')
from reward_machines.sparse_reward_machine import SparseRewardMachine

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


class Door(Entity):
    def __init__(self, i, j):
        super().__init__(i, j)
        self.is_open = False

    def interact(self, agent):
        return self.is_open
        # return True

    def open(self):
        self.is_open = True

    def close(self):
        self.is_open = False

    def __str__(self):
        return "D"


# class Button(Entity):
#     def __init__(self, i, j, label):
#         super().__init__(i, j)
#         self.label = label
#         self.is_pressed = False
#
#     def press_button(self):
#         self.is_pressed = True
#
#     def release_button(self):
#         self.is_pressed = False
#
#     def __str__(self):
#         return self.label

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
# class MineCraftParams:
#     def __init__(self, file_map, consider_night):
#         self.file_map = file_map
#         self.consider_night = consider_night

class PassRoomEnv:

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
        self.nSteps = 0

        self._load_map(self.env_settings['file_map'])
        self.s = self.s_i  # current states are initialized
        self.reward_machine = SparseRewardMachine(rm_file)

        self.u = self.reward_machine.get_initial_state()
        self.last_action = np.full(self.num_agents, -1, dtype=int)  # Initialize last action with garbage values
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
        self.map = []  # a list of entity
        # I use the lower case letters to define the features
        self.class_ids = {}
        self.agents = dict()
        self.button_chars = set()  # characters on the map, e.g. 'a','b'
        room_buttons = set()  # characters on the goal room
        self.door_list = []  # entity of door(s)

        f = open(file_map)
        i, j = 0, 0
        ag_id = 0  # id of agents
        door_pos_list = []
        for l in f:
            # I don't consider empty lines!
            if (len(l.rstrip()) == 0): continue
            # this is not an empty line!
            row = []
            j = 0
            for e in l.rstrip():
                if e in "abcdefghijklmnopq":
                    entity = Empty(i, j, label=e)
                    self.button_chars.add(e)
                    if e not in self.class_ids:
                        self.class_ids[e] = len(self.class_ids)
                    if self._is_goal_room(i, j):
                        room_buttons.add(e)
                # we need to declare the initial positions of agents
                # to be potentially empty espaces (after they moved)
                elif e == "A" or e == " ":
                    entity = Empty(i, j)
                elif e == "X":
                    entity = Obstacle(i, j)
                elif e == "D":
                    entity = Door(i, j)
                    door_pos_list.append((i, j))
                else:
                    raise ValueError('Unkown entity ', e)
                if e == "A":
                    self.agents[ag_id] = Agent(i, j, actions)
                    ag_init_pos.append((i, j))
                    if e not in self.class_ids:
                        self.class_ids[e] = len(self.class_ids)
                    ag_id += 1
                row.append(entity)
                j += 1
            self.map.append(row)
            i += 1
        """
        We use this back map to check what was there when an agent leaves a 
        position
        """
        self.back_map = copy.deepcopy(self.map)  # background, do not include agent
        for door_i, door_j in door_pos_list:
            self.door_list.append(self.back_map[door_i][door_j])
        for agent in self.agents.values():
            i, j = agent.i, agent.j
            self.map[i][j] = agent
        f.close()

        # height width
        self.map_height, self.map_width = len(self.map), \
                                          len(self.map[0])
        self.num_agents = len(self.agents)
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

        # generate event_set_of_agent[ag_id]: possible events of an agent
        # each event is represented as a tuple
        self.event_set_of_agents = dict()
        for ag_id in range(self.num_agents):
            event_set_of_ag_i = set()
            event_set_of_ag_i.add(tuple())  # empty event, represented as a tuple
            for p in self.button_chars:
                p_ag = p + str(ag_id + 1)
                if p in room_buttons:  # buttons in goal room
                    event_set_of_ag_i.add((p_ag, 'r' + str(ag_id + 1),))
                else:
                    event_set_of_ag_i.add((p_ag,))  # in this env, each agent cover at most one character
            event_set_of_ag_i.add(('r' + str(ag_id + 1),))

            self.event_set_of_agents[ag_id] = event_set_of_ag_i

    def environment_step(self, a):
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

        s_next = np.full(self.num_agents, -1, dtype=int)
        for i in range(self.num_agents):
            s_next[i], actual_action = self.get_next_state(a[i], i)
            self.last_action[i] = actual_action

        l = self.get_mdp_label(self.s, s_next, self.u)
        self.s = s_next  # update env states

        u2 = self.reward_machine.get_next_state(self.u, l)
        r = self.reward_machine.get_reward(self.u, u2)
        self.u = u2

        if len(set(l).intersection(self.button_chars)) >= self.num_agents - 1:  # two button pressed
            for door in self.door_list:
                door.open()
        else:
            for door in self.door_list:
                door.close()

        return r, l, s_next

    def is_terminal(self):
        return self.reward_machine.is_terminal_state(self.u)

    def get_state(self):
        return self.s

    def get_next_state(self, a, agent_id):  # get next state for single agent
        """
         Get the next state in the environment given action a is taken from state s.
         Update the last action that was truly taken due to MDP slip.

         Parameters
         ----------
         a : int
             Action to be taken from state s.

         Outputs
         -------
         s_next : int
             Index of the next state.
         last_action : int
             Last action the agent truly took because of slip probability.
         """

        s = self.s[agent_id]
        # action = Actions(a)  # action = up, right, down, left, none
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

        # entity_old = self.back_map[row_old][col_old]  # reference, not deep copy
        entity = self.back_map[row][col]
        if entity.interact(agent_id):  # not obstacle
            s_next = self.pos2state(row, col)
            self.agents[agent_id].change_position(row, col)
            self.map[row_old][col_old] = self.back_map[row_old][col_old]
            self.map[row][col] = self.agents[agent_id]
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
        an example of map_width=5 
        01234
        56789
        .....
        """
        row = np.floor_divide(s, self.map_width)
        col = np.mod(s, self.map_width)

        return row, col

    def _is_goal_room(self, row, col):
        return col > 11

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

    ############## RM-RELATED METHODS ########################################
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

        event_set = set()  # event set without agent id, e.g. {'', 'a','b'}
        event_set_ag = set()  # event set with agent id, e.g. {'a1', 'b2'}
        for i in range(self.num_agents):
            row_i, col_i = self.state2pos(s_next[i])  # position of agent i
            proposition = str(self.back_map[row_i][col_i])
            if proposition == 'A' or proposition == ' ':
                pass
            else:
                event_set.add(proposition)
                proposition = proposition + str(i + 1)  # add subscript
                event_set_ag.add(proposition)
            if self._is_goal_room(row_i, col_i):  # reach the goal room
                event_set_ag.add('r' + str(i + 1))
        return list(event_set) + list(event_set_ag)
        # return list(event_set_ag)

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
                    s += str(self.map[i][j])
            if (i > 0):
                r += "\n"
            r += s
        return r

    def __str__(self):
        return self._get_map_str()

    def show(self):
        print(self.__str__())


def play(map_name, task_name):
    parentDir = os.path.abspath(os.path.join(os.getcwd(), os.pardir, os.pardir, os.pardir))
    rm_string = os.path.join(parentDir, 'reward_machines', 'pass_room', map_name, task_name + 'team.txt')
    env_settings = {'p': 0.98,
                    'file_map': os.path.abspath(os.path.join(os.getcwd(), 'maps', map_name + '.txt'))}
    game = PassRoomEnv(rm_string, env_settings)
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
            print('Action{}?'.format(i + 1), end='')
            usr_inp = input()

            if not (usr_inp in str_to_action):
                print('forbidden action')
                a[i] = str_to_action['x']
            else:
                # print(str_to_action[usr_inp])
                a[i] = str_to_action[usr_inp]

        r, l, s = game.environment_step(a)

        print("---------------------")
        print("Next States: ", s, " Agent 1:", game.state2pos(s[0]))
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
    map_name = '8button5agent'
    task_name = 'pass'
    play(map_name, task_name)
