class SparseRewardMachine:
    def __init__(self, file=None, paras=None):
        # <U,u0,delta_u,delta_r>
        self.U = []  # list of machine states
        self.propositions = set()  # set of propositions
        self.event_set = set()  # set of events, each event is a subset of propositions
        self.u0 = None  # initial state
        self.delta_u = {}  # state-transition function
        self.delta_r = {}  # reward-transition function
        self.T = set()  # set of terminal states (they are automatically detected)
        self.tag = ''  # for trouble shooting, the name of this rm
        if paras is not None:  # paras of RM
            self.paras = paras  # type(paras)==list
            self.num_paras = len(paras)
        if file is not None:
            self._load_reward_machine(file)

    def __repr__(self):
        s = "MACHINE:\n"
        s += "init: {}\n".format(self.u0)
        for trans_init_state in self.delta_u:
            for event in self.delta_u[trans_init_state]:
                trans_end_state = self.delta_u[trans_init_state][event]
                s += '({} ---({},{})--->{})\n'.format(trans_init_state,
                                                      event,
                                                      self.delta_r[trans_init_state][trans_end_state],
                                                      trans_end_state)
        return s

    # Public methods -----------------------------------

    def get_initial_state(self):
        return self.u0

    def get_next_state(self, u1, event):
        # type(event)==list or str, should be converted to tuple
        if type(event) == str:
            event = (event,)
        elif len(event) > 1:
            event = list(set(event) & self.propositions)  # only consider related propositions
            event.sort()
        event = tuple(event)
        if u1 in self.delta_u:
            if event in self.delta_u[u1]:
                return self.delta_u[u1][event]
        # default transition
        return u1

    def get_reward(self, u1, u2, s1=None, a=None, s2=None):
        if u1 in self.delta_r and u2 in self.delta_r[u1]:
            return self.delta_r[u1][u2]
        return 0  # This case occurs when the agent falls from the reward machine

    def get_rewards_and_next_states(self, s1, a, s2, event):
        rewards = []
        next_states = []
        for u1 in self.U:
            u2 = self.get_next_state(u1, event)
            rewards.append(self.get_reward(u1, u2, s1, a, s2))
            next_states.append(u2)
        return rewards, next_states

    def get_states(self):
        return self.U

    def is_terminal_state(self, u1):
        return u1 in self.T

    def get_propositions(self):
        return self.propositions

    def is_event_available(self, u, event):
        is_event_available = False
        if u in self.delta_u:
            if event in self.delta_u[u]:
                is_event_available = True
        return is_event_available

    # Private methods -----------------------------------

    def _load_reward_machine(self, file):  # load rm from file
        """
        Example:
            0 # initial state
            (0, 1, 'a1', 0)
            (0, 2, 'f1', 0)
            (0, 1, 'a2', 0)
            (0, 2, 'f2', 0)
            (0, 3, ('a1','f1'), 1)

            Format: (current state, next state, event, reward)
        """
        # Reading the file
        f = open(file)
        lines = [l.rstrip() for l in f]
        f.close()

        # setting the DFA
        use_paras = False
        e = eval(lines[0])
        if type(e)==int:
            self.u0 = e
        elif type(e)==tuple:
            self.u0 = e[0]
            variables = e[1]  # example: ['i','j','k']
            use_paras = True
            if len(variables) != self.num_paras:
                raise ValueError('The number of variables not equals to parameters.')
        else:
            raise Exception('Invalid RM file: ' + file)

        # adding transitions
        replace_dict = dict()
        if use_paras:
            """
            replace variables with paras
            Example: variables=('i','j'), self.paras=['2','1'], then
            ('ai','bj') --> ('a2','b1')
            """
            for i in range(self.num_paras):
                replace_dict[variables[i]] = self.paras[i]
        for e in lines[1:]:
            event = self._add_transition(*eval(e.translate(str.maketrans(replace_dict))))  # event: sorted tuple
            self.propositions = self.propositions.union(set(event))
            self.event_set.add(event)
        # adding terminal states
        for u1 in self.U:
            if self._is_terminal(u1):
                self.T.add(u1)
        self.U = sorted(self.U)
        self.tag = file

    def build_atom_rm(self, event, propositions, event_set):
        self.propositions = propositions
        for e in event_set:
            if type(e) == str:
                self.event_set.add((e, ))
            else:
                e_sort = list(set(e))
                e_sort = sorted(e_sort)
                self.event_set.add(tuple(e_sort))
        self.u0 = 0
        event = self._add_transition(0, 1, event, 1)
        for event_ in self.event_set.difference({event,}):
            self._add_transition(1, 0, event_, -1)
        self.tag = str(event)

    def calculate_reward(self, trace):
        total_reward = 0
        current_state = self.get_initial_state()

        for event in trace:
            next_state = self.get_next_state(current_state, event)
            reward = self.get_reward(current_state, next_state)
            total_reward += reward
            current_state = next_state
        return total_reward

    def _is_terminal(self, u1):  # terminal iff r=1
        # Check if reward is given for reaching the state in question
        for u0 in self.delta_r:
            if u1 in self.delta_r[u0]:
                if self.delta_r[u0][u1] == 1:
                    return True
        return False

    def _add_state(self, u_list):
        for u in u_list:
            if u not in self.U:
                self.U.append(u)

    def _add_transition(self, u1, u2, event, reward):
        # Adding machine state
        self._add_state([u1, u2])
        # Adding state-transition to delta_u
        if u1 not in self.delta_u:
            self.delta_u[u1] = {}
        if type(event) == str:
            event = (event,)
        else:
            event = list(set(event))
            event.sort()
            event = tuple(event)
        if event not in self.delta_u[u1]:
            self.delta_u[u1][event] = u2
        elif self.delta_u[u1][event] != u2:
            raise Exception('Trying to make rm transition function non-deterministic.')
        # Adding reward-transition to delta_r
        if u1 not in self.delta_r:
            self.delta_r[u1] = {}
        self.delta_r[u1][u2] = reward
        return event  # return normal form: sorted tuple


if __name__ == '__main__':  # for debug only
    import os

    base_file_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir, os.pardir))
    # rm_file1 = os.path.join(base_file_path, 'reward_machines', 'buttons', 'team_buttons_rm.txt')
    # rm_file2 = os.path.join(base_file_path, 'reward_machines', 'buttons', 'buttons_rm_agent_1.txt')
    # test_rm1 = SparseRewardMachine(rm_file1)
    # test_rm2 = SparseRewardMachine(rm_file2)
    #
    # print(test_rm1)

    # print(test_rm1.get_next_state(0, 'by'))
    # print(test_rm1.get_next_state(3, ['a3br', 'a2lr']))
    # print(test_rm1.get_next_state(2, ['a3br', 'a2br']))
    # print(test_rm2.get_next_state(2, ['a3br', 'a2br']))
    # print(test_rm2.get_next_state(2, 'g'))

    # rm_file1 = os.path.join(base_file_path, 'reward_machines', 'minecraft2', 'multiA_map_0', 'task2team.txt')
    # test_rm = SparseRewardMachine(rm_file1)
    # atom_rm = SparseRewardMachine()
    # atom_rm.build_atom_rm(('c1', 'r1'),
    #                       propositions={'a1','b1','c1','d1','r1'},
    #                       event_set={'1','a1','b1','c1','d1','r1',('c1','r1'), ('d1','r1')})

    rm_file1 = os.path.join(base_file_path, 'reward_machines', 'pass_room', '4button3agent', 'passL1ab_c_a.txt')
    test_rm = SparseRewardMachine(rm_file1, ['1','3','2'])

    print(test_rm)
