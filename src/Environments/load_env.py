from Environments.buttons.buttons_env import ButtonsEnv
from Environments.buttons.multi_agent_buttons_env import MultiAgentButtonsEnv
# from src.Environments.rendezvous.gridworld_env import GridWorldEnv
# from src.Environments.rendezvous.multi_agent_gridworld_env import MultiAgentGridWorldEnv
from Environments.minecraft.minecraft_env import MultiAgentMineCraftEnv
from Environments.minecraft2.minecraft2_env import MultiAgentMineCraft2Env
from Environments.pass_room.pass_room_env import PassRoomEnv


def load_testing_env(tester):
    if tester.experiment == 'rendezvous':
        # testing_env = MultiAgentGridWorldEnv(tester.rm_test_file, tester.env_settings)
        raise ValueError('No such environment: ' + tester.experiment)
    elif tester.experiment == 'buttons':
        testing_env = MultiAgentButtonsEnv(tester.rm_test_file, tester.env_settings)
    elif tester.experiment == 'minecraft':
        testing_env = MultiAgentMineCraftEnv(tester.rm_test_file, tester.env_settings)
    elif tester.experiment == 'minecraft2':
        testing_env = MultiAgentMineCraft2Env(tester.rm_test_file, tester.env_settings)
    elif tester.experiment == 'pass_room':
        testing_env = PassRoomEnv(tester.rm_test_file, tester.env_settings)
    else:
        raise ValueError('No such environment: ' + tester.experiment)

    return testing_env


# def load_agent_envs(tester, agent_list):
#     num_agents = len(agent_list)
#     if tester.experiment == 'rendezvous':
#         training_environments = []
#         for i in range(num_agents):
#             training_environments.append(GridWorldEnv(agent_list[i].rm_file, i + 1, tester.env_settings))
#     elif tester.experiment == 'buttons':
#         training_environments = []
#         for i in range(num_agents):
#             training_environments.append(ButtonsEnv(agent_list[i].rm_file, i + 1, tester.env_settings))
#     elif tester.experiment == 'minecraft':
#         training_environments = []
#         for i in range(num_agents):
#             training_environments.append(MultiAgentMineCraftEnv(agent_list[i].rm_file, tester.env_settings))
#     else:
#         raise ValueError('No such environment: ' + tester.experiment)
#     return training_environments
