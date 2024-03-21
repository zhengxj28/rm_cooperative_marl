cd src
nohup python -u run.py --env_name "minecraft2" --map_name "2A_map_0" --task_name "task1" --algorithm "mahrm" --option_elimination --use_wandb --seeds 0 1 2 3 >mahrm.log 2>&1 &
nohup python -u run.py --env_name "minecraft2" --map_name "2A_map_0" --task_name "task1" --algorithm "dqprm" --use_wandb --seeds 0 1 2 3 >dqprm.log 2>&1 &
nohup python -u run.py --env_name "minecraft2" --map_name "2A_map_0" --task_name "task1" --algorithm "iqrm" --use_wandb --seeds 0 1 2 3 >iqrm.log 2>&1 &
nohup python -u run.py --env_name "minecraft2" --map_name "2A_map_0" --task_name "task1" --algorithm "modular" --use_wandb --seeds 0 1 2 3 >modular.log 2>&1 &
