cd src
nohup python -u run.py --env_name "minecraft2" --map_name "3A_map_0" --task_name "task3" --algorithm "hie_iqrm2" --use_wandb --seeds 0 >mahrm.log 2>&1 &
nohup python -u run.py --env_name "minecraft2" --map_name "3A_map_0" --task_name "task3" --algorithm "dqprm" --use_wandb --seeds 0 >mahrm.log 2>&1 &
nohup python -u run.py --env_name "minecraft2" --map_name "3A_map_0" --task_name "task3" --algorithm "iqrm" --use_wandb --seeds 0 >mahrm.log 2>&1 &
nohup python -u run.py --env_name "minecraft2" --map_name "3A_map_0" --task_name "task3" --algorithm "modular" --use_wandb --seeds 0 >mahrm.log 2>&1 &
