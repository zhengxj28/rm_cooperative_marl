cd src
nohup python -u run.py --env_name "pass_room" --map_name "2button2agent" --task_name "pass3" --algorithm "mahrm" --use_wandb --seeds 0 1 2 3 >mahrm.log 2>&1 &
nohup python -u run.py --env_name "pass_room" --map_name "2button2agent" --task_name "pass3" --algorithm "dqprm" --use_wandb --seeds 0 1 2 3 >dqprm.log 2>&1 &
nohup python -u run.py --env_name "pass_room" --map_name "2button2agent" --task_name "pass3" --algorithm "iqrm" --use_wandb --seeds 0 1 2 3 >iqrm.log 2>&1 &
nohup python -u run.py --env_name "pass_room" --map_name "2button2agent" --task_name "pass3" --algorithm "modular" --use_wandb --seeds 0 1 2 3 >modular.log 2>&1 &
