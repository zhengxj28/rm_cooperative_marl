cd src
nohup python -u run.py --env_name "pass_room" --map_name "4button3agent" --task_name "pass3" --algorithm "mahrm" --use_wandb --seeds 0 >mahrm.log 2>&1 &
nohup python -u run.py --env_name "pass_room" --map_name "4button3agent" --task_name "pass3" --algorithm "dqprm" --use_wandb --seeds 0 >mahrm.log 2>&1 &
nohup python -u run.py --env_name "pass_room" --map_name "4button3agent" --task_name "pass3" --algorithm "iqrm" --use_wandb --seeds 0 >mahrm.log 2>&1 &
nohup python -u run.py --env_name "pass_room" --map_name "4button3agent" --task_name "pass3" --algorithm "modular" --use_wandb --seeds 0 >mahrm.log 2>&1 &
nohup python -u run.py --env_name "pass_room" --map_name "4button3agent" --task_name "pass" --algorithm "mahrm_3L" --use_wandb --seeds 0 >mahrm.log 2>&1 &