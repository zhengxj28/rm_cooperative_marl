cd src
nohup python -u run.py --env_name "minecraft2" --map_name "nav_map2" --task_name "navigation" --algorithm "mahrm" --use_wandb --seeds 0 >nohup.log 2>&1 &
nohup python -u run.py --env_name "minecraft2" --map_name "nav_map2" --task_name "navigation" --algorithm "dqprm" --use_wandb --seeds 0 >nohup.log 2>&1 &
nohup python -u run.py --env_name "minecraft2" --map_name "nav_map2" --task_name "navigation" --algorithm "iqrm" --use_wandb --seeds 0 >nohup.log 2>&1 &
nohup python -u run.py --env_name "minecraft2" --map_name "nav_map2" --task_name "navigation" --algorithm "modular" --use_wandb --seeds 0 >nohup.log 2>&1 &2