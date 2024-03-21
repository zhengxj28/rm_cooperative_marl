cd src
#nohup python -u run.py --env_name "minecraft2" --map_name "nav_map1" --task_name "navigation" --algorithm "mahrm" --wandb_name "mahrm-noe" --use_wandb --seeds 0 1 2 3 >ablation.log 2>&1 &
#nohup python -u run.py --env_name "minecraft2" --map_name "nav_map2" --task_name "navigation" --algorithm "mahrm" --wandb_name "mahrm-noe" --use_wandb --seeds 0 1 2 3 >ablation.log 2>&1 &
nohup python -u run.py --env_name "minecraft2" --map_name "nav_map5" --task_name "navigation" --algorithm "mahrm" --wandb_name "mahrm-noe" --use_wandb --seeds 0 1 2 3 >ablation.log 2>&1 &
nohup python -u run.py --env_name "minecraft2" --map_name "2A_map_0" --task_name "task2" --algorithm "mahrm" --wandb_name "mahrm-noe" --use_wandb --seeds 0 1 2 3 >ablation.log 2>&1 &
nohup python -u run.py --env_name "minecraft2" --map_name "2A_map_0" --task_name "task3" --algorithm "mahrm" --wandb_name "mahrm-noe" --use_wandb --seeds 0 1 2 3 >ablation.log 2>&1 &
nohup python -u run.py --env_name "minecraft2" --map_name "3A_map_0" --task_name "task3" --algorithm "mahrm" --wandb_name "mahrm-noe" --use_wandb --seeds 0 1 2 3 >ablation.log 2>&1 &
#nohup python -u run.py --env_name "pass_room" --map_name "2button2agent" --task_name "pass3" --algorithm "mahrm" --wandb_name "mahrm-noe" --use_wandb --seeds 0 1 2 3 >ablation.log 2>&1 &
#nohup python -u run.py --env_name "pass_room" --map_name "4button3agent" --task_name "pass3" --algorithm "mahrm" --wandb_name "mahrm-noe" --use_wandb --seeds 0 1 2 3 >ablation.log 2>&1 &

