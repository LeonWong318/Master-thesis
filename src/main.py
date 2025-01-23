import os
import pathlib

import numpy as np
import matplotlib.pyplot as plt

### GYM import
import gym
from stable_baselines3.common import env_checker
from pkg_dqn.environment.environment import TrajectoryPlannerEnvironment

### MPC import
from interface_mpc import InterfaceMpc
from util.mpc_config import Configurator

### Helper
from main_pre import generate_map, get_geometric_map

### Others
from timer import PieceTimer, LoopTimer
from typing import List, Tuple

MAX_RUN_STEP = 200
DYN_OBS_SIZE = 0.8 + 0.8

def load_mpc(config_path: str):
    config = Configurator(config_path)
    traj_gen = InterfaceMpc(config, motion_model=None) # default motion model is used
    return traj_gen

def est_dyn_obs_positions(last_pos: list, current_pos: list, steps:int=20):
    """
    Estimate the dynamic obstacle positions in the future.
    """
    est_pos = []
    d_pos = [current_pos[0]-last_pos[0], current_pos[1]-last_pos[1]]
    for i in range(steps):
        est_pos.append([current_pos[0]+d_pos[0]*(i+1), current_pos[1]+d_pos[1]*(i+1), DYN_OBS_SIZE, DYN_OBS_SIZE, 0, 1])
    return est_pos

def circle_to_rect(pos: list, radius:float=DYN_OBS_SIZE):
    """
    Convert the circle to a rectangle.
    """
    return [[pos[0]-radius, pos[1]-radius], [pos[0]+radius, pos[1]-radius], [pos[0]+radius, pos[1]+radius], [pos[0]-radius, pos[1]+radius]]

def load_simulation_env(generate_map) -> TrajectoryPlannerEnvironment:
    env_name = 'TrajectoryPlannerEnvironmentRaysReward1-v0'
    # env_name = 'TrajectoryPlannerEnvironmentImgsReward1-v0'

    env_eval: TrajectoryPlannerEnvironment = gym.make(env_name, generate_map=generate_map)
    
    env_checker.check_env(env_eval)
    
    return env_eval

def predict_dynamic_obstacles(dyn_obstacle_list, last_dyn_obstacle_list):
    if last_dyn_obstacle_list is None:
        return [], dyn_obstacle_list
    dyn_obstacle_pred_list = [
        est_dyn_obs_positions(last_dyn_obstacle_list[j], dyn_obs)
        for j, dyn_obs in enumerate(dyn_obstacle_list)
    ]
    return dyn_obstacle_pred_list, dyn_obstacle_list



def main(to_plot=False, scene_option:Tuple[int, int, int]=(1, 1, 1), save_num:int=1):
    time_list = []

    env_eval = load_simulation_env(generate_map(*scene_option))


    CONFIG_FN = 'mpc_longiter.yaml'
    cfg_fpath = os.path.join(pathlib.Path(__file__).resolve().parents[1], 'config', CONFIG_FN)
    traj_gen = load_mpc(cfg_fpath)
    geo_map = get_geometric_map(env_eval.get_map_description(), inflate_margin=0.8)
    traj_gen.update_static_constraints(geo_map.processed_obstacle_list) # if assuming static obstacles not changed

    done = False
    while not done:
        init_state = np.array([*env_eval.agent.position, env_eval.agent.angle])
        goal_state = np.array([*env_eval.goal.position, 0])
        ref_path = list(env_eval.path.coords)
        traj_gen.initialization(init_state, goal_state, ref_path)
        last_mpc_time = 0.0
        chosen_ref_traj = None
        last_dyn_obstacle_list = None      
        for i in range(0, MAX_RUN_STEP):
            
            dyn_obstacle_list = [obs.keyframe.position.tolist() for obs in env_eval.obstacles if not obs.is_static]
            dyn_obstacle_pred_list = []
            dyn_obstacle_pred_list, last_dyn_obstacle_list = predict_dynamic_obstacles(dyn_obstacle_list, last_dyn_obstacle_list)

            env_eval.set_agent_state(traj_gen.state[:2], traj_gen.state[2], 
                                     traj_gen.last_action[0], traj_gen.last_action[1])
            observation, reward, done, info = env_eval.step(0) # just for plotting and updating status
            if dyn_obstacle_list:
                traj_gen.update_dynamic_constraints(dyn_obstacle_pred_list)
            original_ref_traj, local_ref_traj, extra_ref_traj= traj_gen.get_local_ref_traj()
            chosen_ref_traj = original_ref_traj
            timer_mpc = PieceTimer()
            try:
                mpc_output = traj_gen.get_action(chosen_ref_traj)
            except Exception as e:
                done = True
                print(f'MPC fails: {e}')
                break
            last_mpc_time = timer_mpc(4, ms=True)
            if mpc_output is None:
                break
            
            time_list.append(last_mpc_time)
            if to_plot:
                print(f"Step {i}.Runtime (MPC): {last_mpc_time}ms")
            if to_plot & (i%1==0): # render
                env_eval.render(dqn_ref=None, actual_ref=chosen_ref_traj, original_ref=original_ref_traj, save=True, save_num=save_num)
            if i == MAX_RUN_STEP - 1:
                done = True
                print('Time out!')
            if done:
                if to_plot:
                    input(f"Finish (Succeed: {info['success']})! Press enter to continue...")
                break
    return time_list

if __name__ == '__main__':
    scene_option = (1, 4, 1)

    time_list_mpc     = main(to_plot=False, scene_option=scene_option, save_num=1)

    print(f"Average time (MPC): {np.mean(time_list_mpc)}ms")

    fig, axes = plt.subplots(1,2)

    bin_list = np.arange(0, 150, 10)
    axes[0].hist(time_list_mpc, bins=bin_list, color='b', alpha=0.5, label='MPC')
    axes[0].legend()
    axes[1].plot(time_list_mpc, color='b', ls='-', marker='x', label='MPC')

    plt.show()
    input('Press enter to exit...')