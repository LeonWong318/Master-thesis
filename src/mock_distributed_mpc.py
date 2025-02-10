import os
import json
import pathlib
import asyncio

import numpy as np

from basic_motion_model.motion_model import UnicycleModel
from pkg_motion_plan.global_path_coordinate import GlobalPathCoordinator
from pkg_motion_plan.local_traj_plan import LocalTrajPlanner
from pkg_tracker_mpc.trajectory_tracker import TrajectoryTracker
from pkg_distributed_robot.robot import Robot
from pkg_distributed_robot.robot_manager import RobotManager
from pkg_distributed_robot.messages import NetworkDelay

from configs import MpcConfiguration, CircularRobotSpecification
from visualizer.object import CircularObjectVisualizer
from visualizer.mpc_plot import MpcPlotInLoop

class SimulationVisualizer:
    """仿真可视化管理器"""
    def __init__(self, config_robot):
        self.plotter = MpcPlotInLoop(config_robot)
        self.color_list = ["b", "r", "g"]
        
    def initialize(self, current_map, inflated_map, current_graph):
        self.plotter.plot_in_loop_pre(current_map, inflated_map, current_graph)
        
    def add_robot(self, robot, index):
        self.plotter.add_object_to_pre(
            robot.id_,
            robot.planner.ref_traj,
            robot.controller.state,
            robot.controller.final_goal,
            color=self.color_list[index % len(self.color_list)]
        )
        robot.visualizer.plot(self.plotter.map_ax, *robot.state)
        
    def update(self, simulation_result, kt, ts):
        robot_id = simulation_result.robot_id
        self.plotter.update_plot(
            robot_id, kt,
            simulation_result.actions[-1],
            simulation_result.state,
            simulation_result.debug_info['cost'],
            simulation_result.pred_states,
            simulation_result.current_refs
        )
        robot_visual = robot_visual = simulation_result.robot.visualizer
        if robot_visual:
            robot_visual.update(*simulation_result.state)
        
    def step(self, time, autorun=False, zoom_in=None):
        self.plotter.plot_in_loop(time=time, autorun=autorun, zoom_in=zoom_in)
        
    def show(self):
        self.plotter.show()
        
    def close(self):
        self.plotter.close()

async def main():
    # 配置路径
    ROOT_DIR = pathlib.Path(__file__).resolve().parents[1]
    DATA_DIR = os.path.join(ROOT_DIR, "data", "test_data")
    CNFG_DIR = os.path.join(ROOT_DIR, "config")
    VB = False
    TIMEOUT = 1000

    # 加载配置
    config_mpc = MpcConfiguration.from_yaml(os.path.join(CNFG_DIR, "mpc_default.yaml"))
    config_robot = CircularRobotSpecification.from_yaml(os.path.join(CNFG_DIR, "spec_robot.yaml"))

    # 加载地图和调度数据
    with open(os.path.join(DATA_DIR, "robot_start.json"), "r") as f:
        robot_starts = json.load(f)

    # 设置全局路径协调器
    gpc = GlobalPathCoordinator.from_csv(os.path.join(DATA_DIR, "schedule.csv"))
    gpc.load_graph_from_json(os.path.join(DATA_DIR, "graph.json"))
    gpc.load_map_from_json(os.path.join(DATA_DIR, "map.json"), 
                          inflation_margin=config_robot.vehicle_width+0.2)
    
    robot_ids = gpc.robot_ids
    static_obstacles = gpc.inflated_map.obstacle_coords_list

    # 设置网络延迟模拟
    network_delay = NetworkDelay(
        mean_delay=0.1,    # 100ms平均延迟
        std_delay=0.02,    # 20ms标准差
        min_delay=0.05,    # 最小50ms
        max_delay=0.2      # 最大200ms
    )

    # 创建管理器并启动
    robot_manager = RobotManager(network_delay)
    await robot_manager.start()

    # 创建可视化管理器
    visualizer = SimulationVisualizer(config_robot)
    visualizer.initialize(gpc.current_map, gpc.inflated_map, gpc.current_graph)

    # 创建并初始化机器人
    robots = []
    try:
        for i, rid in enumerate(robot_ids):
            # 创建机器人
            robot = Robot(config_robot, UnicycleModel(sampling_time=config_mpc.ts), rid)
            
            # 初始化组件
            planner = LocalTrajPlanner(config_mpc.ts, config_mpc.N_hor, 
                                     config_robot.lin_vel_max, verbose=VB)
            planner.load_map(gpc.inflated_map.boundary_coords, 
                            gpc.inflated_map.obstacle_coords_list)
            
            controller = TrajectoryTracker(config_mpc, config_robot, robot_id=rid, verbose=VB)
            controller.load_motion_model(UnicycleModel(sampling_time=config_mpc.ts))
            
            # 初始化状态
            initial_state = np.asarray(robot_starts[str(rid)])
            controller.load_init_states(initial_state, initial_state)
            
            vis = CircularObjectVisualizer(config_robot.vehicle_width, indicate_angle=True)
            
            # 初始化机器人
            robot.initialize(controller, planner, vis)
            robot.set_state(initial_state)
            
            # 加载路径和设置目标
            path_coords, path_times = gpc.get_robot_schedule(rid)
            robot.load_schedule(path_coords, path_times)
            
            goal_coord = path_coords[-1]
            goal_coord_prev = path_coords[-2]
            goal_heading = np.arctan2(goal_coord[1]-goal_coord_prev[1], 
                                    goal_coord[0]-goal_coord_prev[0])
            goal_state = np.array([*goal_coord, goal_heading])
            controller.load_init_states(initial_state, goal_state)
            controller.set_work_mode('safe', use_predefined_speed=True)
            
            # 启动机器人并订阅到管理器
            await robot.start()
            await robot.subscribe(robot_manager)
            robots.append(robot)
            visualizer.add_robot(robot, i)

        # 主循环
        for kt in range(TIMEOUT):
            # 执行仿真步骤
            results = await robot_manager.simulate_step(kt, config_mpc, static_obstacles)
            
            # 更新可视化
            for result in results:
                visualizer.update(result, kt, config_mpc.ts)
            
            # 步进可视化
            visualizer.step(time=kt*config_mpc.ts)
            
            # 检查是否所有机器人都完成了任务
            if robot_manager._all_complete:
                break

        # 显示最终结果
        visualizer.show()
        input('Press anything to finish!')
        
    finally:
        # 清理资源
        visualizer.close()
        
        # 停止所有机器人和管理器
        await asyncio.gather(*[robot.stop() for robot in robots])
        await robot_manager.stop()

if __name__ == "__main__":
    asyncio.run(main())