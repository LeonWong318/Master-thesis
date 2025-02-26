from typing import Any, Dict, List, Optional
import numpy as np
import asyncio

from .messages import (
    Message, MessageType, Communication, NetworkDelay,
    SimulationParams, SimulationResult, RobotState
)

class RobotManager:
    """机器人管理器类"""
    def __init__(self, network_delay: Optional[NetworkDelay] = None):
        self._robots: Dict[int, Communication] = {}
        self._robot_states: Dict[int, SimulationResult] = {}
        self._step_results: Dict[int, SimulationResult] = {}
        self._step_complete_count: int = 0
        self._all_complete: bool = False
        self._running = False
        self._message_task = None
        self.network = network_delay or NetworkDelay()
        
        pass

    async def start(self):
        """启动管理器"""
        self._running = True
        self._message_task = asyncio.create_task(self._run_message_loop())
        print('RobotManger initialized')


    async def stop(self):
        """停止管理器"""
        self._running = False
        if self._message_task:
            await self._message_task
            self._message_task = None
        
        # 清理资源
        self._robots.clear()
        self._robot_states.clear()
        self._step_results.clear()

    async def _run_message_loop(self):
        """消息处理主循环"""
        while self._running:
            for robot_comm in list(self._robots.values()):
                try:
                    if not robot_comm.outbox.empty():
                        message = await robot_comm.outbox.get()
                        if message is None:  # 消息可能因为网络延迟而丢失
                            continue
                        handler = self._message_handlers.get(message.msg_type)
                        if handler:
                            await handler(message)
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    print(f"Error handling message: {e}")
            
            # 添加短暂延迟避免CPU占用过高
            await asyncio.sleep(0.001)

    def get_robot_state(self, robot_id: int) -> np.ndarray:
        """获取机器人状态"""
        self._check_id(robot_id)
        return self._robot_states[robot_id].state

    def get_pred_states(self, robot_id: int) -> Optional[np.ndarray]:
        """获取机器人预测状态"""
        self._check_id(robot_id)
        return self._robot_states[robot_id].pred_states

    def _check_id(self, robot_id: int) -> None:
        """检查机器人ID是否存在"""
        if robot_id not in self._robot_states:
            raise ValueError(f'Robot {robot_id} does not exist!')

    def get_other_robot_states(self, ego_robot_id: int, config_mpc: Any, default: float = -10.0) -> list:
        """获取其他机器人状态"""
        state_dim = config_mpc.ns
        horizon = config_mpc.N_hor
        num_others = config_mpc.Nother
        
        # 初始化状态列表
        other_robot_states = [default] * state_dim * (horizon+1) * num_others
        
        # 使用独立的索引追踪当前状态和预测状态的位置
        idx = 0                        # 当前状态的索引
        idx_pred = state_dim * num_others  # 预测状态的起始索引

        # 遍历其他机器人
        for rid, result in self._robot_states.items():
            if rid != ego_robot_id:
                # 添加当前状态
                current_state = result.state
                if isinstance(current_state, np.ndarray):
                    current_state = current_state.tolist()
                other_robot_states[idx : idx+state_dim] = current_state
                idx += state_dim

                # 添加预测状态
                pred_states = result.pred_states
                if pred_states is not None:
                    # 确保 pred_states 是列表形式
                    if isinstance(pred_states, np.ndarray):
                        pred_states = pred_states.tolist()
                    # 展平预测状态
                    pred_flat = []
                    for state in pred_states:
                        if isinstance(state, np.ndarray):
                            pred_flat.extend(state.tolist())
                        else:
                            pred_flat.extend(state)
                            
                    # 确保长度正确
                    pred_len = state_dim * horizon
                    if len(pred_flat) > pred_len:
                        pred_flat = pred_flat[:pred_len]
                    elif len(pred_flat) < pred_len:
                        last_state = pred_flat[-state_dim:] if pred_flat else [default] * state_dim
                        while len(pred_flat) < pred_len:
                            pred_flat.extend(last_state)
                            
                    other_robot_states[idx_pred : idx_pred+pred_len] = pred_flat
                    idx_pred += state_dim * horizon

        return other_robot_states

    async def simulate_step(self, kt: int, config_mpc: Any, static_obstacles: Any) -> List[SimulationResult]:
        """执行一步仿真"""
        # 重置步骤计数器和结果存储
        self._step_complete_count = 0
        self._step_results.clear()
        
        # 准备仿真参数
        params = SimulationParams(
            kt=kt,
            ts=config_mpc.ts,
            current_time=kt * config_mpc.ts,
            config_mpc=config_mpc,
            static_obstacles=static_obstacles,
            other_robot_states=self._get_all_robot_states()
        )
        
        # 向所有机器人发送计算请求
        compute_tasks = []
        for robot_comm in self._robots.values():
            compute_tasks.append(
                robot_comm.inbox.put(Message(
                    MessageType.COMPUTE_REQUEST,
                    -1,  # manager ID
                    params
                ))
            )
        
        # 等待所有请求发送完成
        await asyncio.gather(*compute_tasks)
        
        # 等待所有机器人完成计算和状态更新
        while self._step_complete_count < len(self._robots):
            await asyncio.sleep(0.01)
        
        # 返回所有结果
        return list(self._step_results.values())

    def _get_all_robot_states(self) -> List[RobotState]:
        """获取所有机器人的当前状态"""
        states = []
        for result in self._robot_states.values():
            states.append(RobotState(
                position=result.state,
                predicted_states=result.pred_states,
                ref_traj=result.current_refs,
                ref_speed=result.traj_result.ref_speed if result.traj_result else 0.0,
                timestamp=result.timestamp,
                is_idle=result.traj_result.is_complete if result.traj_result else False
            ))
        return states

    async def _handle_registration(self, msg: Message):
        """处理注册消息"""
        robot_id = msg.sender_id
        robot_comm = msg.data
        self._robots[robot_id] = robot_comm

    async def _handle_unregistration(self, msg: Message):
        """处理取消注册消息"""
        robot_id = msg.sender_id
        self._robots.pop(robot_id, None)
        self._robot_states.pop(robot_id, None)
        self._step_results.pop(robot_id, None)

    async def _handle_state_update(self, msg: Message):
        """处理状态更新消息"""
        robot_id = msg.sender_id
        result: SimulationResult = msg.data
        
        # 保存状态
        self._robot_states[robot_id] = result
        self._step_results[robot_id] = result
        
        # 广播新状态给所有机器人
        broadcast_tasks = []
        for robot_comm in self._robots.values():
            broadcast_tasks.append(
                robot_comm.inbox.put(Message(
                    MessageType.ALL_STATES_UPDATE,
                    -1,
                    self._robot_states
                ))
            )
        await asyncio.gather(*broadcast_tasks)

    async def _handle_step_complete(self, msg: Message):
        """处理步骤完成消息"""
        robot_id = msg.sender_id
        is_idle = msg.data
        self._step_complete_count += 1
        
        # 检查是否所有机器人都完成了任务
        if self._step_complete_count == len(self._robots):
            self._all_complete = all(
                result.traj_result and result.traj_result.is_complete
                for result in self._robot_states.values()
            )