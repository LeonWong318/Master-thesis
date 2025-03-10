from typing import Any, Optional, List
from dataclasses import dataclass
import numpy as np
from datetime import datetime
import asyncio

from .messages import (
    Message, MessageType, Communication, NetworkDelay,
    SimulationParams, SimulationResult, TrajectoryResult
)

from .types import PathNode, RobotManagerProtocol


class Robot:
    def __init__(self, config: Any, motion_model: Any, id_: Optional[int] = None):
        self.id_ = id_ if id_ is not None else id(self)
        self.config = config
        self.motion_model = motion_model
        self._state = None
        self.controller = None
        self.planner = None
        self.visualizer = None
        self.pred_states = None
        self._manager = None
        self._next_action = None
        self._running = False
        self._message_task = None
        
        # 设置通信
        network_delay = NetworkDelay(
            mean_delay=0.1,
            std_delay=0.02,
            min_delay=0.05,
            max_delay=0.2
        )
        self.communication = Communication(network_delay)

    async def start(self):
        """启动机器人的通信和控制循环"""
        self._running = True
        # 在后台启动消息循环
        self._message_task = asyncio.create_task(self._run_message_loop())

    async def stop(self):
        """停止机器人"""
        self._running = False
        if self._message_task:
            self._message_task.cancel()
            try:
                await self._message_task
            except asyncio.CancelledError:
                pass
            self._message_task = None
        
        if self._manager:
            await self.unsubscribe()

    async def _run_message_loop(self):
        """消息处理主循环"""
        try:
            while self._running:
                try:
                    # 接收消息
                    message = await asyncio.wait_for(
                        self.communication.receive(),
                        timeout=0.1  # 100ms 超时
                    )
                    
                    if message is None:  # 消息可能因为网络延迟而丢失
                        continue
                        
                    # 处理消息
                    handler = self._message_handlers.get(message.msg_type)
                    if handler:
                        await handler(message)
                        
                except asyncio.TimeoutError:
                    # 超时是正常的，继续循环
                    continue
                except Exception as e:
                    print(f"Error in robot {self.id_} message loop: {e}")
                    
                # 添加短暂延迟避免CPU占用过高
                await asyncio.sleep(0.001)
                
        except asyncio.CancelledError:
            # 正常的取消操作
            pass
        finally:
            self._running = False

    async def subscribe(self, manager: 'RobotManagerProtocol') -> None:
        """订阅到管理器"""
        if self._manager is not None:
            raise ValueError(f"Robot {self.id_} already subscribed to a manager")
        
        self._manager = manager
        # 发送注册消息
        await self.communication.send(Message(
            MessageType.REGISTRATION,
            self.id_,
            self.communication
        ))
        
        # 等待注册确认（可选）
        try:
            register_complete = asyncio.Event()
            await asyncio.wait_for(register_complete.wait(), timeout=1.0)
        except asyncio.TimeoutError:
            print(f"Warning: Registration confirmation timeout for robot {self.id_}")


    def initialize(self, controller: Any, planner: Any, visualizer: Any) -> None:
        """初始化机器人组件"""
        self.controller = controller
        self.planner = planner
        self.visualizer = visualizer
        if self.controller and self._state is not None:
            self.controller.set_current_state(self._state)

    def set_state(self, state: np.ndarray) -> None:
        """设置机器人状态"""
        self._state = state
        if self.controller:
            self.controller.set_current_state(state)

    def load_schedule(self, path_coords: List[PathNode], path_times: Optional[List[float]] = None) -> None:
        """加载路径调度"""
        if self.planner is None:
            raise RuntimeError("Planner not initialized")
        self.planner.load_path(path_coords, path_times, nomial_speed=0.1, method="linear")

    @property
    def state(self) -> np.ndarray:
        """获取当前状态"""
        if self.controller:
            return self.controller.state
        return self._state

    def step(self, action: np.ndarray) -> None:
        """执行一步动作"""
        if not self.controller:
            return

        # 存储当前状态和控制输入
        last_action = action
        self.controller.past_actions.append(last_action)
        
        # 使用运动模型更新状态
        next_state = self.controller.motion_model(
            self.state, 
            action, 
            self.controller.ts
        )
        
        # 更新控制器和内部状态
        self.controller.state = next_state
        self._state = next_state

    async def unsubscribe(self) -> None:
        """取消订阅"""
        if self._manager:
            await self.communication.send(Message(
                MessageType.UNREGISTRATION,
                self.id_,
                None
            ))
            self._manager = None

    async def _compute_trajectory(self, params: SimulationParams) -> TrajectoryResult:
        """计算轨迹"""
        current_pos = (float(self._state[0]), float(self._state[1]))
        ref_states, ref_speed, is_complete = self.planner.get_local_ref(
            params.current_time,
            current_pos
        )
        return TrajectoryResult(
            ref_states=ref_states,
            ref_speed=ref_speed,
            is_complete=is_complete
        )

    async def _compute_next_step(self, params: SimulationParams) -> SimulationResult:
        """计算下一步状态"""
        # 计算轨迹
        traj_result = await self._compute_trajectory(params)
        
        # 更新控制器参考轨迹
        self.controller.set_ref_states(traj_result.ref_states, traj_result.ref_speed)

        # 执行MPC控制计算
        actions, pred_states, current_refs, debug_info = self.controller.run_step(
            static_obstacles=params.static_obstacles,
            full_dyn_obstacle_list=None,
            other_robot_states=params.other_robot_states,
            map_updated=False
        )

        # 保存下一步动作
        self._next_action = actions[-1]
        
        return SimulationResult(
            robot_id=self.id_,
            state=self._state,
            pred_states=np.asarray(pred_states),
            debug_info=debug_info,
            current_refs=current_refs,
            actions=actions,
            traj_result=traj_result,
            timestamp=datetime.now().timestamp()
        )

    async def _handle_compute_request(self, msg: Message):
        """处理计算请求消息"""
        params: SimulationParams = msg.data
        result = await self._compute_next_step(params)
        
        # 发送计算结果
        await self.communication.send(Message(
            MessageType.STATE_UPDATE,
            self.id_,
            result
        ))

    async def _handle_state_update(self, msg: Message):
        """处理状态更新消息"""
        all_states = msg.data
        if self._next_action is not None:
            # 执行动作
            self.step(self._next_action)
            self._next_action = None
            
            # 检查是否完成
            self._idle = self.controller.check_termination_condition(
                external_check=self.planner.idle
            )

        # 通知manager本步完成
        await self.communication.send(Message(
            MessageType.STEP_COMPLETE,
            self.id_,
            self._idle
        ))