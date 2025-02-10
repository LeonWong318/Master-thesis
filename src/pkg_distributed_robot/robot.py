from typing import Any, Optional, Union, Dict, List, Tuple
from dataclasses import dataclass
import numpy as np
import asyncio
from enum import Enum, auto
from datetime import datetime

PathNode = Tuple[float, float]
TrajNode = Tuple[float, float, float]

# 定义消息类型
class MessageType(Enum):
    COMPUTE_REQUEST = auto()       # Manager请求Robot计算下一步
    STATE_UPDATE = auto()          # Robot向Manager更新自己的状态
    ALL_STATES_UPDATE = auto()     # Manager向所有Robot广播状态
    STEP_COMPLETE = auto()         # Robot通知Manager本步计算完成
    REGISTRATION = auto()          # Robot注册到Manager
    UNREGISTRATION = auto()        # Robot从Manager注销
    TRAJ_UPDATE = auto()          # 轨迹更新消息

@dataclass
class RobotState:
    """机器人状态"""
    position: np.ndarray          # 当前位置状态
    predicted_states: np.ndarray  # 预测状态序列
    ref_traj: np.ndarray         # 参考轨迹
    ref_speed: float             # 参考速度
    timestamp: float             # 时间戳
    is_idle: bool               # 是否空闲

@dataclass
class SimulationParams:
    """仿真参数"""
    kt: int                       # 当前时间步
    ts: float                     # 采样时间
    current_time: float          # 当前时间
    config_mpc: Any              # MPC配置
    static_obstacles: List[List[PathNode]]  # 静态障碍物
    other_robot_states: List[RobotState]   # 其他机器人状态

@dataclass
class TrajectoryResult:
    """轨迹计算结果"""
    ref_states: np.ndarray       # 参考状态序列
    ref_speed: float            # 参考速度
    is_complete: bool          # 是否完成

@dataclass
class SimulationResult:
    """仿真结果"""
    robot_id: int
    state: np.ndarray            # 当前状态
    pred_states: np.ndarray      # 预测状态序列
    debug_info: dict            # 调试信息
    current_refs: Any           # 当前参考
    actions: np.ndarray         # 控制动作
    traj_result: TrajectoryResult  # 轨迹计算结果
    timestamp: float           # 时间戳

@dataclass
class Message:
    """消息基类"""
    msg_type: MessageType
    sender_id: int
    data: Any
    timestamp: float = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().timestamp()

class Communication:
    """通信接口"""
    def __init__(self, delay: float = 0.0):
        self.inbox = asyncio.Queue()
        self.outbox = asyncio.Queue()
        self.delay = delay
        
    async def send(self, message: Message):
        """发送消息（考虑延迟）"""
        if self.delay > 0:
            await asyncio.sleep(self.delay)
        await self.outbox.put(message)
        
    async def receive(self) -> Message:
        """接收消息（考虑延迟）"""
        if self.delay > 0:
            await asyncio.sleep(self.delay)
        return await self.inbox.get()

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

    def initialize(self, controller: Any, planner: Any, visualizer: Any) -> None:
        """初始化组件"""
        self.controller = controller
        self.planner = planner
        self.visualizer = visualizer

    def set_state(self, state: np.ndarray) -> None:
        """设置状态"""
        self._state = state
        if self.controller:
            self.controller.set_current_state(state)

    def load_schedule(self, path_coords: List[PathNode], path_times: Optional[List[float]] = None) -> None:
        """加载机器人调度，初始速度设为1"""
        if self.planner is None:
            raise RuntimeError("Planner not initialized")
        self.planner.load_path(path_coords, path_times, nomial_speed=1.0, method="linear")

    @property
    def state(self) -> np.ndarray:
        """获取当前状态"""
        if self.controller:
            return self.controller.state
        return self._state

    def step(self, action: np.ndarray) -> None:
        """执行一步"""
        # 如果没有控制器，直接返回
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

    def subscribe(self, manager: 'RobotManager') -> None:
        """订阅到管理器"""
        if self._manager is not None:
            raise ValueError(f"Robot {self.id_} already subscribed to a manager")
        self._manager = manager
        manager.register_robot(self)

    def unsubscribe(self) -> None:
        """取消订阅"""
        if self._manager:
            self._manager.unregister_robot(self)
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
        # 首先计算轨迹
        traj_result = await self._compute_trajectory(params)
        
        # 更新控制器的参考轨迹
        self.controller.set_ref_states(traj_result.ref_states, traj_result.ref_speed)

        # 执行MPC控制计算
        actions, pred_states, current_refs, debug_info = self.controller.run_step(
            static_obstacles=params.static_obstacles,
            full_dyn_obstacle_list=None,
            other_robot_states=params.other_robot_states,
            map_updated=False
        )

        # 保存下一步的动作，但还不执行
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
        """处理计算请求"""
        params: SimulationParams = msg.data
        result = await self._compute_next_step(params)
        
        # 发送计算结果
        await self.communication.send(Message(
            MessageType.STATE_UPDATE,
            self.id_,
            result
        ))

    async def _handle_state_update(self, msg: Message):
        """处理状态更新"""
        all_states = msg.data
        if self._next_action is not None:
            # 执行动作
            self.controller.step(self._next_action)
            self._state = self.controller.state
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

class RobotManager:
    """机器人管理器"""
    def __init__(self):
        self._robots: Dict[int, Robot] = {}
        self._robot_states: Dict[int, SimulationResult] = {}
        self._step_complete_count: int = 0
        self._all_complete: bool = False

    def register_robot(self, robot: Robot) -> None:
        """注册机器人"""
        if robot.id_ in self._robots:
            raise ValueError(f"Robot with id {robot.id_} already registered")
        self._robots[robot.id_] = robot

    def unregister_robot(self, robot: Robot) -> None:
        """取消注册机器人"""
        self._robots.pop(robot.id_, None)
        self._robot_states.pop(robot.id_, None)

    def get_all_robots(self) -> List[Robot]:
        """获取所有注册的机器人"""
        return list(self._robots.values())
    
    def get_robot_state(self, robot_id: int) -> np.ndarray:
        """获取机器人状态"""
        self._check_id(robot_id)
        return self._robots[robot_id].state

    def _check_id(self, robot_id: int) -> None:
        """检查机器人ID是否存在"""
        if robot_id not in self._robots:
            raise ValueError(f'Robot {robot_id} does not exist!')
        
    def get_pred_states(self, robot_id: int) -> Optional[np.ndarray]:
        """获取机器人预测状态"""
        self._check_id(robot_id)
        return self._robots[robot_id].pred_states

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
        for rid in list(self._robots.keys()):
            if rid != ego_robot_id:
                # 添加当前状态
                current_state = self.get_robot_state(rid)
                if isinstance(current_state, np.ndarray):
                    current_state = current_state.tolist()
                other_robot_states[idx : idx+state_dim] = current_state
                idx += state_dim

                # 添加预测状态
                pred_states = self.get_pred_states(rid)
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
                        # 如果预测状态不够长，用最后一个状态填充
                        last_state = pred_flat[-state_dim:] if pred_flat else [default] * state_dim
                        while len(pred_flat) < pred_len:
                            pred_flat.extend(last_state)
                            
                    other_robot_states[idx_pred : idx_pred+pred_len] = pred_flat
                    idx_pred += state_dim * horizon

        return other_robot_states

    async def simulate_step(self, kt: int, config_mpc: Any, static_obstacles: Any) -> List[SimulationResult]:
        """执行一步仿真"""
        results = []
        for robot in self._robots.values():
            try:
                # 获取局部参考轨迹
                ref_states, ref_speed, _ = robot.planner.get_local_ref(
                    kt * config_mpc.ts,
                    (float(robot.state[0]), float(robot.state[1]))
                )
                
                # 确保参考轨迹长度正确
                if len(ref_states) < config_mpc.N_hor:
                    last_state = ref_states[-1]
                    padding = np.tile(last_state, (config_mpc.N_hor - len(ref_states), 1))
                    ref_states = np.vstack([ref_states, padding])
                elif len(ref_states) > config_mpc.N_hor:
                    ref_states = ref_states[:config_mpc.N_hor]
                
                # 设置控制器的参考轨迹
                robot.controller.set_ref_states(ref_states, ref_speed=ref_speed)
                
                # 获取其他机器人状态
                other_robot_states = self.get_other_robot_states(robot.id_, config_mpc)
                
                # 执行MPC控制计算
                actions, pred_states, current_refs, debug_info = robot.controller.run_step(
                    static_obstacles=static_obstacles,
                    full_dyn_obstacle_list=None,
                    other_robot_states=other_robot_states,
                    map_updated=True
                )

                # 更新机器人状态
                robot.step(actions[-1])
                robot.pred_states = pred_states

                # 保存结果
                result = SimulationResult(
                    robot_id=robot.id_,
                    state=robot.state,
                    pred_states=np.asarray(pred_states) if isinstance(pred_states, list) else pred_states,
                    debug_info=debug_info,
                    current_refs=current_refs,
                    actions=np.array(actions),
                    traj_result=None,
                    timestamp=datetime.now().timestamp()
                )
                results.append(result)
                self._robot_states[robot.id_] = result
                
            except Exception as e:
                print(f"Error simulating robot {robot.id_}: {str(e)}")
                raise

        return results

# 轨迹调度相关的辅助函数
def load_robot_schedule(robot: Robot, path_coords: List[PathNode], path_times: Optional[List[float]]=None):
    """加载机器人调度"""
    robot.planner.load_path(path_coords, path_times)