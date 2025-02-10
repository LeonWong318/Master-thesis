from typing import Any, List, Optional, Dict, Tuple
from dataclasses import dataclass
import numpy as np
import asyncio
from enum import Enum, auto
from datetime import datetime

# 类型别名
PathNode = Tuple[float, float]
TrajNode = Tuple[float, float, float]

class MessageType(Enum):
    """消息类型枚举"""
    COMPUTE_REQUEST = auto()       # Manager请求Robot计算下一步
    STATE_UPDATE = auto()          # Robot向Manager更新自己的状态
    ALL_STATES_UPDATE = auto()     # Manager向所有Robot广播状态
    STEP_COMPLETE = auto()         # Robot通知Manager本步计算完成
    REGISTRATION = auto()          # Robot注册到Manager
    UNREGISTRATION = auto()        # Robot从Manager注销
    TRAJ_UPDATE = auto()          # 轨迹更新消息

@dataclass
class RobotState:
    """机器人状态数据类"""
    position: np.ndarray          # 当前位置状态 (x, y, theta)
    predicted_states: np.ndarray  # 预测状态序列
    ref_traj: np.ndarray         # 参考轨迹
    ref_speed: float             # 参考速度
    timestamp: float             # 时间戳
    is_idle: bool               # 是否空闲

@dataclass
class TrajectoryResult:
    """轨迹计算结果数据类"""
    ref_states: np.ndarray       # 参考状态序列
    ref_speed: float            # 参考速度
    is_complete: bool          # 是否完成

@dataclass
class SimulationParams:
    """仿真参数数据类"""
    kt: int                       # 当前时间步
    ts: float                     # 采样时间
    current_time: float          # 当前时间
    config_mpc: Any              # MPC配置
    static_obstacles: List[List[PathNode]]  # 静态障碍物
    other_robot_states: List[RobotState]   # 其他机器人状态

@dataclass
class SimulationResult:
    """仿真结果数据类"""
    robot_id: int                # 机器人ID
    state: np.ndarray            # 当前状态
    pred_states: np.ndarray      # 预测状态序列
    debug_info: Dict            # 调试信息
    current_refs: Any           # 当前参考
    actions: np.ndarray         # 控制动作
    traj_result: TrajectoryResult  # 轨迹计算结果
    timestamp: float            # 时间戳

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

class NetworkDelay:
    """网络延迟模拟"""
    def __init__(self, mean_delay: float = 0.1, std_delay: float = 0.02, 
                 min_delay: float = 0.05, max_delay: float = 0.2):
        """初始化网络延迟参数
        
        Args:
            mean_delay: 平均延迟（秒），默认100ms
            std_delay: 延迟标准差（秒），默认20ms
            min_delay: 最小延迟（秒），默认50ms
            max_delay: 最大延迟（秒），默认200ms
        """
        self.mean = mean_delay
        self.std = std_delay
        self.min = min_delay
        self.max = max_delay

    async def get_delay(self) -> float:
        """生成一个符合正态分布的延迟时间"""
        delay = np.random.normal(self.mean, self.std)
        return np.clip(delay, self.min, self.max)

class Communication:
    """通信接口"""
    def __init__(self, network_delay: Optional[NetworkDelay] = None):
        self.inbox = asyncio.Queue()
        self.outbox = asyncio.Queue()
        self.network = network_delay or NetworkDelay()
        
    async def send(self, message: Message):
        """发送消息（考虑延迟）"""
        delay = await self.network.get_delay()
        await asyncio.sleep(delay)
        await self.outbox.put(message)
        
    async def receive(self) -> Optional[Message]:
        """接收消息（考虑延迟）"""
        message = await self.inbox.get()
        delay = await self.network.get_delay()
        await asyncio.sleep(delay)
        return message