from enum import Enum
import random

class FlowType(Enum):
  VOIP = 1      # 实时会话
  STREAMING = 2 # 流媒体
  INTERACTIVE = 3 # 交互式/游戏

# 存储每种流类型的 iperf 参数和 QoE 严格要求 (用于 Reward 函数)
FLOW_PROFILES = {
  FlowType.VOIP: {
    'iperf_mode': 'udp',          # -u
    'target_rate': '0.1M',        # 100 Kbps 左右的恒定速率
    'qoe_critical': {'max_delay': 150, 'max_jitter': 50}, # ms
    'reward_fn': 'E-Model'
  },
  FlowType.STREAMING: {
    'iperf_mode': 'tcp',
    'target_rate': '10M',         # 目标带宽 10 Mbps
    'qoe_critical': {'min_bandwidth': 5, 'max_loss_rate': 1e-6}, # Mbps, %
    'reward_fn': '3GPP-QCI6'
  },
  FlowType.INTERACTIVE: {
    'iperf_mode': 'udp',
    'target_rate': '1M',
    'qoe_critical': {'max_delay': 50, 'max_jitter': 30}, # ms
    'reward_fn': '3GPP-QCI80'
  }
}

class FlowGenerator:
  def get_random_flow(self) -> tuple[FlowType, dict]:
    """随机选择一个流类型及其配置文件。"""
    flow_type = random.choice(list(FlowType))
    profile = FLOW_PROFILES[flow_type]
    return flow_type, profile

# --- 示例用法 ---
# flow_gen = FlowGenerator()
# current_type, current_profile = flow_gen.get_random_flow()
# print(f"当前流类型: {current_type.name}")
# print(f"iperf模式: {current_profile['iperf_mode']}, 目标速率: {current_profile['target_rate']}")