import re
import math
import numpy as np
from typing import Any
from ...utils import logger, AttrDict
from ..flow_generator import FlowType, FLOW_PROFILES
from ..e_model import (
    FullG107Calculator, 
    RigorousVideoEvaluator, 
    RigorousLegacyFPSEvaluator)

# [Global Instances - Stateful/Stateless managers]
voip_calc = FullG107Calculator()
video_calc = RigorousVideoEvaluator(target_bitrate_kbps=5000) # 1080p Video
fps_game_calc = RigorousLegacyFPSEvaluator() # CSa

def parse_ditg_output(output_str: str) -> tuple[dict[str, Any], bool]:
    """
    解析 ITGDec 的标准输出文本，提取关键 QoS 指标。
    处理包括正常数值和异常值 (如 nan) 的情况。
    """

    metrics = {
        'delay': 0.0,      # 单位: ms
        'jitter': 0.0,     # 单位: ms
        'bandwidth': 0.0,  # 单位: Mbps
        'loss_rate': 1.0}  # 范围: 0.0 - 1.0 (默认为1.0即全丢，防止无数据时误判为满分)
  
    no_packet_arrive = False
  
    if not output_str:
        logger.log("can't catch output str -- the str is None", tag="Debug")
        return metrics, True

    logger.log("parsing ditg output ...", tag="Debug")
    # logger.log(output_str)
    try:
        # --- 1. 提取平均延迟 (Average delay) ---
        # 示例行: Average delay            =     0.000234 s
        delay_match = re.search(r"Average delay\s+=\s+([-\d\.nan]+)\s+s", output_str)
        if delay_match:
            val = delay_match.group(1)
            if 'nan' not in val.lower(): # 过滤掉 -nan
                metrics['delay'] = float(val) * 1000.0 # 秒 -> 毫秒
        else:
            logger.log("no delay found", tag="Debug")
            logger.log(f"原始输出片段:\n{output_str[:200]}", tag="Debug") # 调试用
    
        jitter_match = re.search(r"Average jitter\s+=\s+([-\d\.nan]+)\s+s", output_str)
        if jitter_match:
            val = jitter_match.group(1)
            if 'nan' not in val.lower():
                metrics['jitter'] = float(val) * 1000.0 # 秒 -> 毫秒
        else:
            logger.log("no jitter found", tag="Debug")

        # --- 3. 提取吞吐量 (Average bitrate) ---
        # 示例行: Average bitrate          =  4096.000000 Kbit/s
        bitrate_match = re.search(r"Average bitrate\s+=\s+([-\d\.nan]+)\s+Kbit/s", output_str)
        if bitrate_match:
            val = bitrate_match.group(1)
            if 'nan' not in val.lower():
                metrics['bandwidth'] = float(val) / 1000.0 # Kbit/s -> Mbps
        else:
            logger.log("no bitrate found", tag="Debug")

        # --- 4. 提取丢包率 (Packets dropped) ---
        # 示例行: Packets dropped          =            5 (0.50 %)
        # 注意：如果没有发包成功，分母为0可能导致 nan，或者 dropped 为 0 但 total 也为 0
        loss_match = re.search(r"Packets dropped\s+=\s+\d+\s+\(([-\d\.nan]+)\s+%\)", output_str)
        if loss_match:
            val = loss_match.group(1)
            if 'nan' not in val.lower():
                metrics['loss_rate'] = float(val)/100.0
        else: 
            logger.log("no loss found", tag="Debug")

        # 如果总包数 (Total packets) 为 0，说明完全没通，强制设置最差指标
        total_pkts_match = re.search(r"Total packets\s+=\s+(\d+)", output_str)
        if total_pkts_match and int(total_pkts_match.group(1)) == 0:
            no_packet_arrive = True
            metrics['loss_rate'] = 1.0
            metrics['bandwidth'] = 0.0

    except Exception as e:
        logger.log(f"解析 D-ITG 输出时出错: {e}", tag="Parse Err")
        logger.log(f"原始输出片段:\n{output_str[:200]}", tag="Debug") # 调试用

    return metrics, no_packet_arrive

def calculate_qoe_reward(qos_metrics: dict, flow_profile: dict) -> float:
    """
    Enhanced QoE Reward Calculation with Gradient Shaping.
  
    Logic:
    1. Base: ITU-T E-model MOS (1.0 - 4.5).
    2. Shaping: Add small bonuses for lower delay/higher BW even if MOS is maxed out.
    3. Normalize: Map to [-1.0, 1.0].
    """
    # 1. 解析 QoS 数据
    d = qos_metrics.get('delay', 1.0)       # ms
    j = qos_metrics.get('jitter', 0.1)      # ms
    l = qos_metrics.get('loss_rate', 0.0) * 100.0 # Convert to % (0-100)
    b = qos_metrics.get('bandwidth', 0.0) * 1000.0 # Convert to kbps
  
    # 2. 识别业务类型

    f_type_enum = flow_profile.get('type') 

    assert f_type_enum is not None, "flow type is None"

    f_type = f_type_enum.name.lower() if hasattr(f_type_enum, 'name') else str(f_type_enum).lower()

    mos = 1.0
    shaping_bonus = 0.0
  
    # 3. 计算 Base MOS 和 Shaping Bonus
    if 'voip' in f_type:
        # --- VoIP Logic ---
        # Base: ITU G.107 (假设 voip_calc 已实现)
        mos = voip_calc.calculate_mos(delay_ms=d, loss_pct=l, jitter_ms=j)
    
        # Shaping: 鼓励延迟 < 150ms。每降低 10ms，奖励增加约 0.003
        # 范围: [0.0, 0.05]
        if mos > 4.0: # 只有体验良好时才谈优化
            shaping_bonus = 0.05 * (1.0 - min(d, 150.0) / 150.0)

    elif 'streaming' in f_type:
        # --- Video Logic ---
        # Base: ITU P.1203 简化版
        mos = video_calc.calculate_mos(
            loss_pct=l, rtt_ms=d*2, physical_bw_kbps=b, 
            duration_sec=6.0, stateless_mode=True
        )
    
        # Shaping: 鼓励带宽冗余。
        # 假设 1080p 需要 5000kbps，如果能提供 8000kbps，给一点奖励作为缓冲安全区
        target_bw = 5000.0 
        if mos > 4.0 and b > target_bw:
            # 范围: [0.0, 0.05]
            # Log函数让收益边际递减，避免Agent为了无限带宽而绕远路
            shaping_bonus = 0.05 * np.tanh((b - target_bw) / 2000.0)

    elif 'gaming' in f_type:
        # --- Gaming Logic ---
        # Base: 针对 FPS 优化的模型
        mos = fps_game_calc.calculate_mos(delay_ms=d, loss_pct=l, jitter_ms=j)
    
        # Shaping: Gaming 对延迟极度敏感，给予更高的引导权重
        # 范围: [0.0, 0.1]
        # 强迫 Agent 区分 20ms 和 5ms
        if mos > 3.5:
            shaping_bonus = 0.1 * (1.0 - min(d, 50.0) / 50.0)

    # 4. 归一化 Reward (Normalization)
    # 原始 MOS: 1.0 (Bad) ~ 4.5 (Excellent)
    # 目标区间: -1.0 ~ 1.0
  
    # 先把 MOS 线性映射到 [-1.0, 0.9] (留 0.1 给 Bonus)
    # (MOS - 1.0) / 3.5 * 1.9 - 1.0 
    # 简化版: (MOS - 3.0) / 1.5 范围约为 [-1.33, 1.0]
  
    # 我们使用更保守的映射，确保加上 Bonus 后不超过 1.0
    normalized_base = (mos - 1.0) / 3.5 # 映射到 [0, 1]
    reward = (normalized_base * 2) - 1 # 映射到 [-1, 1]
  
    # 5. 叠加 Bonus
    final_reward = reward + shaping_bonus
  
    # 6. 悬崖惩罚 (Cliff Penalty) & 边界截断
    # 如果 MOS 太低，说明完全不可用，直接给 -1.0，忽略任何带宽优势
    if mos < 1.5:
        return -1.0
      
    # 确保数值稳定，截断在 [-1.0, 1.0]
    return float(np.clip(final_reward, -1.0, 1.0))

def calculate_qos_reward(delay_ms: float, loss_percent: float, jitter_ms: float, flow_type_str: str, config: AttrDict) -> float:
    qos_reward_info = config.qos_reward
    loss_normalized = loss_percent
    min_delay = qos_reward_info[flow_type_str.upper()]['min_delay']
    max_delay = qos_reward_info[flow_type_str.upper()]['max_delay']
    delay_normalized = min(max((delay_ms - min_delay) / (max_delay - min_delay), 0.0), 1.0)
    min_jitter = qos_reward_info[flow_type_str.upper()]['min_jitter']
    max_jitter = qos_reward_info[flow_type_str.upper()]['max_jitter']
    jitter_normalized = min(max((jitter_ms - min_jitter) / (max_jitter - min_jitter), 0.0), 1.0)
    
    
    weight = qos_reward_info[flow_type_str.upper()]['w']

    qos_reward = 5 - (
        weight[0] * delay_normalized +
        weight[1] * jitter_normalized +
        weight[2] * loss_normalized 
    ) 

    # qos reward in range [-sum_w, sum_w]
    qos_reward = qos_reward/2.5-1
    return qos_reward

    