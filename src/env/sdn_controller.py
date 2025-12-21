import os
import math
import subprocess
from time import sleep, time
import re 
from mininet.net import Mininet
import numpy as np
from enum import Enum 
from contextlib import contextmanager
import torch
import sys
import signal
import shlex
import uuid 
from datetime import datetime
import networkx as nx
from mininet.topo import Topo
from functools import partial
from mininet.log import setLogLevel, info
from scapy.all import rdpcap
from scapy.layers.inet import IP
from mininet.node import OVSKernelSwitch, RemoteController
from mininet.link import TCLink

from ..utils import logger
from ..utils.verbose_logger import vprint, vprint_matrix, vprint_path_status, vprint_qos
from .flow_generator import FlowType, FLOW_PROFILES
from .e_model import (
  FullG107Calculator, 
  RigorousVideoEvaluator, 
  RigorousCloudGamingEvaluator, 
  RigorousLegacyFPSEvaluator)


# Critic
# 根据 QoS 打分

# [Global Instances - Stateful/Stateless managers]
voip_calc = FullG107Calculator()
video_calc = RigorousVideoEvaluator(target_bitrate_kbps=5000) # 1080p Video

fps_game_calc = RigorousLegacyFPSEvaluator() # CSa
# cloud_game_calc = RigorousCloudGamingEvaluator(target_bitrate_kbps=8000) # Cloud Gaming

def parse_ditg_output(output_str: str) -> dict:
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
    logger.log("can't catch output str -- the str is None", tag="Parse Err")
    return metrics, True

  logger.log("parsing ditg output ...", tag="Parser")
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
      logger.log("no delay found", tag="Parse Err")
      logger.log(f"原始输出片段:\n{output_str[:200]}", tag="Parse Err") # 调试用
    
    jitter_match = re.search(r"Average jitter\s+=\s+([-\d\.nan]+)\s+s", output_str)
    if jitter_match:
      val = jitter_match.group(1)
      if 'nan' not in val.lower():
        metrics['jitter'] = float(val) * 1000.0 # 秒 -> 毫秒
    else:
      logger.log("no jitter found", tag="Parse Err")

    # --- 3. 提取吞吐量 (Average bitrate) ---
    # 示例行: Average bitrate          =  4096.000000 Kbit/s
    bitrate_match = re.search(r"Average bitrate\s+=\s+([-\d\.nan]+)\s+Kbit/s", output_str)
    if bitrate_match:
      val = bitrate_match.group(1)
      if 'nan' not in val.lower():
        metrics['bandwidth'] = float(val) / 1000.0 # Kbit/s -> Mbps
    else:
      logger.log("no bitrate found", tag="Parse Err")

    # --- 4. 提取丢包率 (Packets dropped) ---
    # 示例行: Packets dropped          =            5 (0.50 %)
    # 注意：如果没有发包成功，分母为0可能导致 nan，或者 dropped 为 0 但 total 也为 0
    loss_match = re.search(r"Packets dropped\s+=\s+\d+\s+\(([-\d\.nan]+)\s+%\)", output_str)
    if loss_match:
      val = loss_match.group(1)
      if 'nan' not in val.lower():
        metrics['loss_rate'] = float(val)/100.0
    else: 
      logger.log("no loss found", tag="Parse Err")

    # 如果总包数 (Total packets) 为 0，说明完全没通，强制设置最差指标
    total_pkts_match = re.search(r"Total packets\s+=\s+(\d+)", output_str)
    if total_pkts_match and int(total_pkts_match.group(1)) == 0:
      no_packet_arrive = True
      metrics['loss_rate'] = 1.0
      metrics['bandwidth'] = 0.0

  except Exception as e:
    logger.log(f"解析 D-ITG 输出时出错: {e}", tag="Parse Err")
    logger.log(f"原始输出片段:\n{output_str[:200]}", tag="Parse Err") # 调试用

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

def calculate_qos_reward(delay_ms, loss_percent, jitter_ms, bw_mbps, flow_type_str, config):
  """
  重构版 QoS 奖励函数：基于满意度 (Satisfaction) 累加，而非惩罚扣分。
  解决 Gaming 在好网络下拿负分的问题。
  
  Args:
      flow_type_str: 必须确保传入的是纯字符串 "GAMING", "VOIP", "STREAMING"
  """
  
  # --- 1. 安全清洗与配置 ---
  # 确保类型匹配 (处理可能的 FlowType.GAMING 格式)
  f_type = str(flow_type_str).upper().split('.')[-1]
  
  # 定义业务约束 (Hard Constraints) & 期望 (Expectations)
  # 这里的 max_delay 定义为 "超过此值满意度为0"
  params = config.qos_reward
  
  p = params.get(f_type, params['DEFAULT'])
  weights = p['w'] 

  # --- 2. 计算各项满意度 (Score: 0.0 ~ 1.0) ---
  
  # [Delay Score]: 线性下降
  # 23ms / 80ms -> 0.28 usage -> score 0.72
  s_delay = max(0.0, 1.0 - (delay_ms / p['max_delay']))
  
  # [Jitter Score]: 线性下降
  s_jitter = max(0.0, 1.0 - (jitter_ms / p['max_jitter']))
  
  # [Loss Score]: 指数衰减 (对丢包非常敏感)
  # loss 0% -> 1.0; loss 1% -> 0.36; loss 5% -> 0.006
  # 这样能给无丢包的 Gaming 极高的基础分
  s_loss = math.exp(-100.0 * loss_percent)
  
  # [Bandwidth Score]: 带宽豁免逻辑 (核心修复)
  s_bw = 0.0
  is_low_traffic = f_type in ['GAMING', 'VOIP']
  
  if is_low_traffic and loss_percent < 0.001:
      # 如果是小流且没丢包，直接认为带宽满意度 100%
      s_bw = 1.0
  else:
      # 常规对数打分: 达到 ref_bw 即满分
      # bw=0.01, ref=0.5 -> score ~0 (如果不豁免)
      # bw=5.0,  ref=5.0 -> score 1.0
      val = math.log(bw_mbps + 1.0) / math.log(p['ref_bw'] + 1.0)
      s_bw = min(1.0, max(0.0, val))

  # --- 3. 加权总分 (0.0 ~ 1.0) ---
  # 这是一个很稳健的数值，代表 QoS 满足程度
  final_score = (weights[0] * s_delay) + \
                (weights[1] * s_jitter) + \
                (weights[2] * s_loss) + \
                (weights[3] * s_bw)

  # --- 4. 映射到 RL 奖励区间 ---
  # 将 [0, 1] 映射到 [-1, 1] 或者保持 [0, 1]
  # 建议使用 [-1, 1] 区间，中心化 0.0，利于 Critic 收敛
  # 如果 score < 0.3 (很差)，给负分； score > 0.3 给正分
  # 这里的公式：(score * 2) - 1.0
  # 例: 满分 1.0 -> Reward 1.0
  # 例: 分数 0.5 -> Reward 0.0
  # 例: 分数 0.0 -> Reward -1.0
  rl_reward = (final_score * 2.0) - 1.0
  
  # [Debug Print] 这一步非常重要，能让你在日志里看到每一项是多少
  # 格式化输出，只在需要时取消注释，或者保留到日志
  # debug_msg = f"[{f_type}] Rew:{rl_reward:.2f} (S_Tot:{final_score:.2f}) | " \
  #             f"D:{s_delay:.2f} J:{s_jitter:.2f} L:{s_loss:.2f} B:{s_bw:.2f}"
  # print(debug_msg) 

  return rl_reward

def measure_path_qos(server, client, path_route, flow_type, config, resend=False):
  """
  [完美健壮版] 解决时序竞争与解析失败
  """
  # --- 1. 给 OVS 流表下发一点“呼吸时间” ---
  # 解决代码跑得比交换机快的问题
  sleep(0.2) 

  # --- 3. 准备 D-ITG 参数 ---
  random_id = uuid.uuid4().hex[:8]
  log_prefix = f"/dev/shm/itg_{client.name}_{server.name}_{random_id}"
  recv_log = f"{log_prefix}.recv"  
  
  # target_duration = 6 if flow_type==FlowType.STREAMING else 2

  target_duration = 1.5

  if flow_type == FlowType.STREAMING:
    # TCP 给 2.5 倍余量，防止拥塞误杀
    safe_timeout = target_duration* 2.5 + 4  # 6*2.5 + 2 = 17s
  else:
    # UDP 给 2 秒余量即可
    safe_timeout = target_duration + 4  # 2 + 4 = 6s
  
  success = run_itg_safe(
    client_node=client,
    server_node=server,
    log_file=recv_log,
    flow_type=flow_type,
    duration_sec =target_duration,
    timeout_sec=int(safe_timeout))

  if not success:
    # 如果 run_itg_safe 返回 False (连接彻底失败)，直接返回 -1
    return -1.0, -1.0

  # if stderr: return 0
  # 检查文件是否存在 (防止传输完全失败导致无日志)
  check_log = server.cmd(f"ls {recv_log}")
  if "No such file" in check_log and not resend:
    logger.log("No log generated. Resend same cmd again", tag="Sender Err")
    client.cmd(f"rm -f {recv_log}")
    return measure_path_qos(server, client, path_route, flow_type, config, resend=True)

  # 运行解码器拿到文本结果 
  # 解析结果 
  # Meta-DRL 
  try:
    logger.log(f"Running ITGDec on {recv_log}...", tag="Decoder")

    # 1. 启动进程
    with client.popen(
      f"ITGDec {recv_log}", 
      shell=True,
      stdout=subprocess.PIPE, 
      stderr=subprocess.PIPE, 
      text=True 
    ) as dec_proc:

      # 2. 等待进程结束并获取输出 (Block until finished)
      # communicate 会读取 stdout 直到 EOF，确保拿到了所有输出
      stdout, stderr = dec_proc.communicate(timeout=6) # 设置个超时防止卡死

      if dec_proc.returncode != 0 and not resend:
        dec_proc.kill()
        logger.log(f"ITGDec failed with code {dec_proc.returncode}", tag="Decoder Err")
        return measure_path_qos(server, client, path_route, flow_type, config, resend=True)
      else:
        dec_output = stdout
        logger.log("Success dec recieve file", tag="Decoder")

  except subprocess.TimeoutExpired:
    logger.log("ITGDec Timed out!", tag="Decoder Err")
    dec_proc.kill()
    dec_output = ""

  finally:
    # 清理所有临时文件 
    client.cmd(f"rm -f {recv_log}")

  # 2. 解析 D-ITG 输出
  qos_metrics, no_packet_arrive = parse_ditg_output(dec_output)    
  if no_packet_arrive:
    if not resend :
      logger.log(f"No packet arrive : Resend cmd again", tag="Sender Err")
      return measure_path_qos(server, client, path_route, flow_type, config, resend=True)
    else :
      logger.log(f"Fail to send packet, bad path", tag="Sender Err")
      return -1.0, -1.0

  # 假设 qos_metrics 里的数值已经拿到了
  d = qos_metrics['delay']
  j = qos_metrics['jitter']
  b = qos_metrics['bandwidth']
  l = qos_metrics['loss_rate']

  vprint_qos(flow_type.name, delay=d, jitter=j, bw=b, loss=l, tag="QoS")

  # 计算 Reward
  qoe_reward = calculate_qoe_reward(qos_metrics, FLOW_PROFILES[flow_type])
  qos_reward = calculate_qos_reward(
    delay_ms=qos_metrics['delay'],
    loss_percent=qos_metrics['loss_rate'],
    jitter_ms=qos_metrics['jitter'],
    bw_mbps=qos_metrics['bandwidth'],
    flow_type_str=flow_type.name,
    config=config
  )

  return qos_reward, qoe_reward

# Generator :
# 生成一个mininet网络

# mininet 定义
class GraphTopo(Topo):
  def __init__(self, blueprint_g: nx.Graph, is_test=False, **opts):
    Topo.__init__(self, **opts)
    test_str = "T" if is_test else ""

    for node_id in blueprint_g.nodes():
      self.addSwitch(f'{test_str}s{node_id}', protocols='OpenFlow13')
      self.addHost(f'{test_str}h{node_id}')
      self.addLink(f'{test_str}h{node_id}', f'{test_str}s{node_id}', delay='0ms')

    for u, v, data in blueprint_g.edges(data=True):
      bw = data.get('bandwidth', 30.0) # Mbps
      delay = f"{data.get('delay', 10)}ms"
      loss = data.get('loss', 0)
      q_limit = data.get('queue_size', 2000)
      rate_bytes = bw * 1000000 / 8
      # 2. 计算最佳 r2q (确保 quantum ≈ 1500)
      r2q = int(max(1, rate_bytes / 1500))
      # 这里沿用 Mininet 构造函数中设置的 r2q
      logger.log(f"Adding link: {u} <-> {v} | bw: {bw} Mbps | delay: {delay} | loss: {loss}% | r2q: {r2q} | qlimit: {q_limit}", tag="Mini Init")
      self.addLink(f'{test_str}s{u}', f'{test_str}s{v}', cls=TCLink, bw=bw, delay=delay, loss=loss, r2q = r2q, use_htb=True, max_queue_size=q_limit) 

# mininet 启动
@contextmanager
def get_a_mininet(g: nx.Graph, is_test=False, remote_port=None):
  if remote_port:
    controller = partial(RemoteController, ip='127.0.0.1', port=remote_port)
  else:
    controller = None

  # if not vp.MININET_VERBOSE:
  # setLogLevel('critical')

  net = Mininet(
    topo=GraphTopo(g, is_test),
    switch=OVSKernelSwitch,
    link=TCLink,
    controller=controller,
    autoSetMacs=True, 
    autoStaticArp=True)

  try:
    logger.log("Disabling TCP Offload (TSO/GSO/GRO) on all switches...", tag="Mini Init")
    for h in net.hosts:
      for intf in h.intfList():
        if intf.name != 'lo':
          # 使用 ethtool 关闭卸载
          h.cmd(f"ethtool -K {intf.name} tso off gso off gro off > /dev/null 2>&1")
    for sw in net.switches:
      for intf in sw.intfList():
        if intf.name != 'lo':
          sw.cmd(f"ethtool -K {intf.name} tso off gso off gro off > /dev/null 2>&1")
          
    net.start()
    yield net
  finally:
    logger.log("stopping mininet ...", tag="Mini Stop")
    net.stop()

  return net

# 获取一个流量特征张量
def get_a_fingerprint(
  server, 
  client, 
  flow_type: FlowType, 
  n_packets_to_capture=30, 
  **flow_params):

  duration_sec = 15 

  final_tensor = send_packet_and_capture(
    server=server,
    client=client,
    flow_type=flow_type,
    duration_sec=duration_sec,
    n_packets_to_capture=n_packets_to_capture)
  
  while final_tensor.size(0) < 30:
    sleep(1)
    final_tensor = send_packet_and_capture(
      server=server,
      client=client,
      flow_type=flow_type,
      duration_sec=duration_sec,
      n_packets_to_capture=n_packets_to_capture)
  
  return normalize_fingerprint(final_tensor).unsqueeze(0)

# 发送流量并捕获包特征
def send_packet_and_capture(
  server, 
  client, 
  flow_type: FlowType, 
  duration_sec=15, 
  n_packets_to_capture=30, 
  **flow_params):
  """
  在 Mininet 中运行 D-ITG 流量, 并同时使用 tshark 管道实时捕获特征。
  [Fix] 集成了 ensure_server_surgical 以防止端口冲突。
  """

  server_ip = server.IP()
  client_ip = client.IP()
  
  # 1. 查找监听接口
  server_intf = None
  for intf in server.intfList():
    if intf.name != 'lo' and intf.link:
      server_intf = intf
      break
  if server_intf is None:
    raise Exception(f"在 {server.name} 上找不到已连接的数据接口!")

  switch_intf = server_intf.link.intf2 if server_intf.link.intf1 == server_intf else server_intf.link.intf1
  switch_intf_name = switch_intf.name
  
  feature_matrix = []
  client_proc = None
  tshark_proc = None
  server_proc = None

  try:
    # 2. [Action 1] 安全启动服务端 (获取动态端口)
    # 这会清理旧进程并返回一个干净的端口
    server_proc, actual_port = ensure_server_surgical(server)

    # 3. [Action 2] 获取客户端命令 (使用动态端口)
    client_cmd = get_flow_command(
      flow_type=flow_type,
      target_ip=server_ip,
      duration_sec=duration_sec,
      sig_port=actual_port, # [Fix] 使用实际端口
      **flow_params)

    MARK_TOS = 32
    
    # 4. [Action 3] 启动 tshark
    display_filter = f"src host {client_ip} and dst host {server_ip} and ip[1] == {MARK_TOS}"
    # 增加一点超时余量
    timeout_duration = duration_sec + 5

    tshark_cmd = [
      'sudo', 'tshark',
      '-c', str(n_packets_to_capture),
      '-a', f'duration:{timeout_duration}',
      '-i', switch_intf_name,
      '-l', 
      '-T', 'fields',
      '-e', 'frame.len',
      '-e', 'frame.time_delta',
      '-e', 'ip.src',
      '-e', 'ip.dst',
      '-E', 'separator=,',
      '-f', display_filter]
  
    tshark_proc = subprocess.Popen(
      tshark_cmd, 
      stdout=subprocess.PIPE, 
      stderr=subprocess.DEVNULL,
      text=True
    )

    # 5. [Action 4] 启动客户端流量
    sleep_time = 1.0 # 给 tshark 一点启动时间
    sleep(sleep_time)
    
    # 使用 popen 启动客户端
    client_proc = client.popen(client_cmd, shell=True)

    # 6. [核心] 实时读取
    for line in tshark_proc.stdout:
      line = line.strip()
      if not line: continue
      try:
        size_str, iat_str, src_ip, dst_ip = line.split(',')
        size = float(size_str)
        try:
          iat = float(iat_str)
        except ValueError:
          iat = 0.0
        
        feature_vector = [size, iat]
        feature_matrix.append(feature_vector)
      except ValueError:
        logger.log("Grap packet failed!", tag="Tshark Err")
        pass # 忽略解析错误
      
  except Exception as e:
    logger.log(f"采集指纹出错: {e}", tag="Tshark Err")
  
  finally:
    # [Fix] 统一清理资源
    if tshark_proc:
      tshark_proc.kill()
    
    if client_proc:
      # 向进程组发送信号，确保杀死 chrt 启动的子进程
      try:
        client_proc.kill()
        # 如果使用了 os.setsid (虽然这里没显式用，但为了保险)
        # os.killpg(os.getpgid(client_proc.pid), signal.SIGKILL)
      except:
        pass
          
    if server_proc:
      try:
        server_proc.terminate()
        server_proc.wait(timeout=0.5)
      except:
        server_proc.kill()

  # 如果没抓到包，返回全0或者随机噪声防止报错，但在训练初期这可能导致冷启动问题
  if len(feature_matrix) == 0:
    return torch.zeros((n_packets_to_capture, 2), dtype=float)

  fingerprint_tensor = torch.tensor(feature_matrix, dtype=float)
  return fingerprint_tensor

# 根据流类型，返回不同的 D-ITG 命令。
def get_flow_command(
  flow_type: str, 
  target_ip: str, 
  duration_sec: int, 
  sig_port: int = 15000,
  log_file: str = None,
  **kwargs
  ) -> str:
  """
  Generates a high-priority D-ITG command string.
  Supports both TCP (Streaming) and UDP (VoIP/Gaming).
  """
  # Map friendly names to FlowType Enum or dict keys
  # Assuming FLOW_PROFILES is a global dict defined elsewhere
  profile = FLOW_PROFILES.get(flow_type, FLOW_PROFILES.get('voip')) 
  
  protocol = profile['protocol'] # 'UDP' or 'TCP'
  duration_ms = int(duration_sec * 1000)
  
  # Log file argument
  log_str = f"-x {log_file}" if log_file else ""
  flow_tos = 32
  # 1. Base ITGSend Arguments
  # -a: Target IP
  # -rp: Remote Port (Must match server!)
  # -Sdp: Signaling Port (Must match server!)
  # -t: Duration in ms
  # -T: Transport Protocol
  itg_args = (
    f"-a {shlex.quote(target_ip)} "
    f"-rp 12000 "
    f"-b {flow_tos} "
    f"-Sdp {sig_port} "  # Use same port for signaling to keep it simple
    f"{log_str} "
    f"-t {duration_ms} "
    f"-T {protocol}")

  # 2. Add Payload Parameters (Size/Rate)
  # Priority: Manual args > Preset profile > Default
  if 'ditg_manual' in profile:
    specific_args = profile['ditg_manual']
  elif 'ditg_preset' in profile:
    specific_args = profile['ditg_preset']
  else:
    specific_args = "-C 100 -c 100 " # Safe default

  # 3. Apply "Nuclear Option" (Real-Time Priority)
  # chrt -r 99: Run as Real-Time Round-Robin process with max priority
  # This ensures the marker flow isn't starved by background traffic.
  # wrapper = "chrt -r 99" 
  wrapper = "" 
  
  # 4. Construct Final Command
  final_cmd = f"{wrapper} ITGSend {itg_args} {specific_args}"
  
  return final_cmd

# 启动itg命令
def ensure_server_surgical(host_node, start_port=15000, max_retries=3):
  """
  Ensures an ITGRecv instance is listening on a specific port.
  If the port is busy (TIME_WAIT), it tries the next one (9002, 9003...).
  Does NOT kill all ITGRecv processes, preserving background traffic.
  """
  current_port = start_port
  
  for attempt in range(max_retries):
    # A. Surgical Clean: Kill only the process holding this port
    # netstat flags: -n(numeric) -l(listening) -p(show pid)
    # awk vprints the "PID/ProgramName" column
    check_cmd = f"netstat -nlp | grep :{current_port} | awk '{{print $7}}'"
    pid_info = host_node.cmd(check_cmd).strip()
    
    if pid_info:
      pid = pid_info.split('/')[0]
      if pid.isdigit():
        logger.log(f"Port {current_port} busy by PID {pid}. Cleaning...", tag="Recv Start")
        host_node.cmd(f"kill -9 {pid}")
        sleep(0.1) # Yield to OS

    # B. Start New Server (High Priority)
    try:
      # Note: -Sp defines the signaling port. ITGRecv uses this for setup.
      # chrt is used here too so the receiver doesn't drop packets due to CPU load.
      cmd = f"chrt -r 99 ITGRecv -Sp {current_port}"
      
      # Start process via Mininet's popen
      proc = host_node.popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
      
      # C. Verification (The most important step)
      sleep(0.2) # Allow bind
      
      # Check 1: Is process alive?
      if proc.poll() is not None:
        # Process died immediately
        continue 

      # Check 2: Is port actually listening?
      # -u (udp) -t (tcp) -l (listening) -n (numeric)
      out = host_node.cmd(f"netstat -an | grep :{current_port}")
      if str(current_port) in out:
        return proc, current_port # Success!
      
      # If we got here, process is alive but port isn't open? Kill and retry.
      proc.kill()
        
    except Exception as e:
      logger.log(f"Start failed on {current_port}: {e}", tag="Recv Err")
        
    # Increment port and retry
    current_port += 1

  raise RuntimeError(f"Failed to start ITGRecv on {host_node.name} after {max_retries} attempts.")

# --- 3. Safe Client Execution ---
def run_itg_safe(client_node, server_node, log_file, flow_type, duration_sec, timeout_sec, retry_count=0):
  """
  Orchestrates the measurement:
  1. Starts Server (Surgical) -> Gets Port
  2. Starts Client (Safe Popen) -> Sends to that Port
  3. Handles Timeouts -> Sends SIGINT to save logs
  """
  server_proc = None

  try:
    # --- Step 1: Start Server ---
    server_proc, actual_port = ensure_server_surgical(server_node)
    logger.log(f"Server {server_node.name} listening on {actual_port}", tag="Recv Start")

    # --- Step 2: Generate Client Command ---
    # Crucial: Client must send to 'actual_port'
    target_ip = server_node.IP()
    cmd = get_flow_command(
      flow_type=flow_type,
      target_ip=target_ip,
      duration_sec=duration_sec,
      sig_port=actual_port, # Sync ports!
      log_file=log_file )
    
    logger.log(f"{client_node.name} -> {target_ip}:{actual_port} ({flow_type}); Timeout: {timeout_sec}", tag="Send Start")
    logger.log(f"Send command: {cmd}", tag="Send Cmd")
    # --- Step 3: Start Client ---
    # os.setsid creates a new process group, allowing us to kill the whole tree later
    client_proc = client_node.popen(
      cmd, 
      shell=True, 
      stdout=subprocess.PIPE, 
      stderr=subprocess.PIPE,
      preexec_fn=os.setsid 
    )

    # --- Step 4: Wait with Timeout ---
    try:
      stdout, stderr = client_proc.communicate(timeout=timeout_sec)
      # Check for immediate D-ITG errors in stderr
      logger.log(f"{stdout}", tag="Send Out")
      if stderr:
        err_str = stderr.decode('utf-8', errors='ignore')
        if "Connection refused" in err_str or "Connect error" in err_str:
          raise ConnectionError(err_str)
      
      return True # Success

    except subprocess.TimeoutExpired as e:
      logger.log(f"Flow timed out (> {timeout_sec}s). Saving logs...", tag="Send Fail")
      logger.log(f"ITGSend output: {e.stdout}", tag="Send Fail")
      # Graceful Shutdown: Send SIGINT to the Process Group
      # This tells D-ITG to stop sending and flush logs to disk
      try:
        os.killpg(os.getpgid(client_proc.pid), signal.SIGINT)
        client_proc.communicate(timeout=2) # Give it 2s to write file
      except:
        logger.log("Process unresponsive.", tag="Send Fail")
        os.killpg(os.getpgid(client_proc.pid), signal.SIGKILL)
      
      # For TCP, a timeout is a valid result (congestion), not necessarily a crash.
      # We return True so the log parser can see the packet loss/delay.
      return True 

  except ConnectionError as e:
    # --- Step 5: Retry Logic ---
    if retry_count < 2: # Retry once
      logger.log(f"Connection failed. Retrying...", tag="Send Retry")
      # Clean up server before retrying
      if server_proc: 
        server_proc.terminate()
        server_proc.wait()
      return run_itg_safe(client_node, server_node, log_file, flow_type, duration_sec, timeout_sec, retry_count + 1)
    else:
      logger.log(f"Connection refused after retries.", tag="Send Err")
      return False

  except Exception as e:
    logger.log(f"Execution failed: {e}", tag="Send Err")
    return False

  finally:
    # --- Step 6: Cleanup Server ---
    if server_proc:
      try:
        server_proc.terminate()
        server_proc.wait(timeout=1)
      except:
        server_proc.kill()

# 将特征向量归一化
def normalize_fingerprint(tensor: torch.Tensor) -> torch.Tensor:
  """
  对流量指纹 Tensor 进行归一化处理。
  输入形状: (N, 3) -> [Size, IAT, Direction]
  """
  # 1. 克隆 Tensor 以免修改原始数据 (可选)
  norm_tensor = tensor.clone()
  
  # --- 列 0: 包大小 (Size) ---
  # 使用 Min-Max 归一化。
  # 网络包最大通常是 1514 (MTU + Ethernet Header)。
  # 将其缩放到 [0, 1] 范围内。
  norm_tensor[:, 0] = norm_tensor[:, 0] / 1600.0
  
  iat_cap = 0.1  # 100ms
  iat_clamped = torch.clamp(norm_tensor[:, 1], max=iat_cap)   # 截断 (Clamp)
  norm_tensor[:, 1] = iat_clamped / iat_cap                   # 归一化
  
  return norm_tensor


# Editor :
# 管理流表规则

# 清除流表规则
def clean_flow_rules(net, cookie=0xA000, mask=0xF000):
  """
  清理流表。
  :param cookie: 要匹配的 Cookie 值 (前缀)
  :param mask: 掩码。0xF000 表示只匹配前 4 位 (即 0xA...)，忽略后面具体的步数。
  """
  
  # 构造匹配字符串: cookie=0xA000/0xF000
  # 这告诉 OVS: "只要 Cookie 以 A 开头 (高4位是A)，不管后面几位是什么，统统删掉！"
  match_str = f"cookie={hex(cookie)}/{hex(mask)}"
  
  # logger.log(f"Cleaning flows matching {match_str}...", tag="Rule Clean")

  for sw in net.switches:
    # 注意：这里去掉了 /-1，改用我们计算出的掩码
    cmd = f'ovs-ofctl -O OpenFlow13 del-flows {sw.name} "{match_str}"'
    sw.cmd(cmd)
      
  sleep(0.5) # 给 OVS 一点反应时间
  verify_cleanup(net, cookie)

def verify_cleanup(net, cookie=0xA000):
  """
  检查网络中是否还有残留的指定 cookie 的流表。
  返回: True (清理干净了), False (还有残留)
  """
  has_residue = False
  
  for sw in net.switches:
    # 获取该交换机的所有流表
    # 注意: grep 是 shell 命令，通过 sw.cmd 调用
    # result 包含查询结果字符串
    result = sw.cmd(f'ovs-ofctl -O OpenFlow13 dump-flows {sw.name} | grep "cookie={hex(cookie)}"')
    
    # 如果 result 不为空，说明找到了残留
    if result.strip():
      logger.log(f"Residue flows found on {sw.name}:\n{result.strip()}", tag="Clean Err")
      has_residue = True
        
  if not has_residue:
    # logger.log(f"Flow cleanup verified. No rules with cookie={hex(cookie)} found.", tag="Clean OK")
    return True
  else:
    logger.log(f"Flow cleanup failed. Error cookie: {cookie}", tag="Clean Err")
    return False

# 下发流表规则
def install_path_rules(net, path_nodes, cookie, actual_sig_port=15000, tos=None, dst_port=None, protocol='UDP'):
  """
  安装端到端路径流表 (严格模式)
  140: 只放行 ARP、Ping 和 TCP 握手 (SYN)
  150: 放行信令数据和带 ToS 的业务数据
  100: 反向回包保底
  """
  # 1. 获取基础网络对象与 IP
  src_id, dst_id = path_nodes[0], path_nodes[-1]
  h_src, h_dst = net.get(f'h{src_id}'), net.get(f'h{dst_id}')
  src_ip, dst_ip = h_src.IP(), h_dst.IP()

  for i, current_node_id in enumerate(path_nodes):
    sw_name = f's{current_node_id}'
    switch = net.get(sw_name)

    # ==========================================
    # A. 端口预计算 (必须先算好正向和反向，才能下流表)
    # ==========================================
    
    # 1. 正向输出端口 (Out Port)
    out_port = None
    if i == len(path_nodes) - 1:
      links = net.linksBetween(switch, h_dst)
      if links:
        link = links[0]
        out_intf = link.intf1 if link.intf1.node == switch else link.intf2
        out_port = switch.ports[out_intf]
    else:
      next_sw = net.get(f's{path_nodes[i+1]}')
      links = net.linksBetween(switch, next_sw)
      if links:
        link = links[0]
        out_intf = link.intf1 if link.intf1.node == switch else link.intf2
        out_port = switch.ports[out_intf]

    # 2. 反向输出端口 (Reverse Port)
    rev_port = None
    if i == 0:
      links_rev = net.linksBetween(switch, h_src)
      if links_rev:
        link_rev = links_rev[0]
        rev_intf = link_rev.intf1 if link_rev.intf1.node == switch else link_rev.intf2
        rev_port = switch.ports[rev_intf]
    else:
      prev_sw = net.get(f's{path_nodes[i-1]}')
      links_rev = net.linksBetween(switch, prev_sw)
      if links_rev:
        link_rev = links_rev[0]
        rev_intf = link_rev.intf1 if link_rev.intf1.node == switch else link_rev.intf2
        rev_port = switch.ports[rev_intf]

    if out_port is None or rev_port is None:
      continue # 容错处理

    # ==========================================
    # B. 下发流表 (严格匹配逻辑)
    # ==========================================

    # [Priority 140: 基础设施层] 
    
    # [ARP 规则]: 允许二层地址解析。匹配: dl_type=0x0806 (ARP 协议)。
    switch.cmd(f'ovs-ofctl -O OpenFlow13 add-flow {sw_name} "priority=140,dl_type=0x0806,actions=NORMAL"')

    # [Ping 正向]: 匹配: dl_type=0x0800 (IP), nw_proto=1 (ICMP), 目标 IP。
    switch.cmd(f'ovs-ofctl -O OpenFlow13 add-flow {sw_name} "cookie={cookie},priority=140,dl_type=0x0800,nw_proto=1,nw_dst={dst_ip},actions=output:{out_port}"')
    # [Ping 反向]: 匹配: 目标 IP 为源主机。
    switch.cmd(f'ovs-ofctl -O OpenFlow13 add-flow {sw_name} "cookie={cookie},priority=140,dl_type=0x0800,nw_proto=1,nw_dst={src_ip},actions=output:{rev_port}"')

    # [TCP 握手正向]: 匹配: nw_proto=6 (TCP), tcp_flags=0x002/0x002 (仅 SYN 位为 1)。
    switch.cmd(f'ovs-ofctl -O OpenFlow13 add-flow {sw_name} "cookie={cookie},priority=140,dl_type=0x0800,nw_proto=6,tcp_flags=0x002/0x002,nw_dst={dst_ip},actions=output:{out_port}"')
    # [TCP 握手反向]: 匹配: 反向路径的 SYN/ACK。
    switch.cmd(f'ovs-ofctl -O OpenFlow13 add-flow {sw_name} "cookie={cookie},priority=140,dl_type=0x0800,nw_proto=6,tcp_flags=0x002/0x002,nw_dst={src_ip},actions=output:{rev_port}"')
    # tcp_flags=0x010/0x010 匹配 ACK 位。
    switch.cmd(f'ovs-ofctl -O OpenFlow13 add-flow {sw_name} "cookie={cookie},priority=140,tcp,nw_dst={dst_ip},tcp_flags=0x010/0x010,actions=output:{out_port}"')
    # [Priority 150: 业务通道层] 
    if tos is not None:
      # [信令通道 15000]: 匹配: TCP 协议, 目的端口 15000。
      switch.cmd(f'ovs-ofctl -O OpenFlow13 add-flow {sw_name} "cookie={cookie},priority=150,dl_type=0x0800,nw_proto=6,tp_dst={actual_sig_port},nw_dst={dst_ip},actions=output:{out_port}"')
      # [信令回包]: 匹配: 源端口 15000 (ACK)。
      switch.cmd(f'ovs-ofctl -O OpenFlow13 add-flow {sw_name} "cookie={cookie},priority=150,dl_type=0x0800,nw_proto=6,tp_src={actual_sig_port},nw_dst={src_ip},actions=output:{rev_port}"')

      # [VIP 数据流]: 匹配: 指定协议, 指定 ToS, 目标 IP。
      p_num = 6 if protocol.upper() == 'TCP' else 17
      match_data = f"cookie={cookie},priority=150,dl_type=0x0800,nw_proto={p_num},nw_tos={tos},nw_dst={dst_ip}"
      # 仅 UDP 精确匹配数据端口，TCP 建议放宽端口以兼容。
      if protocol.upper() == 'UDP' and dst_port:
        match_data += f",tp_dst={dst_port}"
      
      switch.cmd(f'ovs-ofctl -O OpenFlow13 add-flow {sw_name} "{match_data},actions=output:{out_port}"')

    # [Priority 100: 反向回包保底] 
    
    # [反向保底]: 匹配: 只要发往源主机的 IP 包。用于承载所有 ACK/Reply，优先级最低。
    switch.cmd(f'ovs-ofctl -O OpenFlow13 add-flow {sw_name} "cookie={cookie},priority=100,dl_type=0x0800,nw_dst={src_ip},actions=output:{rev_port}"')

# 根据gnn输出的 logits来生成路径--贪婪/概率选择：
# 替换 MS/Env/MininetController.py 中的 sample_path 函数
def sample_path(edge_logits, edge_index, s_node, d_node, max_steps=100, G_fallback=None, greedy=False):
  """
  增强版路径采样：支持 Masking (防环) + Dijkstra Fallback (兜底)
  返回: path, log_prob_sum, ai_success, path_complete
  """
  # 1. 构建邻接表
  adj = {}
  num_edges = edge_index.shape[1]
  for i in range(num_edges):
    u = edge_index[0, i].item()
    v = edge_index[1, i].item()
    if u not in adj: adj[u] = []
    adj[u].append((v, i)) 

  current = s_node
  path = [current]
  visited = {current}
  log_probs = [] 
  
  # ---------------------------------------------------------
  # A. AI 游走阶段
  # ---------------------------------------------------------
  for _ in range(max_steps):
    if current == d_node: break
    if current not in adj: break
    
    # Action Masking: 禁止走回头路
    neighbors = adj[current]
    valid_options = [n for n in neighbors if n[0] not in visited]
    if not valid_options: break # 死胡同
    
    # 提取 Logits
    candidate_logits = []
    candidate_nodes = []
    for next_node, edge_idx in valid_options:
      candidate_logits.append(edge_logits[edge_idx])
      candidate_nodes.append(next_node)
      
    logits_tensor = torch.stack(candidate_logits)
    
    if greedy:
      action_idx = torch.argmax(logits_tensor).item()
      log_prob = torch.tensor(0.0).to(logits_tensor.device)
    else:
      probs = torch.softmax(logits_tensor, dim=0)
      dist = torch.distributions.Categorical(probs)
      action_tensor = dist.sample()
      action_idx = action_tensor.item()
      log_prob = dist.log_prob(action_tensor)
    
    log_probs.append(log_prob)
    
    next_hop = candidate_nodes[action_idx]
    path.append(next_hop)
    visited.add(next_hop)
    current = next_hop

  # ---------------------------------------------------------
  # B. 状态判定与兜底 (Safety Net) [核心修复]
  # ---------------------------------------------------------
  ai_success = False      # AI 是否独立完成
  path_complete = False   # 最终路径是否连通 (含兜底)

  if path[-1] == d_node:
    # Case 1: AI 独立成功
    ai_success = True
    path_complete = True
  else:
    # Case 2: AI 失败，尝试 Dijkstra 兜底
    if G_fallback is not None:
      try:
        # 从断点找最短路补全
        remaining_path = nx.shortest_path(G_fallback, source=path[-1], target=d_node, weight='delay')
        path.extend(remaining_path[1:]) # 拼接到路径后
        
        ai_success = False    # AI 没完成
        path_complete = True  # 但路通了
      except nx.NetworkXNoPath:
        # 物理隔离，彻底失败
        ai_success = False
        path_complete = False
    else:
      # 没有兜底机制，且 AI 没走通
      ai_success = False
      path_complete = False
  
  # 汇总 Log Probs
  if log_probs:
    log_prob_sum = torch.stack(log_probs).sum()
  else:
    log_prob_sum = torch.tensor(0.0).to(edge_logits.device)

  # 返回 4 个值：路径，梯度概率，AI是否成功，最终是否连通
  return path, log_prob_sum, ai_success, path_complete
  
# 监视器，负责动态提取mininet拓扑状态
class NetworkMonitor:
  def __init__(self, net):
    self.net = net

  def _read_sys_file(self, path):
    """快速读取系统文件"""
    try:
      with open(path, 'r') as f:
        return int(f.read().strip())
    except:
      return 0

  def _get_queue_limit(self, node_name, intf_name):
    """
    动态读取接口的队列上限 (Capacity)
    返回: limit (int, 单位: packets)
    """
    try:
      node = self.net.get(node_name)
      # -d 参数用于显示详细配置 (details)，包含 limit
      cmd = f"tc -d qdisc show dev {intf_name}"
      output = node.cmd(cmd)
      
      # 1. 尝试匹配 'limit 100p' (pfifo/bfifo_fast 等)
      # 这里的 p 代表 packets
      match_p = re.search(r'limit\s+(\d+)p', output)
      if match_p:
        return int(match_p.group(1))
      
      # 2. 尝试匹配 'limit 10000b' (bfifo，基于字节)
      # 如果是字节，我们需要估算包数。假设平均包大小 1500B
      match_b = re.search(r'limit\s+(\d+)b', output)
      if match_b:
        bytes_limit = int(match_b.group(1))
        return max(1, bytes_limit // 1500)
      
      # 3. 如果没找到 limit，通常是默认的 txqueuelen (通常是 1000)
      # 可以读取 /sys/class/net/.../tx_queue_len
      cmd_tx = f"cat /sys/class/net/{intf_name}/tx_queue_len"
      tx_len = int(node.cmd(cmd_tx).strip())
      return tx_len
        
    except Exception as e:
      logger.log(f"Error reading queue limit: {e}", tag="Monit Err")
      return 100 # 兜底默认值

  def _get_all_interfaces_stats(self, G):
    """
    获取图中所有涉及接口的 tx_bytes 和 backlog
    返回: dict { (u, v): {'bytes': int, 'qlen': int} }
    """
    stats = {}
    for u, v in G.edges():
      # 映射节点名
      s_u_name = f"Ts{u}" if f"Ts{u}" in self.net else f"s{u}"
      s_v_name = f"Ts{v}" if f"Ts{v}" in self.net else f"s{v}"
      
      if s_u_name not in self.net: continue
      s_u = self.net.get(s_u_name)
      s_v = self.net.get(s_v_name)
      
      # 查找接口
      links = self.net.linksBetween(s_u, s_v)
      if not links: continue
      link = links[0]
      intf = link.intf1 if link.intf1.node == s_u else link.intf2
      
      # 1. 读取 Bytes (直接读宿主机文件，假设是在 Root Namespace 或路径正确)
      # 注意: Mininet 的虚拟接口通常在 Root NS 可见 (/sys/class/net/s1-eth2/...)
      # 如果读不到，可能需要用 node.cmd('cat ...')，但那样太慢。
      # 这里假设是标准 Mininet OVS 环境，接口都在 Root NS。
      tx_bytes = self._read_sys_file(f"/sys/class/net/{intf.name}/statistics/tx_bytes")
      
      # 2. 读取队列 (只能通过 tc 命令，较慢，但在 sampling 期间只读一次也行)
      # 为了速度，这里我们可以只在采样结束时读一次队列
      # 这里先存名字，稍后处理
      stats[(u, v)] = {'bytes': tx_bytes, 'intf': intf.name, 'node': s_u}
        
    return stats

  def sync_state_to_graph(self, G: nx.Graph, duration = 0.05):
    """
    [主动采样模式]
    休眠一小段时间，计算精确的瞬时速率。
    """
    SAMPLE_WINDOW = duration # 采样窗口 50ms
    
    # 1. 第一次快照
    snapshot1 = self._get_all_interfaces_stats(G)
    t1 = time()
    # 2. 等待
    sleep(SAMPLE_WINDOW)
    # 3. 第二次快照
    snapshot2 = self._get_all_interfaces_stats(G)
    t2 = time()
    
    delta_t = t2 - t1
    if delta_t <= 0: delta_t = 1e-6

    node_stats_buffer = {n: [] for n in G.nodes()}
    node_stats_util   = {n: [] for n in G.nodes()}

    # 4. 计算并更新
    for u, v, data in G.edges(data=True):
      key = (u, v)
      if key not in snapshot1 or key not in snapshot2:
        continue
          
      # --- A. 计算瞬时利用率 ---
      b1 = snapshot1[key]['bytes']
      b2 = snapshot2[key]['bytes']
      
      # 速率 (Mbps)
      speed_mbps = ((b2 - b1) * 8) / (delta_t * 1_000_000)
      
      capacity = data.get('capacity', 30.0)
      assert(capacity >= 30)
      util = min(speed_mbps / (capacity + 1e-6), 1.0)
      
      # 写入图
      # 更新 (平滑一点)

      data['utilization'] = 0.3 * data.get('utilization', util) + 0.7 * util
      data['measured_speed'] = speed_mbps

      # --- B. 获取瞬时队列 (Buffer) ---
      # 队列长度只需要读一次（取最新状态）
      # 解析 tc 输出
      node = snapshot2[key]['node']
      intf_name = snapshot2[key]['intf']
      try:
        # 这是一个 shell 调用，稍微耗时，但比读文件快
        tc_out = node.cmd(f"tc -s qdisc show dev {intf_name}")
        match = re.search(r'backlog\s+\d+b\s+(\d+)p', tc_out)
        q_len = int(match.group(1)) if match else 0
      except:
        q_len = 0
      max_q = 2000  # 假设最大队列长度为 2000 packets
      buf_occ = min(q_len / max_q, 1.0)
      if u in node_stats_buffer:
        node_stats_buffer[u].append(buf_occ)
        node_stats_util[u].append(util)

    # 写入节点特征 (源节点 Buffer)
    # 更新源节点 u 的状态
    for n in G.nodes():
      # --- 聚合 Buffer (取最大值) ---
      # 含义：如果有一个方向堵了，这个节点就标红
      buffer_list = node_stats_buffer[n]
      if buffer_list:
        max_buffer = max(buffer_list) 
      else:
        max_buffer = 0.0
        
      util_list = node_stats_util[n]
      if util_list:
        max_util = max(util_list)
      else:
        max_util = 0.0

      # 简单估算 Proc Delay
      proc_delay = 0.01 * (1 + 5 * max_util**2) # 拥塞时 CPU 处理变慢

      G.nodes[n]['buffer_occupancy'] = 0.3 * G.nodes[n].get('buffer_occupancy', 0) + 0.7 * max_buffer
      G.nodes[n]['proc_delay'] = 0.3 * G.nodes[n].get('proc_delay', 0)             + 0.7 * proc_delay

    return G
