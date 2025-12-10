import os
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
from MS.Env import VerbosePrint as vp
vprint = vp.vprint
from .FlowGenerator import FlowType, FLOW_PROFILES
from mininet.node import OVSKernelSwitch, RemoteController
from mininet.link import TCLink
from MS.Env.E_model import (
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
    vprint("[Error] can't catch output str -- the str is None")
    return metrics

  vprint("[Parse] parsing ditg output ...")
  # print(output_str)
  try:
    # --- 1. 提取平均延迟 (Average delay) ---
    # 示例行: Average delay            =     0.000234 s
    delay_match = re.search(r"Average delay\s+=\s+([-\d\.nan]+)\s+s", output_str)
    if delay_match:
      val = delay_match.group(1)
      if 'nan' not in val.lower(): # 过滤掉 -nan
        metrics['delay'] = float(val) * 1000.0 # 秒 -> 毫秒
    else:
      vprint("[Error Parse] no delay found")
      # --- 2. 提取平均抖动 (Average jitter) ---
      # 示例行: Average jitter           =     0.000012 s
    
    jitter_match = re.search(r"Average jitter\s+=\s+([-\d\.nan]+)\s+s", output_str)
    if jitter_match:
      val = jitter_match.group(1)
      if 'nan' not in val.lower():
        metrics['jitter'] = float(val) * 1000.0 # 秒 -> 毫秒
    else:
      vprint("[Error Parse] no jitter found")

    # --- 3. 提取吞吐量 (Average bitrate) ---
    # 示例行: Average bitrate          =  4096.000000 Kbit/s
    bitrate_match = re.search(r"Average bitrate\s+=\s+([-\d\.nan]+)\s+Kbit/s", output_str)
    if bitrate_match:
      val = bitrate_match.group(1)
      if 'nan' not in val.lower():
        metrics['bandwidth'] = float(val) / 1000.0 # Kbit/s -> Mbps
    else:
      vprint("[Error Parse] no bitrate found")

    # --- 4. 提取丢包率 (Packets dropped) ---
    # 示例行: Packets dropped          =            5 (0.50 %)
    # 注意：如果没有发包成功，分母为0可能导致 nan，或者 dropped 为 0 但 total 也为 0
    loss_match = re.search(r"Packets dropped\s+=\s+\d+\s+\(([-\d\.nan]+)\s+%\)", output_str)
    if loss_match:
      val = loss_match.group(1)
      if 'nan' not in val.lower():
        metrics['loss_rate'] = float(val) 
    else: 
      vprint("[Error Parse] no loss found")

    # 如果总包数 (Total packets) 为 0，说明完全没通，强制设置最差指标
    total_pkts_match = re.search(r"Total packets\s+=\s+(\d+)", output_str)
    if total_pkts_match and int(total_pkts_match.group(1)) == 0:
      no_packet_arrive = True
      metrics['loss_rate'] = 1.0
      metrics['bandwidth'] = 0.0

  except Exception as e:
    vprint(f"[Error Parse] 解析 D-ITG 输出时出错: {e}")
    vprint(f"原始输出片段:\n{output_str[:200]}") # 调试用

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

def measure_path_qos(server, client, path_route, flow_type, resend = False):
  """
  使用 D-ITG 测量路径 QoS (基于内存文件系统 /dev/shm)
  """

  # 1. 准备内存日志路径 (使用 /dev/shm 实现“伪管道”)
  # 使用 uuid 防止文件名冲突
  random_id = uuid.uuid4().hex[:8]
  log_prefix = f"/dev/shm/itg_{client.name}_{server.name}_{random_id}"
  recv_log = f"{log_prefix}.recv"  # D-ITG 会自动加上后缀，但我们在命令里显式指定更安全
  
  duration_sec = 6 if flow_type==FlowType.STREAMING else 2
  # 构建命令
  cmd = get_flow_command(
    flow_type=flow_type,
    target_ip=server.IP(),
    log_file=recv_log,
    duration_sec=duration_sec)
  
  
  try:
    # 启动接听命令
    server_proc = server.popen("nice -n 2 ITGRecv -Sp 9001")
    stdout, stderr = run_itg_safe(client, cmd, timeout_sec=duration_sec+2)
  except Exception as e:
    vprint(f"[Error Send] 实验执行出错: {e}")
  finally:
    # 清理服务端
    if server_proc:
      try:
        server_proc.terminate()
        server_proc.wait(timeout=1)
      except:
        server_proc.kill()
      sleep(1)
  
  # if stderr: return 0
  # 检查文件是否存在 (防止传输完全失败导致无日志)
  check_log = server.cmd(f"ls {recv_log}")
  if "No such file" in check_log:
    vprint("[Error Send] No log generated. resend same cmd again...")
    client.cmd(f"rm -f {recv_log}")
    return measure_path_qos(server, client, path_route, flow_type)
      
  # 运行解码器拿到文本结果 
  # 解析结果 
  # Meta-DRL 
  try:
    vprint(f"[Decoder] Running ITGDec on {recv_log}...")

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
      stdout, stderr = dec_proc.communicate(timeout=5) # 设置个超时防止卡死

      if dec_proc.returncode != 0:
        vprint(f"[Decoder Error] ITGDec failed with code {dec_proc.returncode}: {stderr}")
        dec_output = "" # 失败则为空
      else:
        dec_output = stdout
        vprint("[Decoder] success dec recieve file")

  except subprocess.TimeoutExpired:
    vprint("[Decoder Error] ITGDec Timed out!")
    dec_proc.kill()
    dec_output = ""
  finally:
    # 移除临时日志文件
    client.cmd(f"rm -f {recv_log}")

  # if flow_type == FlowType.STREAMING:
  #   vprint(f"[Streaming] \n{dec_output}")
  qos_metrics, no_packet_arrive = parse_ditg_output(dec_output)    
  if no_packet_arrive:
    if not resend :
      vprint(f"No packet arrive : Resend cmd ...")
      return measure_path_qos(server, client, path_route, flow_type, True)
    else :
      vprint(f"Fail to send packet, bad path")
      return -1

  vprint(f"[QoS] successfully get {flow_type} QoS: ")
  vprint(f"      delay:     {qos_metrics['delay']}")
  vprint(f"      jitter:    {qos_metrics['jitter']}")
  vprint(f"      bandwidth: {qos_metrics['bandwidth']}")
  vprint(f"      loss_rate: {qos_metrics['loss_rate']}")

  # 计算 Reward
  reward = calculate_qoe_reward(qos_metrics, FLOW_PROFILES[flow_type])

  vprint(f"[QoE] calculate reward: {reward}")
  return reward

def vprint_network_status(G):
  """打印全网高负载链路"""
  vprint("-" * 60)
  vprint(f"📊 [Global] High Load Links (>10%):")
  vprint(f"  {'Link':<12} | {'Cap (Mbps)':<10} | {'Util %':<8} | {'Status'}")
  vprint("-" * 60)
  
  count = 0
  for u, v, data in G.edges(data=True):
    util = data.get('utilization', 0.0)
    cap = data.get('capacity', 100.0)
    if util > 0.1: # 只显示活跃链路
      status = "\033[91mFULL\033[0m" if util > 0.9 else "BUSY"
      vprint(f"  {u:<2} <-> {v:<2}    | {cap:<10.1f} | {util:<8.2%} | {status}")
      count += 1
  if count == 0:
    vprint("  (No congested links found)")
  vprint("-" * 60)

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
      bw = data.get('bandwidth', 200)
      delay = f"{data.get('delay', 1)}ms"
      loss = data.get('loss', 0)
      q_limit = data.get('queue_size', 20)
      rate_bytes = bw * 1000000 / 8
      # 2. 计算最佳 r2q (确保 quantum ≈ 1500)
      r2q = int(max(1, rate_bytes / 1500))
      # 这里沿用 Mininet 构造函数中设置的 r2q
      self.addLink(f'{test_str}s{u}', f'{test_str}s{v}', cls=TCLink, bw=bw, delay=delay, loss=loss, r2q = r2q, use_htb=True, max_queue_size=q_limit) 

# mininet 启动
@contextmanager
def get_a_mininet(g: nx.Graph, is_test=False, remote_port=None):
  if remote_port:
    controller = partial(RemoteController, ip='127.0.0.1', port=remote_port)
  else:
    controller = None

  # if not vp.MININET_VERBOSE:
  setLogLevel('critical')

  net = Mininet(
    topo=GraphTopo(g, is_test),
    switch=OVSKernelSwitch,
    link=TCLink,
    controller=controller,
    autoSetMacs=True, 
    autoStaticArp=True)

  try:
    vprint("[Mini] Disabling TCP Offload (TSO/GSO/GRO) on all switches...")
    for h in net.hosts:
      for intf in h.intfList():
        if intf.name != 'lo':
          # 使用 ethtool 关闭卸载
          h.cmd(f"ethtool -K {intf.name} tso off gso off gro off > /dev/null 2>&1")
    for sw in net.switches:
      for intf in sw.intfList():
        if intf.name != 'lo':
          # 使用 ethtool 关闭卸载
          sw.cmd(f"ethtool -K {intf.name} tso off gso off gro off > /dev/null 2>&1")
          
    net.start()
    yield net
  finally:
    vprint("[Mini] stopping mininet ...")
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
    # vprint(f"{flow_type.name}====={final_tensor.size(0)}")
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
  # 发送流并抓包
  """
  在 Mininet 中运行 D-ITG 流量, 并同时使用 tshark 管道实时捕获特征。

  参数:
    net: Mininet 网络对象。
    flow_type (str): 'voip', 'gaming', 'streaming'.
    duration_sec (int): D-ITG 流量的*总*运行时长。
    n_packets_to_capture (int): tshark 在捕获 N 个包后自动停止。
    **flow_params: 传递给 generate_ditg_command 的额外参数。
  
  返回:
    torch.tensor: 形状为 (N, 3) 的特征矩阵 [[Size, IAT], ...]。
  """

  server_ip = server.IP()
  client_ip = client.IP()
  
  # print(f"server ip: {server_ip} client_ip: {client_ip}")
  # 2. 找到要监听的接口 (s1-eth1)
  server_intf = None
  for intf in server.intfList():
    if intf.name != 'lo' and intf.link: # 确保它不是 'lo' 并且已连接
      server_intf = intf
      break
  if server_intf is None:
    raise Exception(f"在 {server.name} 上找不到已连接的数据接口!")

  switch_intf = server_intf.link.intf2 if server_intf.link.intf1 == server_intf else server_intf.link.intf1
  switch_intf_name = switch_intf.name
  
  # 3. [Action 1] 获取 D-ITG 命令
  client_cmd = get_flow_command(
    flow_type=flow_type,
    target_ip=server_ip,
    duration_sec=duration_sec,
    **flow_params)
  MARK_TOS = 32
  # 4. 准备 tshark 命令 (这是最快的方法)
  display_filter = f"src host {client_ip} and dst host {server_ip} and ip[1] == {MARK_TOS}"
  timeout_duration = duration_sec+5

  tshark_cmd = [
    'sudo',
    'tshark',
    '-c', str(n_packets_to_capture), # 抓 N 个包后停止
    '-a', f'duration:{timeout_duration}',
    '-i', switch_intf_name,
    '-l', # 行缓冲 (实时)
    '-T', 'fields',
    '-e', 'frame.len',        # 特征 1: Size
    '-e', 'frame.time_delta', # 特征 2: IAT
    '-e', 'ip.src',
    '-e', 'ip.dst',
    '-E', 'separator=,',
    '-f', display_filter]
  
  feature_matrix = []
  client_proc = None
  tshark_proc = None
  server_proc = None

  try:
    # print(f"[Capture] 启动 tshark 管道: {' '.join(tshark_cmd)}")
    tshark_proc = subprocess.Popen(
      tshark_cmd, 
      stdout=subprocess.PIPE, 
      stderr=subprocess.DEVNULL,
      text=True
    )

    # 5. [Action 2] 启动流量
    # print(f"[Net] 启动 D-ITG 接收端 (h1)...")
    server_proc = server.popen('ITGRecv')
  
    # 6. [Action 3] 启动 tshark 捕获管道
    sleep_time= 1.2 if flow_type==FlowType.STREAMING else 0.1
    sleep(sleep_time)
    # print(f"[Net] 启动 D-ITG 发送端 (h2): {client_cmd}")
    client_proc = client.popen(client_cmd)
    # 7. [核心] 实时从管道读取并封装向量
    for line in tshark_proc.stdout:
      line = line.strip()
      if not line:
        continue
      # print(f"[RAW CAPTURE] 抓到了: {line}")
      try:

        size_str, iat_str, src_ip, dst_ip = line.split(',')
        
        size = float(size_str)
        
        # 处理第一个包 (IAT 不是数字)
        try:
          iat = float(iat_str)
        except ValueError:
          iat = 0.0 # 第一个包的 IAT 为 0
        
        # 实时封装成向量
        feature_vector = [size, iat]
        feature_matrix.append(feature_vector)
      except ValueError as e:
        print(f"[Parser] 跳过 tshark 行: {line}. 错误: {e}")
      
  except Exception as e:
    print(f"[Error] 实验执行出错: {e}")
  finally:
    # print("[Net] 清理进程...")
    # 清理 tshark
    if tshark_proc:
      tshark_proc.kill()
    
    # 清理客户端
    if client_proc:
      client_proc.kill() # 确保杀死
    # 清理服务端
    if server_proc:
      server_proc.kill()

  # print(f"[Capture] 捕获完成. 获得 {len(feature_matrix)} 个向量。")
  fingerprint_tensor = torch.tensor(feature_matrix, dtype=float)

  return fingerprint_tensor

# 根据流类型，返回不同的 D-ITG 命令。
def get_flow_command(
  flow_type: str, 
  target_ip: str, 
  duration_sec: int, 
  log_file: str=None,
  **kwargs) -> str:
    """
    根据流量模式和参数, 生成一个 D-ITG (ITGSend) 命令字符串。
    所有命令均使用 D-ITG (ITGSend) 工具。

    参数:
      flow_type (str): 流量模式。支持: 'voip', 'gaming', 'streaming'.
      target_ip (str): 目标服务器的 IP 地址 (例如: '10.0.0.1').
      duration_sec (int): 流量的总持续时间 (秒).
    
    返回:
      str: 一个完整的、可在 Mininet 主机上运行的 ITGSend 命令字符串。
    """
    MARK_TOS = 32
    profile = FLOW_PROFILES[flow_type]

    protocol = profile['protocol']
    duration_ms = duration_sec * 1000
    if not log_file == None: 
      log_str = f"-x {log_file}"
    else :
      log_str = ""
      
    base_cmd = f"nice -n -1 ITGSend -a {shlex.quote(target_ip)} {log_str} -rp 12000 -Sdp 9001 -t {duration_ms} -b {MARK_TOS} -T {protocol}"
    
    if 'ditg_preset' in profile:
      # 使用预设 (VoIP, Gaming)
      specific_args = profile['ditg_preset']
    elif 'ditg_manual' in profile:
      # 使用手动参数 (Streaming)
      specific_args = profile['ditg_manual']
    else:
      raise ValueError("Profile 配置不完整")
   

    # 组合命令, 并在末尾添加 '&' 使其在后台运行
    final_cmd = f"{base_cmd} {specific_args}"
    
    return final_cmd

# 启动itg命令
def run_itg_safe(h_src, cmd, timeout_sec=20):
  """
  安全运行 ITGSend：如果超时，发送 SIGINT 让其写入日志后退出。
  """
  vprint(f"[Sender] 启动命令 {cmd} (超时限制 {timeout_sec}s)...")
  
  # 注意：这里我们使用 h_src.popen 的底层 subprocess 对象
  # start_new_session=True 是为了能够向进程组发送信号 (可选，视 Mininet 实现而定)
  proc = h_src.popen(cmd.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
  
  try:
    # 1. 正常等待 (Wait)
    # 如果 D-ITG 在规定时间内跑完，这里会正常返回
    stdout, stderr = proc.communicate(timeout=timeout_sec)
      
  except subprocess.TimeoutExpired:
    # 2. 发生超时 (Hang) -> 触发熔断
    vprint(f"[Timeout] ITFSend 发送卡死 (> {timeout_sec}s)！正在强制结算...")
    
    # 【关键步骤】发送 SIGINT (Ctrl+C)
    # 这告诉 D-ITG: "别发了，赶紧写日志收工！"
    proc.send_signal(signal.SIGINT)
      
    try:
      # 给它 1-2 秒时间处理后事 (写文件)
      stdout, stderr = proc.communicate(timeout=2)
      vprint("[Safe Kill] 进程已优雅退出，日志应已保存。")
    except subprocess.TimeoutExpired:
      # 3. 敬酒不吃吃罚酒 -> 强杀
      vprint("[Force Kill] 进程无响应，执行 SIGKILL。")
      proc.kill()
      stdout, stderr = proc.communicate()
          
  # 打印输出以便调试
  if stderr:
    vprint(f"[Error] Stderr: {stderr.decode('utf-8')}")
    vprint(f"[Send ] resend same cmd...")
    return run_itg_safe(h_src, cmd, timeout_sec)
      
  return stdout, stderr

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
  
  vprint(f"[Cleaner] Cleaning flows matching {match_str}...")

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
      vprint(f"[Error Clean] Residue flows found on {sw.name}:\n{result.strip()}")
      has_residue = True
        
  if not has_residue:
    vprint(f"[Check] Flow cleanup verified. No rules with cookie={hex(cookie)} found.")
    return True
  else:
    return False

# 下发流表规则
def install_path_rules(net, path_nodes, tos=None, dst_port=None, cookie=0x1234, do_ping=True):
  """
  安装端到端路径流表 (完整版)
  
  参数:
    net: Mininet 网络对象
    path_nodes: 节点ID列表 (如 [0, 1, 2])
    tos: 流量 ToS 标记 (None=泛洪, 32=Agent, 184=Background)
    dst_port: 目的端口 (None=通配所有端口, 12000=精确匹配)
    cookie: 流表标记
    do_ping: 是否执行 Ping 测试 (仅在 tos=None 时建议开启)
  """
  # 1. 获取源主机和目的主机对象
  src_id = path_nodes[0]
  dst_id = path_nodes[-1]
  
  h_src = net.get(f'h{src_id}')
  h_dst = net.get(f'h{dst_id}')
  dst_ip = h_dst.IP()
  src_ip = h_src.IP()
  
  # Debug 信息
  port_str = f"Port={dst_port}" if dst_port else "Port=ANY"
  # vprint(f"[Install] h{src_id}->h{dst_id} | ToS={tos} | {port_str}")

  # 2. 遍历路径上的每一跳交换机
  for i, current_node_id in enumerate(path_nodes):
    sw_name = f's{current_node_id}'
    switch = net.get(sw_name)
    
    # ==========================================
    # A. 计算正向输出端口 (Out Port)
    # ==========================================
    out_port = None
    if i == len(path_nodes) - 1:
      # Case: 最后一跳 -> 目的主机
      links = net.linksBetween(switch, h_dst)
      if links:
        link = links[0]
        # 确定 switch 侧的接口
        out_intf = link.intf1 if link.intf1.node == switch else link.intf2
        out_port = switch.ports[out_intf]
    else:
      # Case: 中间跳 -> 下一跳交换机
      next_node_id = path_nodes[i+1]
      next_switch = net.get(f's{next_node_id}')
      links = net.linksBetween(switch, next_switch)
      if links:
        link = links[0]
        out_intf = link.intf1 if link.intf1.node == switch else link.intf2
        out_port = switch.ports[out_intf]
    
    if out_port is None:
      print(f"❌ Error: Cannot find link from {sw_name}")
      continue

    # ==========================================
    # B. 下发正向规则 (Forwarding Rules)
    # ==========================================

    if tos is not None:
      # === QoS 专用模式 (区分业务) ===
      
      # [规则 1.A]: D-ITG 默认信令 (TCP 9000) - 给背景流用
      cmd_sig = (f'ovs-ofctl -O OpenFlow13 add-flow {sw_name} '
                 f'"cookie={cookie},priority=150,dl_type=0x0800,'
                 f'nw_proto=6,tp_dst=9000,nw_dst={dst_ip},actions=output:{out_port}"')
      switch.cmd(cmd_sig)

      # [规则 1.B]: D-ITG VIP 信令 (TCP 9001) - 给智能体流用 (NEW!)
      # 必须加这条，否则 Step 2 的 VIP 通道不通！
      cmd_sig_vip = (f'ovs-ofctl -O OpenFlow13 add-flow {sw_name} '
                 f'"cookie={cookie},priority=150,dl_type=0x0800,'
                 f'nw_proto=6,tp_dst=9001,nw_dst={dst_ip},actions=output:{out_port}"')
      switch.cmd(cmd_sig_vip)
      
      # [规则 2]: 业务数据流 (UDP) - 优先级 150
      # 动态构建匹配条件
      match_str = (f"cookie={cookie},priority=150,dl_type=0x0800,"
                   f"nw_proto=17,nw_tos={tos},nw_dst={dst_ip}")
      
      # 如果指定了端口，加入精确匹配；否则通配所有端口
      if dst_port is not None:
        match_str += f",tp_dst={dst_port}"
      
      cmd_data = f'ovs-ofctl -O OpenFlow13 add-flow {sw_name} "{match_str},actions=output:{out_port}"'
      switch.cmd(cmd_data)
      
    else:
      # === 兼容/泛洪模式 (Ping) ===
      # [规则 3]: 通用 IP 转发 - 优先级 100
      cmd_gen = (f'ovs-ofctl -O OpenFlow13 add-flow {sw_name} '
                 f'"cookie={cookie},priority=100,dl_type=0x0800,'
                 f'nw_dst={dst_ip},actions=output:{out_port}"')
      switch.cmd(cmd_gen)

    # ==========================================
    # C. 计算反向输出端口 (Reverse Port)
    # ==========================================
    rev_port = None
    if i == 0:
      # Case: 第一跳 -> 源主机 (回包终点)
      links_rev = net.linksBetween(switch, h_src)
      if links_rev:
        link_rev = links_rev[0]
        rev_intf = link_rev.intf1 if link_rev.intf1.node == switch else link_rev.intf2
        rev_port = switch.ports[rev_intf]
    else:
      # Case: 中间跳 -> 上一跳交换机
      prev_node_id = path_nodes[i-1]
      prev_switch = net.get(f's{prev_node_id}')
      links_rev = net.linksBetween(switch, prev_switch)
      if links_rev:
        link_rev = links_rev[0]
        rev_intf = link_rev.intf1 if link_rev.intf1.node == switch else link_rev.intf2
        rev_port = switch.ports[rev_intf]

    if rev_port is None:
      print(f"[Error] : Cannot find reverse link from {sw_name}")
      continue

    # ==========================================
    # D. 下发反向规则 (Reverse Rules)
    # ==========================================
    
    # [规则 4]: 反向 IP 回包 - 优先级 100
    # 确保 TCP ACK (信令回应) 和 Ping Reply 能回来
    # 这里放宽条件，匹配所有发往 src_ip 的包
    cmd_rev_ip = (f'ovs-ofctl -O OpenFlow13 add-flow {sw_name} '
                  f'"cookie={cookie},priority=100,dl_type=0x0800,'
                  f'nw_dst={src_ip},actions=output:{rev_port}"')
    switch.cmd(cmd_rev_ip)

  # 3. 连通性验证 (仅在非 QoS 模式下推荐，因为 Ping 不带 ToS)
  if do_ping and tos is None:
    loss = net.ping([h_src, h_dst], timeout=0.1)
    if loss > 50:
      vprint(f"[Warning] High ping loss ({loss}%) - Path{path_nodes} might be broken")
    else:
      vprint(f"[Mini] Path installed successfully.")

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
      print(f"Error reading queue limit: {e}")
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
      
      capacity = data.get('bandwidth', 10.0)
      util = min(speed_mbps / (capacity + 1e-6), 1.0)
      
      # 写入图
      data['utilization'] = util
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
      
      # 写入节点特征 (源节点 Buffer)
      # 更新源节点 u 的状态
      node_u = G.nodes[u]
      max_q = 50.0
      buf_occ = min(q_len / max_q, 1.0)
      
      # 简单估算 Proc Delay
      proc_delay = 0.01 * (1 + 5 * util**2) # 拥塞时 CPU 处理变慢
      
      # 更新 (平滑一点)
      node_u['buffer_occupancy'] = 0.3 * node_u.get('buffer_occupancy', 0) + 0.7 * buf_occ
      node_u['proc_delay'] = 0.3 * node_u.get('proc_delay', 0) + 0.7 * proc_delay

    return G
