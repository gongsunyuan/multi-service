import os
import subprocess
from time import sleep, time
import re 
from mininet.net import Mininet
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
from MS.Env.VebosePrint import vprint
from .FlowGenerator import FlowType, FLOW_PROFILES
from mininet.node import OVSKernelSwitch, RemoteController
from mininet.link import TCLink

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
  
  
  if not output_str:
    vprint("[Error] can't catch output str -- the str is None")
    return metrics

  try:
    # --- 1. 提取平均延迟 (Average delay) ---
    # 示例行: Average delay            =     0.000234 s
    delay_match = re.search(r"Average delay\s+=\s+([-\d\.nan]+)\s+s", output_str)
    if delay_match:
      val = delay_match.group(1)
      if 'nan' not in val.lower(): # 过滤掉 -nan
        metrics['delay'] = float(val) * 1000.0 # 秒 -> 毫秒
    else:
      vprint("[Error] no delay found")
      # --- 2. 提取平均抖动 (Average jitter) ---
      # 示例行: Average jitter           =     0.000012 s
    jitter_match = re.search(r"Average jitter\s+=\s+([-\d\.nan]+)\s+s", output_str)
    if jitter_match:
      val = jitter_match.group(1)
      if 'nan' not in val.lower():
        metrics['jitter'] = float(val) * 1000.0 # 秒 -> 毫秒
    else:
      vprint("[Error] no jitter found")

    # --- 3. 提取吞吐量 (Average bitrate) ---
    # 示例行: Average bitrate          =  4096.000000 Kbit/s
    bitrate_match = re.search(r"Average bitrate\s+=\s+([-\d\.nan]+)\s+Kbit/s", output_str)
    if bitrate_match:
      val = bitrate_match.group(1)
      if 'nan' not in val.lower():
        metrics['bandwidth'] = float(val) / 1000.0 # Kbit/s -> Mbps
    else:
      vprint("[Error] no bitrate found")

      # --- 4. 提取丢包率 (Packets dropped) ---
      # 示例行: Packets dropped          =            5 (0.50 %)
      # 注意：如果没有发包成功，分母为0可能导致 nan，或者 dropped 为 0 但 total 也为 0
      loss_match = re.search(r"Packets dropped\s+=\s+\d+\s+\(([-\d\.nan]+)\s+%\)", output_str)
      if loss_match:
        val = loss_match.group(1)
        if 'nan' not in val.lower():
          metrics['loss_rate'] = float(val) / 100.0 # 0.50% -> 0.005
      else: 
        vprint("[Error] no loss found")
      
      # 如果总包数 (Total packets) 为 0，说明完全没通，强制设置最差指标
      total_pkts_match = re.search(r"Total packets\s+=\s+(\d+)", output_str)
      if total_pkts_match and int(total_pkts_match.group(1)) == 0:
        vprint("[Error] no packets arrive !!")
        metrics['loss_rate'] = 1.0
        metrics['bandwidth'] = 0.0
      else:
        vprint(f"[get QoS] successfully get Qoe: {metrics}")

  except Exception as e:
    vprint(f"[Parser] 解析 D-ITG 输出时出错: {e}")
    if MININET_VERBOSE: vprint(f"原始输出片段:\n{output_str[:200]}") # 调试用

  return metrics

def calculate_qoe_reward(qos_metrics: dict, flow_profile: dict) -> float:
  """
  【新增】根据 QoE “悬崖”计算奖励。
  这是 RL 训练信号的核心。
  """
  
  # 默认奖励为正，表示成功
  reward = 10.0 
  
  # 获取 QoE 关键阈值
  qoe_reqs = flow_profile.get('qoe_critical', {})
  
  # 1. 检查延迟 (来自 Ping)
  max_delay = qoe_reqs.get('max_delay')
  if max_delay is not None and qos_metrics.get('delay', 1000.0) > max_delay:
    vprint(f"REWARD PENALTY: Delay cliff! {qos_metrics.get('delay')} > {max_delay}")
    return -100.0 # 巨大的惩罚
      
  # 2. 检查抖动 (来自 iperf -u)
  max_jitter = qoe_reqs.get('max_jitter')
  if max_jitter is not None and qos_metrics.get('jitter', 1000.0) > max_jitter:
    vprint(f"REWARD PENALTY: Jitter cliff! {qos_metrics.get('jitter')} > {max_jitter}")
    return -100.0 # 巨大的惩罚

  # 3. 检查带宽 
  min_bw = qoe_reqs.get('min_bandwidth')
  if min_bw is not None and qos_metrics.get('bandwidth', 0.0) < min_bw:
    vprint(f"REWARD PENALTY: Bandwidth cliff! {qos_metrics.get('bandwidth')} < {min_bw}")
    return -100.0 # 巨大的惩罚
      
  # 4. 检查丢包率 (来自 iperf -u)
  max_loss = qoe_reqs.get('max_loss_rate')
  if max_loss is not None and qos_metrics.get('loss_rate', 1.0) > max_loss:
    vprint(f"REWARD PENALTY: Loss cliff! {qos_metrics.get('loss_rate')} > {max_loss}")
    return -100.0 # 巨大的惩罚

  # 如果通过了所有悬崖测试，可以返回一个更精细的奖励
  # (例如，VoIP 可以基于 E-Model 计算 R-factor)
  # 简单起见，我们先返回一个固定的正奖励
  # TODO
  return reward

def measure_path_qos(server, client, path_route, flow_type):
  """
  使用 D-ITG 测量路径 QoS (基于内存文件系统 /dev/shm)
  """

  # 1. 准备内存日志路径 (使用 /dev/shm 实现“伪管道”)
  # 使用 uuid 防止文件名冲突
  random_id = uuid.uuid4().hex[:8]
  log_prefix = f"/dev/shm/itg_{client.name}_{server.name}_{random_id}"
  recv_log = f"{log_prefix}.recv"  # D-ITG 会自动加上后缀，但我们在命令里显式指定更安全
  
  # 构建命令
  cmd = get_flow_command(
    flow_type=flow_type,
    target_ip=server.IP(),
    duration_sec=3)
  
  cmd += f" -x {recv_log}"
  
  try:
    # 启动接听命令
    server_proc = server.popen("ITGRecv")
    # 执行发送 (阻塞)
    client_proc = client.popen(cmd)
    try:
      client_proc.wait(timeout=10) 
    except Exception:
      # 如果超时还没停，说明可能卡死了，再强制杀
      vprint("[Error] client send flow timeout")
  except Exception as e:
    vprint(f"[Error] 实验执行出错: {e}")
  finally:
    if client_proc:
      try:
        client_proc.terminate()
        client_proc.wait(timeout=1)
      except:
        client_proc.kill() # 确保杀死
    # 清理服务端
    if server_proc:
      try:
        server_proc.terminate()
        server_proc.wait(timeout=1)
      except:
        server_proc.kill()
  
  # 检查文件是否存在 (防止传输完全失败导致无日志)
  check_log = server.cmd(f"ls {recv_log}")
  if "No such file" in check_log:
    vprint("[ERROR] No log generated. Link might be down.")
    return -100.0 # 惩罚
      
  # 运行解码器拿到文本结果 


  # 解析结果 
  
  
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
      stdout, stderr = dec_proc.communicate(timeout=10) # 设置个超时防止卡死

      if dec_proc.returncode != 0:
        vprint(f"   [Decoder Error] ITGDec failed with code {dec_proc.returncode}: {stderr}")
        dec_output = "" # 失败则为空
      else:
        dec_output = stdout
        vprint("    [Decoder] success dec recieve file")
        
    qos_metrics = parse_ditg_output(dec_output)

  except subprocess.TimeoutExpired:
    vprint("    [Decoder Error] ITGDec Timed out!")
    dec_proc.kill()
    dec_output = ""

  # print(f"--- QoS: {qos_metrics} ---")

  # 立即删除内存文件 
  # 虽然是在内存里，但也要清理以防占满 RAM
  client.cmd(f"rm -f {recv_log}")

  # 计算 Reward
  reward = calculate_qoe_reward(qos_metrics, FLOW_PROFILES[flow_type])
  
  return reward

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
      bw = data.get('bandwidth', 1000)
      delay = f"{data.get('delay', 1)}ms"
      loss = data.get('loss', 0)
      q_limit = data.get('queue_size', 100)
      # 这里沿用 Mininet 构造函数中设置的 r2q
      self.addLink(f'{test_str}s{u}', f'{test_str}s{v}', delay=delay, loss=loss, use_htb=True, max_queue_size=q_limit) 

# mininet 启动
@contextmanager
def get_a_mininet(g: nx.Graph, is_test=False, remote_port=None):
  if remote_port:
    controller = partial(RemoteController, ip='127.0.0.1', port=remote_port)
  else:
    print("[ms] 没有输入远程控制器，请自己配置流表规则！")
    controller = None

  setLogLevel('error')

  net = Mininet(
    topo=GraphTopo(g, is_test),
    switch=OVSKernelSwitch,
    link=TCLink,
    controller=controller
  )

  try:
    net.start()
    yield net
  finally:
    net.stop()

  return net

# 获取一个流量特征张量
def get_a_fingerprint(
  server, 
  client, 
  flow_type: FlowType, 
  n_packets_to_capture=30, 
  **flow_params
  ):
  duration_sec = 15 

  final_tensor = send_packet_and_capture(
    server=server,
    client=client,
    flow_type=flow_type,
    duration_sec=duration_sec,
    n_packets_to_capture=n_packets_to_capture)
  
  while final_tensor.size(0) < 30:
    vprint(f"{flow_type.name}----{final_tensor.size(0)}")
    sleep(1)
    final_tensor = send_packet_and_capture(
      server=server,
      client=client,
      flow_type=flow_type,
      duration_sec=duration_sec,
      n_packets_to_capture=n_packets_to_capture)
  
  # if flow_type==FlowType.STREAMING:
  #   print(f"{flow_type.name}{final_tensor.size(0)}")
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

  # 4. 准备 tshark 命令 (这是最快的方法)
  display_filter = f"src host {client_ip} and dst host {server_ip}"
  timeout_duration = duration_sec+1

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
    # 5. [Action 2] 启动流量
    # print(f"[Net] 启动 D-ITG 接收端 (h1)...")
    server_proc = server.popen('ITGRecv')
  
    # 6. [Action 3] 启动 tshark 捕获管道
    # print(f"[Capture] 启动 tshark 管道: {' '.join(tshark_cmd)}")
    tshark_proc = subprocess.Popen(
      tshark_cmd, 
      stdout=subprocess.PIPE, 
      stderr=subprocess.DEVNULL,
      text=True
    )
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
    profile = FLOW_PROFILES[flow_type]

    protocol = profile['protocol']
    duration_ms = duration_sec * 1000
    
    # 基础命令 (所有模式通用), 明确使用 D-ITG 的 ITGSend
    base_cmd = f"ITGSend -a {shlex.quote(target_ip)} -t {duration_ms} -T {protocol}"
    
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
def clean_flow_rules(net, cookie=0x1234):
  for sw in net.switches:
    sw.cmd(f'ovs-ofctl -O OpenFlow13 del-flows {sw.name} "cookie={cookie}/-1"')
  sleep(0.5)

def verify_cleanup(net, cookie=0x1234):
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
      vprint(f"[Warning] Residue flows found on {sw.name}:\n{result.strip()}")
      has_residue = True
        
  if not has_residue:
    vprint(f"[Check] Flow cleanup verified. No rules with cookie={hex(cookie)} found.")
    return True
  else:
    return False
# 下发流表规则
def install_path_rules(net, path_nodes, cookie=0x1234):
  """
  将逻辑路径转化为 OpenFlow 流表规则并下发。
  支持双向联通 (TCP/ARP 需要回路)，或者仅单向 (UDP)。
  
  参数:
    net: Mininet 网络对象
    path_nodes: 节点 ID 列表，例如 [0, 2, 5] 代表 s0 -> s2 -> s5
    cookie: 规则标记，用于清理
  """
  # 1. 获取源主机和目的主机对象
  # 根据 GraphTopo 的命名规则: 节点 i -> 主机 hi, 交换机 si
  src_id = path_nodes[0]
  dst_id = path_nodes[-1]
  
  h_src = net.get(f'h{src_id}')
  h_dst = net.get(f'h{dst_id}')
  dst_ip = h_dst.IP()
  src_ip = h_src.IP()
  
  # print(f"[Controller] Installing path: h{src_id} -> ... -> h{dst_id}")

  # 2. 遍历路径上的每一跳交换机
  for i, current_node_id in enumerate(path_nodes):
    sw_name = f's{current_node_id}'
    switch = net.get(sw_name)
    
    # --- 确定输出端口 (Out Port) ---
    if i == len(path_nodes) - 1:
      # Case A: 最后一跳 (s_dst)，要去往主机 h_dst
      # 获取 switch 连接到 h_dst 的端口
      # linksBetween 返回 [(intf1, intf2)], 我们需要 switch 侧的 intf
      links = net.linksBetween(switch, h_dst)
      if not links: continue
      link = links[0]
      # 确定哪个接口属于 switch
      out_intf = link.intf1 if link.intf1.node == switch else link.intf2
      out_port = switch.ports[out_intf]
    else:
      # Case B: 中间跳，要去往下一跳交换机 s_next
      next_node_id = path_nodes[i+1]
      next_switch = net.get(f's{next_node_id}')
      
      links = net.linksBetween(switch, next_switch)
      if not links: continue
      link = links[0]
      out_intf = link.intf1 if link.intf1.node == switch else link.intf2
      out_port = switch.ports[out_intf]

    # --- 下发规则 (Forwarding) ---
    # 匹配: 目的 IP 是 dst_ip
    # 动作: 转发到 out_port
    # 优先级: 100 (高于默认规则)
    cmd = (
      f'ovs-ofctl -O OpenFlow13 add-flow {sw_name} '
      f'"cookie={cookie},priority=100,dl_type=0x0800,nw_dst={dst_ip},actions=output:{out_port}"'
    )
    switch.cmd(cmd)
    
    # --- [重要] 下发反向规则 (Reverse Path) ---
    # TCP 握手和 iperf 结束报告需要回包。
    # 简单起见，我们让反向流量沿原路返回。
    if i == 0:
      # 第一跳的反向出口是去往 h_src
      links_rev = net.linksBetween(switch, h_src)
      if links_rev:
        link_rev = links_rev[0]
        rev_intf = link_rev.intf1 if link_rev.intf1.node == switch else link_rev.intf2
        rev_port = switch.ports[rev_intf]
    else:
      # 中间跳的反向出口是去往上一跳交换机 s_prev
      prev_node_id = path_nodes[i-1]
      prev_switch = net.get(f's{prev_node_id}')
      links_rev = net.linksBetween(switch, prev_switch)
      if links_rev:
        link_rev = links_rev[0]
        rev_intf = link_rev.intf1 if link_rev.intf1.node == switch else link_rev.intf2
        rev_port = switch.ports[rev_intf]
    
    # 反向规则: 匹配目的 IP 是 src_ip (即回包)
    cmd_rev = (
      f'ovs-ofctl -O OpenFlow13 add-flow {sw_name} '
      f'"cookie={cookie},priority=100,dl_type=0x0800,nw_dst={src_ip},actions=output:{rev_port}"')
    switch.cmd(cmd_rev)

    # --- [可选] 处理 ARP ---
    # 如果没有 Controller 处理 ARP，需要广播 ARP 请求
    # 动作: output:FLOOD (或者 normal 如果有默认学习规则)
    switch.cmd(f'ovs-ofctl -O OpenFlow13 add-flow {sw_name} "cookie={cookie},priority=100,dl_type=0x0806,actions=FLOOD"')
    sleep(1)
    # After install_path_rules()
  loss = net.ping([h_src, h_dst], timeout=4)
  if loss > 50:
    vprint(f"[Warning] High ping loss ({loss}%) - path might be broken")
  else:
    vprint(f"[ms] install path success !")

# 根据gnn输出的 logits来生成路径--贪婪/概率选择：
# 替换 MS/Env/MininetController.py 中的 sample_path 函数
def sample_path(edge_logits, edge_index, s_node, d_node, max_steps=100, greedy=False):
  """
  通用路径采样函数 (支持 RL 梯度计算)。
  返回: path (list), log_prob_sum (tensor), is_success (bool)
  """
  # 1. 构建邻接表
  adj = {}
  num_edges = edge_index.shape[1]
  for i in range(num_edges):
    u = edge_index[0, i].item()
    v = edge_index[1, i].item()
    if u not in adj: adj[u] = []
    adj[u].append((v, i)) # (邻居节点, 边索引)

  current = s_node
  path = [current]
  visited = {current}
  log_probs = [] # [新增] 用于存储每一步的 log_prob
  
  for _ in range(max_steps):
    if current == d_node: break
    if current not in adj: break
    
    # 获取合法邻居
    neighbors = adj[current]
    valid_options = [n for n in neighbors if n[0] not in visited]
    if not valid_options: break
    
    # 提取候选边的 Logits
    candidate_logits = []
    candidate_nodes = []
    for next_node, edge_idx in valid_options:
      candidate_logits.append(edge_logits[edge_idx])
      candidate_nodes.append(next_node)
        
    # --- 核心决策逻辑 ---
    logits_tensor = torch.stack(candidate_logits)
    
    if greedy:
      # [验证模式]
      action_idx = torch.argmax(logits_tensor).item()
      # greedy 模式下 log_prob 通常设为 0 或不计算，因为不进行梯度更新
      log_prob = torch.tensor(0.0).to(logits_tensor.device)
    else:
      # [RL 训练模式]
      probs = torch.softmax(logits_tensor, dim=0)
      dist = torch.distributions.Categorical(probs)
      action_tensor = dist.sample()
      action_idx = action_tensor.item()
      
      # [关键] 计算这一步的对数概率，保留梯度图
      log_prob = dist.log_prob(action_tensor)
    
    # 记录
    log_probs.append(log_prob)
    
    next_hop = candidate_nodes[action_idx]
    path.append(next_hop)
    visited.add(next_hop)
    current = next_hop
      
  is_success = (path[-1] == d_node)
  
  # [新增] 汇总整条路径的 log_prob (加法)
  if log_probs:
    log_prob_sum = torch.stack(log_probs).sum()
  else:
    log_prob_sum = torch.tensor(0.0).to(edge_logits.device)

  # 返回 3 个值
  return path, log_prob_sum, is_success

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

  def sync_state_to_graph(self, G: nx.Graph):
    """
    [主动采样模式]
    休眠一小段时间，计算精确的瞬时速率。
    """
    SAMPLE_WINDOW = 0.05 # 采样窗口 50ms
    
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


