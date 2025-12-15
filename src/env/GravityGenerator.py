import random
import numpy as np
import networkx as nx
import time
from enum import Enum
from collections import defaultdict
from src.utils import VerbosePrint as vp

vprint = vp.vprint

class FlowType(Enum):
  VOIP = 1
  STREAMING = 2
  GAMING = 3

# 针对 NSFNet (瓶颈 30Mbps, 核心 200Mbps) 的流量规格配置
FLOW_SPECS = {
  FlowType.VOIP: {
    'protocol': 'UDP', 'pkt_size': 200, 
    'min_bw': 0.1, 'max_bw': 0.5
  },
  FlowType.STREAMING: {
    'protocol': 'TCP', 'pkt_size': 1000, 
    # 设为 2.0 - 12.0 Mbps
    # 这样 3 条流就能堵死 30M 的瓶颈链路，体现汇聚效应
    'min_bw': 2.0, 'max_bw': 25.0
  },
  FlowType.GAMING: {
    'protocol': 'UDP', 'pkt_size': 100, 
    'min_bw': 0.5, 'max_bw': 2.0 
  }
}

class GravityGenerator:
  def __init__(self, G_nx: nx.Graph):
    """
    初始化生成器
    :param G_nx: NetworkX 拓扑图
    """
    self.G = G_nx
    self.nodes = list(G_nx.nodes())
    self.node_mass = {}
    self.raw_capacity = {} # 用于调试显示的原始容量

    # --- 1. 计算节点质量 (Mass) ---
    # 节点的质量 = 它连接的所有链路的总带宽
    for n in self.nodes:
      total_cap = 0.0
      for nbr in self.G.neighbors(n):
        edge_data = self.G[n][nbr]
        # 优先读取 capacity (d4), 其次读取 bandwidth (d1), 默认 100.0
        cap = float(edge_data.get('capacity', edge_data.get('bandwidth', 100.0)))
        total_cap += cap
      
      self.raw_capacity[n] = total_cap
      self.node_mass[n] = total_cap

    # 归一化质量 (最大值为 1.0)
    max_val = max(self.node_mass.values()) if self.node_mass else 1.0
    for n in self.nodes:
      self.node_mass[n] /= max_val
      
    vprint(f"[Gravity] Init complete. Max Mass Node: {max(self.node_mass, key=self.node_mass.get)}")

  def print_node_masses(self):
    """
    [调试功能 1] 打印所有节点的重力质量
    """
    print("\n" + "="*60)
    print(f"{'[DEBUG] Node Gravity Analysis':^60}")
    print("="*60)
    print(f" {'Node ID':<8} | {'Total Cap (Mbps)':<18} | {'Gravity (Mass)':<15}")
    print("-" * 60)
    
    # 按重力从大到小排序
    sorted_nodes = sorted(self.nodes, key=lambda n: self.node_mass[n], reverse=True)
    
    for n in sorted_nodes:
      raw = self.raw_capacity[n]
      mass = self.node_mass[n]
      # 高亮显示 Mass > 0.8 的核心节点
      star = "*" if mass > 0.8 else " "
      print(f" {n:<8} | {raw:<18.1f} | {mass:<8.4f} {star}")
    print("="*60 + "\n")

  def analyze_traffic_impact(self, flows):
    """
    [调试功能 2] 预演流量矩阵，计算每条链路的理论需求负载
    注意：这里将双向流量合并统计，以显示链路的总体繁忙程度
    """
    # link_load[(u, v)] = total_mbps
    link_load = defaultdict(float)
    
    # 1. 模拟路由 (假设背景流都走最短路)
    for f in flows:
      src, dst, bw = f['src'], f['dst'], f['bw']
      try:
        path = nx.shortest_path(self.G, src, dst)
        # 累加路径上每一跳的负载
        for i in range(len(path) - 1):
          u, v = path[i], path[i+1]
          # 统一存为无向边 (u, v) 其中 u < v，方便聚合统计
          if u > v: u, v = v, u
          link_load[(u, v)] += bw
      except nx.NetworkXNoPath:
        continue

    # 2. 打印报表
    print("\n" + "="*80)
    print(f"{'[DEBUG] Link Utilization Prediction (Demand / Capacity)':^80}")
    print("="*80)
    print(f" {'Link (u-v)':<12} | {'Cap (Mbps)':<10} | {'Demand':<10} | {'Util %':<8} | {'Status'}")
    print("-" * 80)

    # 获取所有边并排序
    all_edges = []
    for u, v, data in self.G.edges(data=True):
      if u > v: u, v = v, u
      cap = float(data.get('capacity', data.get('bandwidth', 100.0)))
      load = link_load[(u, v)]
      util = (load / cap) * 100.0
      all_edges.append((u, v, cap, load, util))

    # 按利用率降序排序 (最堵的在前面)
    all_edges.sort(key=lambda x: x[4], reverse=True)

    for u, v, cap, load, util in all_edges:
      # 状态判定
      if util > 100.0:   status = "\033[91mOVERLOAD !!!\033[0m" # 红色
      elif util > 80.0:  status = "\033[93mCongested\033[0m"    # 黄色
      elif util > 50.0:  status = "\033[96mBusy\033[0m"         # 青色
      else:              status = "OK"

      # 只显示有负载的，或者利用率 > 1% 的
      if util > 1.0:
        print(f" {u:<2} <-> {v:<2}    | {cap:<10.1f} | {load:<10.1f} | {util:<6.1f} % | {status}")
    
    print("="*80 + "\n")

  def generate_snapshot(self, total_load_mbps=500.0, sparsity=0.3):
    """
    生成流量矩阵快照并自动进行分析
    :param total_load_mbps: 全网总流量需求目标
    :param sparsity: 稀疏度 (0.0 - 1.0)
    """
    traffic_flows = []
    
    # --- 1. 计算引力分数 ---
    pairs = []
    scores = []
    
    for u in self.nodes:
      for v in self.nodes:
        if u == v: continue
        # Gravity = Mass(u) * Mass(v)
        score = self.node_mass[u] * self.node_mass[v]
        pairs.append((u, v))
        scores.append(score)
        
    scores = np.array(scores)
    # 防止全零
    if scores.sum() == 0: scores[:] = 1.0
    probs = scores / scores.sum()
    
    # --- 2. 概率采样 ---
    num_flows = int(len(pairs) * sparsity)
    selected_indices = np.random.choice(
      len(pairs), size=num_flows, replace=False, p=probs
    )
    
    # --- 3. 分配流量 ---
    # 平均每条流的目标带宽
    avg_bw_per_flow = total_load_mbps / max(1, num_flows)
    mean_score = scores.mean()

    for idx in selected_indices:
      u, v = pairs[idx]
      score = scores[idx]
      
      # 随机业务类型 (增加 STREAMING 比例以制造压力)
      ftype = np.random.choice(
        [FlowType.VOIP, FlowType.GAMING, FlowType.STREAMING], 
        p=[0.3, 0.3, 0.4] 
      )
      spec = FLOW_SPECS[ftype]
      
      # 带宽计算: 平均值 * 引力系数 * 随机扰动
      gravity_factor = score / mean_score 
      raw_bw = avg_bw_per_flow * gravity_factor * random.uniform(0.6, 1.4)
      
      # 钳制 1: 业务类型物理限制
      bw = max(spec['min_bw'], min(raw_bw, spec['max_bw']))
      
      # 钳制 2: 源端保护 (防止单条流超过边缘链路 1000M 的 10%)
      bw = min(bw, 100.0) 

      traffic_flows.append({
        'src': u, 'dst': v, 
        'type': ftype, 
        'bw': round(bw, 2),
        'pkt_size': spec['pkt_size'], 
        'protocol': spec['protocol']
      })

    vprint(f"[Gravity] Generated {len(traffic_flows)} flows. Total Demand: {sum(f['bw'] for f in traffic_flows):.1f} Mbps")
    
    # --- 4. 执行调试分析 ---
    self.print_node_masses()
    self.analyze_traffic_impact(traffic_flows)
    
    return traffic_flows

  def apply_to_mininet(self, net, flow_list, install_rules_func, duration=15):
    """
    Injects background traffic using 'Ghost Traffic' strategy.
    
    Args:
        net: Mininet object
        flow_list: List of dicts generated by generate_snapshot
        install_rules_func: Function to install OVS rules (install_path_rules)
        duration: Duration in seconds
    """
    vprint(f"[TM] Injecting {len(flow_list)} flows (Ghost Strategy, {duration}s)...")
    
    BG_COOKIE = 0xB000
    BG_TOS = 184 # High Priority / Background Marking
    
    # --- Phase 0: Ensure Receivers ---
    vprint("[TM] Restarting ITGRecv daemons...")
    for h in net.hosts:
      h.cmd("killall -9 ITGRecv > /dev/null 2>&1")
      h.popen("ITGRecv")
    
    time.sleep(1)

    # --- Phase 1: Install Forwarding Rules + Drop Trap ---
    vprint("[TM] Installing routing rules & Ghost drop policies...")
    configured_drops = set()

    for flow in flow_list:
      u, v = flow['src'], flow['dst']
      try:
        # Calculate Shortest Path
        path_nodes = nx.shortest_path(self.G, u, v, weight=None)
      except nx.NetworkXNoPath:
        continue

      # 1.1 Install Forwarding Rules (Path)
      if len(path_nodes) > 1:
        # Use provided function to install rules along the path
        install_rules_func(net, path_nodes, tos=BG_TOS, dst_port=11000, cookie=BG_COOKIE, do_ping=False)

      # 1.2 Install Drop Trap (The "Ghost" Logic)
      # We drop the packet at the LAST switch before the host.
      # This generates load on the network links but saves the destination Host CPU.
      dst_host = net.get(f'h{v}')
      dst_ip = dst_host.IP()
      
      # The last switch in the path (connected to dst host)
      last_switch_name = f's{path_nodes[-1]}' 
      last_switch = net.get(last_switch_name)
      
      drop_key = (last_switch_name, dst_ip)
      
      # Only install one drop rule per destination switch to avoid duplicates
      if drop_key not in configured_drops:
        # Match: Cookie, Priority=200 (Higher than forwarding), IP Dest, ToS
        # Action: DROP
        cmd_drop = (
          f'ovs-ofctl -O OpenFlow13 add-flow {last_switch_name} '
          f'"cookie={hex(BG_COOKIE)},priority=200,dl_type=0x0800,'
          f'nw_dst={dst_ip},nw_proto=17,nw_tos={BG_TOS},actions=drop"'
        )
        # nw_proto=17 means UDP. We only drop UDP background traffic. 
        # TCP control traffic (if any) might still need to pass.
        
        last_switch.cmd(cmd_drop)
        configured_drops.add(drop_key)

    time.sleep(1.0) # Wait for OVS rules to propagate

    # --- Phase 2: Start Senders ---
    vprint("[TM] Starting Senders...")
    bg_processes = []
    
    for i, flow in enumerate(flow_list):
      src = net.get(f"h{flow['src']}")
      dst = net.get(f"h{flow['dst']}")
      dst_ip = dst.IP()
      
      # Calculate PPS based on flow-specific packet size
      # Overhead: 20(IP) + 8(UDP) + 18(Eth) approx 46 bytes
      packet_size_bits = (flow['pkt_size'] + 46) * 8
      pps = min(int(flow['bw'] * 1_000_000 / packet_size_bits), 20000)
      
      if pps < 1: pps = 1
      
      # Construct ITGSend command
      cmd = (
        f"ITGSend -a {dst_ip} "
        f"-T {flow['protocol']} "
        f"-C {pps} "
        f"-c {flow['pkt_size']} "
        f"-t {duration * 1000} "
        f"-rp 11000 "
        f"-b {BG_TOS} &" # Run in background
      )
      
      # Execute
      src.cmd(cmd)
      
      if i == 0:
        vprint(f"[DEBUG CHECK] First Flow CMD: {cmd}")

    return bg_processes
