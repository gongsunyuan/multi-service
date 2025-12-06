import random
import networkx as nx
import time
import heapq
import os  # 需要引入 os 模块来创建目录
from enum import Enum
from MS.Env import VerbosePrint as vp

vprint = vp.vprint

# === 配置与定义 ===
class FlowType(Enum):
  VOIP = 1
  STREAMING = 2
  GAMING = 3

FLOW_PROFILES = {
  FlowType.VOIP: {
    'type': 'VOIP', 'protocol': 'UDP', 'ditg_preset': 'VoIP -x G.711.2',
    'qoe_critical': {'max_delay': 150, 'max_jitter': 50}, 'reward_fn': 'E-Model'
  },
  FlowType.STREAMING: {
    'type': 'STREAMING', 'protocol': 'TCP',
    'ditg_manual': '-B U 500 1000 C 100 -c 1460 -C 1000',
    'qoe_critical': {'min_bandwidth': 5, 'max_loss_rate': 1e-6}, 'reward_fn': '3GPP-QCI6'
  },
  FlowType.GAMING: {
    'type': 'GAMING', 'protocol': 'UDP', 'ditg_preset': 'CSa',
    'qoe_critical': {'max_delay': 50, 'max_jitter': 30}, 'reward_fn': '3GPP-QCI80'
  }
}

class FlowGenerator:
  def __init__(self):
    pass

  def get_random_flow(self) -> tuple[FlowType, dict]:
    """随机选择一个流类型及其配置文件。"""
    flow_type = random.choice(list(FlowType))
    profile = FLOW_PROFILES[flow_type]
    return flow_type, profile

  # =========================================================================
  # [核心模块] 背景流量生成 (Gravity Model + Aggregation)
  # =========================================================================

  def generate_traffic_matrix(self, nodes, total_load_mbps=500.0):
    """
    [Step 1: Generate]
    使用重力模型生成流量矩阵，并进行聚合优化。
    """
    # 1. 分配随机权重
    node_weights = {node: random.uniform(0.1, 5.0) for node in nodes}
    total_weight = sum(node_weights.values())
    tm = {}
    
    # 2. 计算原始矩阵
    for u in nodes:
      for v in nodes:
        if u == v: continue
        interaction = node_weights[u] * node_weights[v]
        bw = (interaction / (total_weight ** 2)) * total_load_mbps
        # 过滤掉微小流
        if bw > 1:
          tm[(u, v)] = bw

    # 3. [流量聚合] 限制最大并发数，保护 CPU
    MAX_BG_FLOWS = 20

    return tm

    if len(tm) > MAX_BG_FLOWS:
      # 选出 Top N 大流
      top_flows = heapq.nlargest(MAX_BG_FLOWS, tm.items(), key=lambda x: x[1])
      
      # 计算缩放因子，保持总负载不变
      total_original = sum(tm.values())
      total_top = sum([bw for _, bw in top_flows])
      scale = total_original / total_top if total_top > 0 else 1.0
      
      # 重构矩阵
      final_tm = {k: v * scale for k, v in top_flows}
      vprint(f"[TM] Aggregated: {len(tm)} -> {len(final_tm)} flows (Scale x{scale:.2f})")
      return final_tm
    else:
      return tm

  def simulate_tm_on_graph(self, G, tm_dict):
    """
    [Step 2: Simulate]
    在内存中预演流量矩阵，计算利用率、延迟和丢包。
    """
    edge_loads = {e: 0.0 for e in G.edges()}

    # 模拟最短路路由
    for (s, d), bw in tm_dict.items():
      try:
        path = nx.shortest_path(G, s, d, weight=None)
        for i in range(len(path) - 1):
          u, v = path[i], path[i+1]
          if (u, v) in edge_loads: edge_loads[(u, v)] += bw
          elif (v, u) in edge_loads: edge_loads[(v, u)] += bw
      except nx.NetworkXNoPath:
        continue 

    # 更新图属性 (MM1 模型)
    for u, v, data in G.edges(data=True):
      capacity = float(data.get('capacity', 100.0))
      prop_delay = float(data.get('base_delay', 5.0)) # 假设原始延迟存为 base_delay

      current_load = edge_loads.get((u, v), 0.0) + edge_loads.get((v, u), 0.0)
      rho = min(current_load / (capacity + 1e-6), 0.999)
      
      # M/M/1 延迟公式
      queue_delay = 10.0 * (rho / (1.0 - rho))
      total_delay = prop_delay + min(queue_delay, 500.0)
      
      # 丢包率模型 (Soft Threshold)
      if rho < 0.8: loss = 0.0
      else: loss = 0.05 * ((rho - 0.8) / 0.2) ** 2

      data['utilization'] = rho
      data['bandwidth'] = capacity * (1.0 - rho)
      data['delay'] = total_delay
      data['loss'] = loss

    return G

  def apply_traffic_matrix_to_mininet(self, net, tm_dict, G_nx, install_rules_func, duration=1000):
    """
    [Step 4: Execute]
    注入背景流 (Ghost Traffic)。
    增强功能：详细的生命周期打印 & 独立日志记录。
    """
    print(f"\n[TM] 🚀 Injecting {len(tm_dict)} background flows (Ghost Strategy)...")
    print(f"[TM] 🕒 Duration set to: {duration} seconds (Ensure this > target flow time!)")
    
    BG_COOKIE = 0xB000
    BG_TOS = 184
    
    # 创建统一的日志目录，方便排查
    log_dir = "/tmp/bg_logs"
    os.makedirs(log_dir, exist_ok=True)
    # 清理旧日志 (可选)
    os.system(f"rm -f {log_dir}/*.log")

    # ==========================================
    # Phase 0: 确保接收端在线 (Signaling Ready)
    # ==========================================
    print("[TM] 🛠️  Starting ITGRecv daemons on all hosts...")
    for h in net.hosts:
      # 建议记录 Recv 日志以便排查控制平面问题
      h.cmd(f"ITGRecv -l {log_dir}/recv_{h.name}.log > {log_dir}/recv_{h.name}.out 2>&1 &")
    
    # ==========================================
    # Phase 1: 铺设路径 & 设置陷阱
    # ==========================================
    print("[TM] 🚧 Installing routing rules & Ghost drop policies...")
    configured_drops = set()

    for (u, v) in tm_dict.keys():
      try:
        path_nodes = nx.shortest_path(G_nx, u, v, weight=None)
      except:
        continue

      # 1.1 全程铺路 (Forwarding)
      if len(path_nodes) > 1:
        install_rules_func(net, path_nodes, tos=BG_TOS, dst_port=11000, cookie=BG_COOKIE, do_ping=False)

      # 1.2 终点设卡 (Drop UDP Only)
      dst_host = net.get(f'h{v}')
      dst_ip = dst_host.IP()
      last_switch_name = f's{path_nodes[-1]}'
      last_switch = net.get(last_switch_name)
      
      drop_key = (last_switch_name, dst_ip)
      if drop_key not in configured_drops:
        # [CRITICAL FIX] 只丢弃 UDP 数据包 (nw_proto=17)，放行 TCP 握手信令
        cmd_drop = (
          f'ovs-ofctl -O OpenFlow13 add-flow {last_switch_name} '
          f'"cookie={hex(BG_COOKIE)},priority=200,dl_type=0x0800,'
          f'nw_dst={dst_ip},nw_proto=17,nw_tos={BG_TOS},actions=drop"'
        )
        last_switch.cmd(cmd_drop)
        configured_drops.add(drop_key)

    time.sleep(1.0) # 稍微多给点时间让流表同步到 Datapath

    # ==========================================
    # Phase 2: 启动发送端 (带详细监控)
    # ==========================================
    print(f"\n{'='*60}")
    print(f"{'Src':<5} -> {'Dst':<5} | {'BW (Mbps)':<10} | {'PPS':<8} | {'PID':<8} | {'Log File'}")
    print(f"{'-'*60}")

    bg_processes = [] # 存储进程对象，防止被垃圾回收

    count = 0
    for (u, v), bw in tm_dict.items():
      h_src = net.get(f'h{u}')
      h_dst = net.get(f'h{v}')
      dst_ip = h_dst.IP()
      
      # 计算 PPS
      pps = min(int(bw * 1_000_000 / 8000), 3000)
      if pps < 1: pps = 1
      
      # 日志文件路径
      log_file = f"{log_dir}/send_h{u}_to_h{v}.log"

      # ITGSend 命令
      # 使用 nohup 或直接后台运行，并将输出重定向到文件
      cmd_send = (
        f"ITGSend -a {dst_ip} "
        f"-T UDP "
        f"-C {pps} "
        f"-c 1000 "
        f"-rp 11000 "
        f"-t {duration * 1000} "
        f"-b {BG_TOS} "
      )
      
      # 使用 popen 启动并在 Python 层持有句柄
      proc = h_src.popen(cmd_send, shell=True)
      bg_processes.append(proc)
      
      # 打印生命周期信息
      print(f"h{u:<4} -> h{v:<4} | {bw:<10.2f} | {pps:<8} | {proc.pid:<8} | .../send_h{u}_to_h{v}.log")
      
      count += 1
      if count % 10 == 0: time.sleep(0.1)

    print(f"{'='*60}")
    print(f"[TM] ✅ All {count} background flows dispatched.")
    print(f"[TM] 💡 Check {log_dir}/ for specific error messages if flows die unexpectedly.\n")
    
    # 可选：返回进程列表，以便外部脚本可以在测试结束后显式 kill 它们
    return bg_processes