import os
import sys
import time
import networkx as nx
import re
import numpy as np
from mininet.log import setLogLevel, info
from mininet.cli import CLI
# 确保路径正确
sys.path.append(os.getcwd())

from MS.Env.NetworkGenerator import TopologyGenerator
from MS.Env.FlowGenerator import FlowGenerator
from MS.Env.MininetController import get_a_mininet, install_path_rules, clean_flow_rules, NetworkMonitor

class Config:
  MAX_BW = 90.0
  MIN_BW = 7.5
  MIN_DELAY = 1.0
  MAX_DELAY = 200.0
  MAX_NODES_NUM = 14

def verify_offload_status(net):
  """
  遍历网络中的所有交换机接口，打印 TCP Segmentation Offload 状态。
  用于验证 '幽灵带宽' 问题是否已解决。
  """
  from MS.Env.VerbosePrint import vprint # 假设你有这个打印函数，或者直接用 print
  
  print("\n" + "="*40)
  print("🔍 [DEBUG] Verifying TSO/GSO Status on Switches")
  print("="*40)
  
  # 抽样检查：如果接口太多，可以只检查前几个交换机
  for sw in net.switches:
    for intf in sw.intfList():
      if intf.name == 'lo': continue # 跳过回环接口
      
      # 在该节点(Switch)内部执行 shell 命令
      # grep 过滤只显示 tcp-segmentation 相关的行
      cmd = f"ethtool -k {intf.name} | grep 'tcp-segmentation-offload'"
      result = sw.cmd(cmd).strip()
      
      # 解析结果：如果是 'on'，用红色或显眼标记；如果是 'off'，表示成功
      status_mark = "✅ OFF (Safe)" if ": off" in result else "❌ ON (Ghost Bandwidth Risk!)"
      
      print(f"[{sw.name}][{intf.name}]: {status_mark}")
      print(f"    Raw: {result}") # 如果需要详细信息可取消注释

  print("="*40 + "\n")

def print_network_status(G):
  """打印全网高负载链路"""
  print("-" * 60)
  print(f"📊 [Global] High Load Links (>10%):")
  print(f"  {'Link':<12} | {'Cap (Mbps)':<10} | {'Util %':<8} | {'Status'}")
  print("-" * 60)
  
  count = 0
  for u, v, data in G.edges(data=True):
    util = data.get('utilization', 0.0)
    cap = data.get('capacity', 100.0)
    if util > 0.1: # 只显示活跃链路
      status = "\033[91mFULL\033[0m" if util > 0.9 else "BUSY"
      print(f"  {u:<2} <-> {v:<2}    | {cap:<10.1f} | {util:<8.2%} | {status}")
      count += 1
  if count == 0:
    print("  (No congested links found)")
  print("-" * 60)

def print_path_status(G, path):
  """打印特定路径的详细状态"""
  print("-" * 60)
  print(f"🛣️ [Target Path] Status: {path}")
  print(f"  {'Hop':<12} | {'Cap (Mbps)':<10} | {'Load':<8} |{'Util %':<8} | {'Delay':<8} | {'Status'}")
  print("-" * 60)
  
  total_delay = 0.0
  is_bad = False
  
  for u, v in zip(path[:-1], path[1:]):
    data = G[u][v]
    util = data.get('utilization', 0.0)
    cap = data.get('capacity', 100.0)
    
    # 估算延迟 (Prop + Queue)
    prop_delay = data.get('delay', 5.0) # 注意: 这里假设 NetworkGenerator 还没把 delay 变成 total_delay
    # 如果 NetworkGenerator 里的 delay 已经是 total_delay，直接用即可
    # 这里为了保险，我们用 MM1 手算一下展示给用户看
    if util > 0.99: queue = 100.0 
    else: queue = 10.0 * util / (1.0 - util)
    
    delay = prop_delay + queue
    total_delay += delay
    
    status = "OK"
    if util > 0.8: 
      status = "\033[91mBAD\033[0m"
      is_bad = True
    load = data.get('measured_speed', 0.0) # 使用真实值
    print(f"  {u:<2} -> {v:<2}     | {cap:<10.1f} | {load:<8.2f} | {util:<8.2%} | {delay:<6.1f}ms | {status}")

  print(f"  [Summary] Path Total Estimated Delay: {total_delay:.2f} ms")
  if is_bad:
    print("  ⚠️ 警告: 此路径包含拥塞路段，预计会有丢包！")
  else:
    print("  ✅ 路径通畅。")
  print("-" * 60)

def run_survival_test():
  setLogLevel('info')
  print("====== [Test] Verifying Target Flow Survival ======")
  
  topo_gen = TopologyGenerator()
  flow_gen = FlowGenerator()
  
  try:
    # 加载拓扑 (不使用 copy，我们需要实时更新 G_nx 的属性)
    G_nx = topo_gen.load_topology("nsfnet.graphml")
  except:
    print("❌ 未找到 nsfnet.graphml")
    return

  with get_a_mininet(G_nx) as net:
    verify_offload_status(net)
    print("[System] Mininet started.")
    monitor = NetworkMonitor(net) # 1. 初始化 Monitor
    
    clean_flow_rules(net, cookie=0xB000, mask=0xFFFF)
    clean_flow_rules(net, cookie=0xA001, mask=0xFFFF)
    
    # --- Step 1: 制造拥塞 ---
    print("\n[Step 1] 注入背景流 (Target: 600 Mbps)...")
    tm = flow_gen.generate_traffic_matrix(G_nx.nodes(), total_load_mbps=600.0)
    
    proccesses = flow_gen.apply_traffic_matrix_to_mininet(
      net, tm, G_nx, install_path_rules, duration=1000
    )
    
    print(f"[Ghost] There is {len(proccesses)} num of proc")
    print("[Wait] 等待 5 秒让背景流稳定...")
    time.sleep(5)
    
    # --- [NEW] Step 1.5: 感知与输出状态 ---
    print("\n[Monitor] 同步网络状态...")
    monitor.sync_state_to_graph(G_nx) # 读取真实物理状态写入 G_nx
    
    # 1. 打印全网状态 (看看是不是真的堵了)
    print_network_status(G_nx)
    # CLI(net)
    # --- Step 2: 发送目标流 ---
    s, d = 0, 13
    h_src, h_dst = net.get(f'h{s}'), net.get(f'h{d}')
    print(f"\n[Step 2] 准备发送目标流: h{s} -> h{d}")
    
    # 2.1 规划路径 (选择最短路)
    try:
      path = nx.shortest_path(G_nx, s, d, weight='delay')
    except:
      print("  -> 无路可走!")
      return
      
    # 2. [NEW] 打印目标路径状态 (看看这车是不是往火坑里开)
    print_path_status(G_nx, path)

    # 2.2 下发规则 & 启动接收端
    install_path_rules(net, path, cookie=0xA001, do_ping=False)
    
    recv_log = "/tmp/target_survival.log"
    if os.path.exists(recv_log): os.remove(recv_log)
    
    # 确保 ITGRecv 重启
    # h_dst.cmd("killall -9 ITGRecv")
    h_dst.cmd("ITGRecv > /dev/null 2>&1 &")
    time.sleep(0.5)

    print(f"[Ghost] There is {len(proccesses)} num of proc")
    time.sleep(20)
    print(f"[Ghost] There is {len(proccesses)} num of proc")
    monitor.sync_state_to_graph(G_nx) # 读取真实物理状态写入 G_nx
    # 1. 打印全网状态 (看看是不是真的堵了)
    print_network_status(G_nx)

    os.system("sudo killall -9 ITGSend ITGRecv > /dev/null 2>&1")

if __name__ == "__main__":
  if os.getuid() != 0:
    print("❌ 必须使用 sudo 运行")
  else:
    run_survival_test()