import os
import sys
import time
import networkx as nx
import re
import numpy as np
from mininet.log import setLogLevel, info

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
    print("[System] Mininet started.")
    monitor = NetworkMonitor(net) # 1. 初始化 Monitor
    
    clean_flow_rules(net, cookie=0xB000, mask=0xFFFF)
    clean_flow_rules(net, cookie=0xA001, mask=0xFFFF)
    
    # --- Step 1: 制造拥塞 ---
    print("\n[Step 1] 注入背景流 (Target: 600 Mbps)...")
    tm = flow_gen.generate_traffic_matrix(G_nx.nodes(), total_load_mbps=600.0)
    
    flow_gen.apply_traffic_matrix_to_mininet(
      net, tm, G_nx, install_path_rules, duration=30
    )
    
    print("[Wait] 等待 5 秒让背景流稳定...")
    time.sleep(5)
    
    # --- [NEW] Step 1.5: 感知与输出状态 ---
    print("\n[Monitor] 同步网络状态...")
    monitor.sync_state_to_graph(G_nx) # 读取真实物理状态写入 G_nx
    
    # 1. 打印全网状态 (看看是不是真的堵了)
    print_network_status(G_nx)
    
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
    h_dst.cmd("killall -9 ITGRecv")
    h_dst.cmd("ITGRecv > /dev/null 2>&1 &")
    time.sleep(0.5)

    # 2.3 启动发送端
    cmd = (f"ITGSend -a {h_dst.IP()} "
           f"-T TCP "
           f"-C 500 -c 1600 -t 15000 "
           f"-x {recv_log} "
           f"-b 32") # 护身符
           
    print(f"  -> 发送命令: {cmd}")
    h_src.cmd(cmd)
    
    print("  -> 等待传输完成...")
    time.sleep(4)
    
    # --- Step 3: 验证结果 ---
    print("\n[Step 3] 验证 D-ITG 接收结果:")
    
    if not os.path.exists(recv_log):
      print(f"❌ 失败：日志未生成。")
    else:
      out = h_src.cmd(f"ITGDec {recv_log}")
      pkts_match = re.search(r"Total packets\s+=\s+(\d+)", out)
      loss_match = re.search(r"Packets dropped\s+=\s+\d+\s+\(([\d\.]+)\s+%\)", out)
      delay_match = re.search(r"Average delay\s+=\s+([\d\.]+)\s+s", out)
      print(out)
      if pkts_match:
        total = int(pkts_match.group(1))
        loss = float(loss_match.group(1)) if loss_match else 0.0
        delay = float(delay_match.group(1)) * 1000 if delay_match else 0.0
        
        print(f"  - 收到包数: {total}")
        print(f"  - 实际丢包: {loss}%")
        print(f"  - 实际延迟: {delay:.2f} ms")
        
        if total > 0:
            print("\n🎉 结论: 目标流存活！双流隔离有效。")
            if loss > 0 or delay > 20:
                print("💪 效果: 目标流受到了背景流的真实干扰 (符合预期)。")
            else:
                print("🤔 效果: 目标流未受干扰 (路径太顺了？建议加大背景负载)。")
      else:
        print(f"❌ 解析失败。")

    os.system("sudo killall -9 ITGSend ITGRecv > /dev/null 2>&1")

if __name__ == "__main__":
  if os.getuid() != 0:
    print("❌ 必须使用 sudo 运行")
  else:
    run_survival_test()