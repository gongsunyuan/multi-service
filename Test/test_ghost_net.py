import os
import sys
import time
import numpy as np
import networkx as nx
from mininet.log import setLogLevel, info

# 确保路径正确
sys.path.append(os.getcwd())

from MS.Env.NetworkGenerator import TopologyGenerator
from MS.Env.FlowGenerator import FlowGenerator
from MS.Env.MininetController import get_a_mininet, NetworkMonitor, install_path_rules, clean_flow_rules

# 配置类
class Config:
  MAX_BW = 90.0
  MIN_BW = 7.5
  MIN_DELAY = 1.0
  MAX_DELAY = 200.0
  MAX_NODES_NUM = 14

def analyze_network_state(G_nx):
  """
  分析图中的利用率分布
  """
  utilizations = []
  total_throughput_mbps = 0.0
  
  print("\n" + "-"*60)
  print(f"{'Link':<15} | {'Capacity (Mbps)':<15} | {'Load (Mbps)':<12} | {'Util %':<8} | {'Status'}")
  print("-" * 60)
  
  for u, v, data in G_nx.edges(data=True):
    util = data.get('utilization', 0.0)
    cap = data.get('capacity', 100.0)
    
    # 反推当前负载速率
    current_load = util * cap
    total_throughput_mbps += current_load
    utilizations.append(util)
    
    # 状态标记
    status = ""
    if util > 0.8: status = "\033[91mCONGESTED\033[0m" # 红色
    elif util > 0.4: status = "\033[93mBUSY\033[0m"      # 黄色
    elif util > 0.01: status = "ACTIVE"
    else: status = "IDLE"
    
    # 只打印活跃链路以节省屏幕空间
    if util > 0.01:
      print(f"{u:<2} <--> {v:<2}      | {cap:<15.1f} | {current_load:<12.2f} | {util:<8.2%} | {status}")
      
  return total_throughput_mbps, utilizations

def run_tm_monitor_test():
  setLogLevel('info')
  print("====== [Test] Traffic Matrix Generation & Monitoring ======")
  
  # 1. 初始化
  topo_gen = TopologyGenerator()
  flow_gen = FlowGenerator()
  
  try:
    # 加载 NSFNet (注意: 这里不需要 copy，因为我们要实时更新它)
    G_nx = topo_gen.load_topology("nsfnet.graphml")
  except:
    print("❌ 未找到 nsfnet.graphml")
    return

  # 2. 启动 Mininet
  with get_a_mininet(G_nx) as net:
    print("[System] Mininet started.")
    monitor = NetworkMonitor(net)
    
    # 确保环境干净
    clean_flow_rules(net, cookie=0xB000, mask=0xFFFF)
    
    # 3. 生成流量矩阵 (设定一个中等偏高的负载，观察拥塞)
    TARGET_LOAD = 400.0 # Mbps
    print(f"\n[Gen] 生成流量矩阵 (Target: {TARGET_LOAD} Mbps)...")
    
    # 使用重力模型生成
    tm = flow_gen.generate_traffic_matrix(G_nx.nodes(), total_load_mbps=TARGET_LOAD)
    tm_sum = sum(tm.values())
    print(f"  -> 理论 TM 总流量: {tm_sum:.2f} Mbps")
    
    print("[Debug] Testing basic connectivity...")
 
    # 选一对有流量的节点，比如 node 0 和 node 1
    h0 = net.get('h0')
    h1 = net.get('h1')

    # 1. 下发一条临时测试流表 (双向)
    path = nx.shortest_path(G_nx, 0, 1)
    install_path_rules(net, path, cookie=0x9999) # 确保 install_path_rules 里有双向规则

    # 2. Ping 测试
    result = net.ping([h0, h1])
    print(f"[Debug] Ping h0->h1 drop rate: {result}%")

    if result == 100:
      print("❌ 严重错误：基础网络不通！流表可能未生效，D-ITG 根本发不出去。")
    # 4. 注入 Mininet (Ghost Traffic)
    print("[Inject] 注入背景流并等待稳定...")
    flow_gen.apply_traffic_matrix_to_mininet(
      net, tm, G_nx, install_path_rules, duration=30
    )
    
    print("[Wait] 等待 8 秒让流量爬坡...")
    time.sleep(8)
    
    # 5. 使用 Monitor 感知网络状态
    print("\n[Monitor] 正在同步网络状态 (采样 1.0s)...")
    # 这会读取 Mininet 的真实计数器，并更新到 G_nx 的 'utilization' 属性中
    monitor.sync_state_to_graph(G_nx, duration=1.0)

    print("\n[Debug] Checking OVS Flow Stats on s0...")
    # 查看 s0 上的流表统计，看看 packet_count 是否在增加
    os.system("sudo ovs-ofctl -O OpenFlow13 dump-flows s0")
    # 6. 分析结果
    real_throughput, utils = analyze_network_state(G_nx)
    
    print("-" * 60)
    print(f"📊 统计结果:")
    print(f"  1. 理论注入: {tm_sum:.2f} Mbps")
    # 注意: real_throughput 可能会比 tm_sum 大，因为它是所有链路负载之和。
    # 如果一个流经过 3 跳，它会被计算 3 次。这是符合物理规律的 (Network Load)。
    print(f"  2. 全网链路总负载: {real_throughput:.2f} Mbps (含多跳叠加)")
    print(f"  3. 链路利用率分布: Max={max(utils):.2%}, Avg={np.mean(utils):.2%}")
    
    # 7. 验证重力模型的效果 (是否出现了结构化拥塞?)
    congested_links = len([u for u in utils if u > 0.5])
    if congested_links > 0:
      print(f"✅ 验证成功: 检测到 {congested_links} 条拥塞链路 (重力模型生效)。")
    else:
      print("⚠️ 验证警告: 网络过于空闲，可能需要增加 TARGET_LOAD。")
    
    print("=" * 60)

    
    # 清理
    os.system("sudo killall -9 ITGSend > /dev/null 2>&1")

if __name__ == "__main__":
  if os.getuid() != 0:
    print("❌ 必须使用 sudo 运行")
  else:
    run_tm_monitor_test()