import os
import sys
import time
import networkx as nx
import logging
from mininet.net import Mininet
from mininet.node import OVSKernelSwitch, RemoteController
from mininet.link import TCLink
from mininet.log import setLogLevel

# 确保可以导入 MS 模块
sys.path.append(os.getcwd())
from src.env.MininetController import NetworkMonitor, get_a_mininet
from src.env.NetworkGenerator import TopologyGenerator
from src.utils.MyParaser import TopoParaser

class Test_config:
  M_BA = 2
  MIN_BW = 5.0
  MAX_BW = 20.0
  MIN_LOSS = 0.0
  MAX_LOSS = 3.0
  MIN_DELAY = 1.0
  MAX_DELAY = 20.0
  MIN_NODES_NUM = 6
  MAX_NODES_NUM = 10

def test_monitor_live():
  print("[ms] 开始 NetworkMonitor 功能测试...")
  # setLogLevel('info')
    
  topo_parser=TopoParaser()
  args = topo_parser.parse_args()
  
  flow_gen = TopologyGenerator(config=Test_config())
  G = flow_gen.generate_topology()

  with get_a_mininet(G, remote_port=args.remote_port) as net: 
    # CLI(net)
    try:
      h1 = net.get('h1')
      h2 = net.get('h2')
      # 简单的 Ping 测试
      loss = net.ping([h1, h2])
      if loss > 0:
        print("[ms] Warning: Ping 丢包，可能未完全连通")
      
      # 初始化监控器
      monitor = NetworkMonitor(net)
      
      # --- 阶段 A: 空闲状态测试 ---
      print("\n[3] 测试空闲状态 (Idle)...")
      monitor.sync_state_to_graph(G)
      
      u, v = 0, 1
      util_idle = G[u][v]['utilization']
      buf_idle = G.nodes[u]['buffer_occupancy']
      print(f"   -> Idle Util: {util_idle:.2%} (预期接近 0%)")
      print(f"   -> Idle Buffer: {buf_idle:.2%} (预期 0%)")
      
      # --- 阶段 B: 拥塞状态测试 ---
      print("\n[4] 启动 iperf 制造拥塞 (UDP 20M -> 10M 链路)...")
      # 从 h1 发送 20M 的 UDP 流量给 h2，持续 10 秒
      # 这里的 bandwidth 是 10M，发 20M 肯定会堵死
      server_cmd = "ITGRecv &"
      h2.cmd(server_cmd)
      
      client_cmd = f"ITGSend -a {h2.IP()} -t 100000 -T UDP -c 1460 -C 2000 -x flow.log"
      h1.cmd(client_cmd)
      
      print("   -> 流量发送中，等待 2 秒让队列堆积...")
      time.sleep(2)
      
      # 再次同步状态
      print("[5] 读取拥塞状态...")
      monitor.sync_state_to_graph(G)
      
      util_busy = G[u][v]['utilization']
      buf_busy = G.nodes[u]['buffer_occupancy'] # s1 的 buffer 应该满了
      
      print(f"   -> Busy Util: {util_busy:.2%} (预期接近 100%)")
      print(f"   -> Busy Buffer: {buf_busy:.2%} (预期 > 0%)")
      
      # --- 验证结果 ---
      print("\n" + "="*30)
      if util_busy > 0.5:
        print("[ms] 利用率监控: 成功 (检测到高负载)")
      else:
        print(f"[ms] 利用率监控: 失败 (读数为 {util_busy:.2f}, 可能没读到 tx_bytes)")
          
      if buf_busy > 0.0:
        print("[ms] 缓冲区监控: 成功 (检测到排队)")
      else:
        print(f"❌ 缓冲区监控: 失败 (读数为 {buf_busy}, 可能 tc 命令解析失败)")
      print("="*30 + "\n")

    except Exception as e:
      print(f"[error] 测试出错: {e}")
      import traceback
      traceback.print_exc()

if __name__ == '__main__':
    if os.getuid() != 0:
      print("请使用 sudo 运行此脚本！")
    else:
      test_monitor_live()