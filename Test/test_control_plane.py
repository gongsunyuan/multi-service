import os
import sys
import time
import torch
import networkx as nx
from mininet.net import Mininet
from mininet.node import OVSKernelSwitch, RemoteController
from mininet.link import TCLink
from mininet.log import setLogLevel

# 确保能导入项目模块
sys.path.append(os.getcwd())

try:
  from MS.Env.MininetController import install_path_rules, clean_flow_rules, sample_path
except ImportError:
  print("❌ 错误: 无法导入 MS.Env.MininetController。请确保在项目根目录下运行。")
  sys.exit(1)

def test_path_extraction():
  print("\n[阶段 1] 测试路径提取逻辑...")
  
  # 构建一个简单的图: 0 -> 1 -> 2 (最优), 0 -> 2 (直连但分低)
  # 边列表: (0,1), (1,2), (0,2)
  edge_index = torch.tensor([
    [0, 1, 0],
    [1, 2, 2]
  ], dtype=torch.long)
  
  # 模拟 Logits: 赋予 (0,1) 和 (1,2) 高分，(0,2) 低分
  # 边 0: 0->1 (Score 5.0)
  # 边 1: 1->2 (Score 5.0)
  # 边 2: 0->2 (Score -1.0)
  edge_logits = torch.tensor([5.0, 5.0, -1.0])
  
  path, is_success = sample_path(edge_logits, edge_index, 0, 2, greedy=True)
  
  expected = [0, 1, 2]
  if path == expected:
    print(f"[info] 路径提取成功: {path}")
  else:
    print(f"[error] 路径提取失败: 预期 {expected}, 实际 {path}")

def test_flow_control():
  print("\n[阶段 2] 测试流表下发与清理 (Mininet)...")
  setLogLevel('warning') # 减少刷屏
  
  # 1. 构建拓扑: h1 -- s1 -- s2 -- h2
  print("   [环境] 启动拓扑: h1 -- s1 -- s2 -- h2")
  net = Mininet(switch=OVSKernelSwitch, link=TCLink, controller=None)
  
  h1 = net.addHost('h1', ip='10.0.0.1')
  h2 = net.addHost('h2', ip='10.0.0.2')
  s1 = net.addSwitch('s1', protocols='OpenFlow13')
  s2 = net.addSwitch('s2', protocols='OpenFlow13')
  
  net.addLink(h1, s1)
  net.addLink(s1, s2)
  net.addLink(s2, h2)
    
  try:
    net.start()
    time.sleep(1)
    
    # --- 测试 A: 初始状态 (应该不通) ---
    print("   [Step A] 测试初始连通性 (预期: 不通)")
    # 使用 ping -w 1 设置 1秒超时，防止卡住
    p1 = net.ping([h1, h2], timeout=1)
    if p1 > 0:
      print("   [info ] 初始状态正确 (Ping 失败)")
    else:
      print("   [error] 警告: 初始状态竟然通了？可能有残留流表。")
      clean_flow_rules(net) # 尝试清理一次

    # --- 测试 B: 下发流表 ---
    # 路径: s1(ID=1) -> s2(ID=2)
    # 注意: MininetController 里假设 ID 是数字。
    # 你的代码通常假设 node_id 对应 s{node_id}。
    # 这里我们需要确认你的 ID 映射逻辑。
    # 假设: 路径节点列表是 [1, 2] 代表 s1 -> s2
    print("   [Step B] 下发路径规则: s1 -> s2")
    path_nodes = [1, 2] 
    
    # 调用待测函数
    install_path_rules(net, path_nodes, cookie=0x9999)
    
    # 等待 OVS 处理
    time.sleep(1)
    
    # 验证连通性
    print("   [验证] 再次 Ping (预期: 通)")
    # pingFull 返回 (sent, received) 列表
    loss = net.ping([h1, h2], timeout=1)
    if loss == 0:
      print("流表下发成功！网络已连通。")
    else:
      print(f"[error] 流表下发失败！Ping 丢包率: {loss}%")
      # 打印流表帮助调试
      print("   >>> s1 流表:")
      os.system("ovs-ofctl -O OpenFlow13 dump-flows s1")
      print("   >>> s2 流表:")
      os.system("ovs-ofctl -O OpenFlow13 dump-flows s2")

    # --- 测试 C: 清理流表 ---
    print("   [Step C] 清理流表 (Cookie=0x9999)")
    clean_flow_rules(net, cookie=0x9999)
    time.sleep(1)
    
    print("   [验证] 清理后 Ping (预期: 不通)")
    loss_clean = net.ping([h1, h2], timeout=1)
    if loss_clean > 0:
      print("[info] 流表清理成功！网络再次断开。")
    else:
      print("[error] 流表清理失败！网络依然连通。")
      os.system("ovs-ofctl -O OpenFlow13 dump-flows s1")

  except Exception as e:
    print(f"[error] 测试异常: {e}")
    import traceback
    traceback.print_exc()
  finally:
    print("   [环境] 清理 Mininet...")
    net.stop()
    os.system('sudo mn -c > /dev/null 2>&1')

if __name__ == '__main__':
    if os.getuid() != 0:
      print("请使用 sudo 运行！")
    else:
      test_path_extraction()
      test_flow_control()