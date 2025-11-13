import os
import sys
import networkx as nx
from mininet.net import Mininet
from mininet.topo import Topo
from mininet.node import OVSKernelSwitch, Controller
from mininet.link import TCLink
from mininet.log import setLogLevel

# --- 导入环境文件 ---
from MS.Env.NetworkGenerator import TopologyGenerator
from MS.Env.FlowGenerator import FlowGenerator, FlowType
from MS.Env.MininetController import get_flow_fingerprint, measure_path_qos

# -----------------------------------------------
# 创建一个自定义拓扑，将 NetworkX Graph 转换为 Mininet Topo
# -----------------------------------------------
class GraphTopo(Topo):
  """
  一个 Mininet 拓扑，它从 NetworkGenerator 的 nx.Graph 构建。
  我们假设 G 中的节点是 *交换机*。
  我们为 G 中的每个节点（交换机）连接一个 *主机*。
  """
  def __init__(self, g: nx.Graph, **opts):
    Topo.__init__(self, **opts)
    
    # self.g = g
    
    # 1. 创建交换机 (基于 G 的节点)
    for node_id in g.nodes():
      self.addSwitch(f's{node_id}')
      
      # 2. 为每个交换机添加一个主机
      self.addHost(f'h{node_id}')
      
      # 3. 添加主机到交换机的连接 (大带宽, 低延迟)
      self.addLink(f'h{node_id}', f's{node_id}', bw=1000, delay='0.1ms')

    # 4. 创建交换机之间的连接 (基于 G 的边)
    for u, v, data in g.edges(data=True):
      bw = data.get('bandwidth', 1000)
      delay = f"{data.get('delay', 1)}ms"
      
      self.addLink(f's{u}', f's{v}', bw=bw, delay=delay)


def run_simulation():
  """主运行函数"""
  
  # --- 步骤 1: 生成拓扑 [NetworkGenerator] ---
  print("--- 步骤 1: 生成拓扑 [NetworkGenerator] ---")
  topo_gen = TopologyGenerator(num_nodes_range=(5, 8)) # 5-8个节点
  g = topo_gen.generate_topology()
  print(f"拓扑生成完毕: {len(g.nodes())} 个节点 (交换机), {len(g.edges())} 条边。")

  # --- 步骤 2: 启动 Mininet ---
  print("\n--- 步骤 2: 启动 Mininet ---")
  
  mn_topo = GraphTopo(g)
  net = Mininet(
    topo=mn_topo,
    switch=OVSKernelSwitch,  # 必须使用 OVS 交换机
    link=TCLink,             # 必须使用 TCLink 以支持 QoS
    controller=None          # 不使用外部 SDN 控制器
  )
  
  net.start()
  print("Mininet 启动。")
  # (可选) net.pingAll()

  # --- 步骤 3: 生成流配置 [FlowGenerator] ---
  print("\n--- 步骤 3: 生成流配置 [FlowGenerator] ---")
  flow_gen = FlowGenerator()
  flow_type, flow_profile = flow_gen.get_random_flow()
  print(f"已选择流类型: {flow_type.name}")

  # --- 步骤 4.A: 测试流指纹生成 (用于 阶段 1.A) ---
  print("\n--- 步骤 4.A: 测试流指纹生成 [MininetController] ---")
  
  # 4.A.1: 选择源/目的交换机 ID
  s_id, d_id = topo_gen.select_source_destination()
  # 4.A.2: 获取 Mininet 中的 *主机* 对象
  S_host = net.get(f'h{s_id}')
  D_host = net.get(f'h{d_id}')
  
  print(f"源主机: h{s_id} ({S_host.IP()})")
  print(f"目的主机: h{d_id} ({D_host.IP()})")

  # 4.A.3: 调用 MininetController!
  fingerprint_matrix = get_flow_fingerprint(S_host, D_host, flow_profile)
  
  if fingerprint_matrix is not None:
    print("\n--- 指纹生成成功! ---")
    print(f"已为 {flow_type.name} 流获取了指纹。")
    print(f"指纹矩阵形状: {fingerprint_matrix.shape}")
    # print(fingerprint_matrix)
  else:
    print("\n--- 指纹生成失败! ---")

  # --- 步骤 4.B: 测试 QoS 测量 (用于 阶段 2) ---
  print("\n--- 步骤 4.B: 测试 QoS 测量 (RL step) ---")
  
  # 这是一个“虚拟”的路径，在你的 RL 中，它应该来自 GNN 的输出
  # 这里我们只测试默认路径（即 s_id 和 d_id 之间的最短路径）
  dummy_path = [s_id, d_id] 
  print(f"正在测试路径 {dummy_path} 的 QoS...")
  
  # 4.B.1: 调用 MininetController!
  reward = measure_path_qos(S_host, D_host, dummy_path, flow_profile)
  
  print(f"\n--- QoS 测量成功! ---")
  print(f"路径 {dummy_path} 在承载 {flow_type.name} 流时的奖励为: {reward}")
  
  # --- 步骤 5: 清理 ---
  print("\n--- 步骤 5: 停止 Mininet ---")
  net.stop()


if __name__ == '__main__':
    # 设置 Mininet 的日志级别
    setLogLevel('info')
    
    # 检查是否以 root 身份运行
    if os.getuid() != 0:
      print("错误：此脚本必须以 root 权限 (sudo) 运行。")
      print("请使用: sudo python tmp_runner.py")
    else:
      run_simulation()