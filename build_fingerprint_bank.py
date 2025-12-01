import torch
import os
import sys
from tqdm import tqdm

# 引入你的环境模块
from MS.Env.MininetController import get_a_mininet, get_a_fingerprint
from MS.Env.FlowGenerator import FlowType, FLOW_PROFILES
from MS.Env.NetworkGenerator import TopologyGenerator
from mininet.node import OVSKernelSwitch, RemoteController
from mininet.link import TCLink
import networkx as nx

def build_bank():
  print("🏦 正在构建真实流量指纹库 (Fingerprint Bank)...")
  
  # 1. 创建一个最简单的点对点拓扑 (h1 -- s1 -- h2) 用于发包
  # 不需要复杂的 NSFNet，只要能通就行
  g = nx.Graph()
  g.add_edge(0, 1, bandwidth=100, delay=1, loss=0)
  
  # 存储容器
  bank = {
    'voip': [],
    'gaming': [],
    'streaming': []
  }
  
  SAMPLES_PER_CLASS = 100 # 每类采集 100 个样本 (足够覆盖方差)
  
  # 2. 启动 Mininet
  # 注意：这里利用 get_a_mininet 上下文
  # 我们临时 mock 一下 NetworkGenerator 的接口或者手动起 Mininet
  # 为了方便，直接手动起一个最简 Mininet
  from mininet.net import Mininet
  net = Mininet(switch=OVSKernelSwitch, link=TCLink)
  h1 = net.addHost('h1')
  h2 = net.addHost('h2')
  s1 = net.addSwitch('s1')
  net.addLink(h1, s1, cls=TCLink, bw=100, delay='1ms')
  net.addLink(s1, h2, cls=TCLink, bw=100, delay='1ms')
  
  try:
    net.start()
    # 下发流表确保连通
    s1.cmd("ovs-ofctl add-flow s1 actions=normal") 
    net.pingAll()
    
    server = h2
    client = h1
    
    for f_type in [FlowType.VOIP, FlowType.GAMING, FlowType.STREAMING]:
      print(f"   正在采集 {f_type.name} ...")
      
      for _ in tqdm(range(SAMPLES_PER_CLASS)):
        # 调用你核心的抓包函数
        # 注意：n_packets_to_capture 必须和你 Config.N_PACKETS 一致 (30)
        fingerprint = get_a_fingerprint(
          server=server, 
          client=client, 
          flow_type=f_type, 
          n_packets_to_capture=30
        ).squeeze(0)
        
        
        # 存入列表
        bank[f_type.name.lower()].append(fingerprint)
              
  finally:
    net.stop()
    os.system("sudo mn -c > /dev/null 2>&1")
      
  # 3. 保存为 PyTorch 文件
  save_path = "./dataset/fingerprint_bank.pt"
  os.makedirs("./dataset", exist_ok=True)
  torch.save(bank, save_path)
  print(f"✅ 指纹库已保存至: {save_path}")

if __name__ == '__main__':
  if os.geteuid() != 0:
    print("❌ 需要 sudo 权限来运行 Mininet")
  else:
    build_bank()