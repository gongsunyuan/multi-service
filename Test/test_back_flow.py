import re
import time 
import networkx as nx
import MS.Env.VerbosePrint as vp
from mininet.cli import CLI
from mininet.log import setLogLevel, info
# from MS.Env.FlowGenerator import generate_background_flow
from MS.Env.MininetController import get_a_mininet, install_path_rules

vprint = vp.vprint

def run_test():
  setLogLevel('info')
  
  # 1. 创建一个最简单的拓扑: h0 -- s0 -- h1
  vprint("[Test] Setting up minimal topology...")
  g = nx.Graph()
  g.add_nodes_from([0, 1, 2])
  g.add_edges_from([(0, 1),(2, 1)])
  g[0][1]['delay'] = 2
  g[1][2]['delay'] = 2
  g[0][1]['bandwidth'] = 8
  g[1][2]['bandwidth'] = 8
  for u, v, data in g.edges(data=True):
    vprint(f"[Test] link ({u}, {v}) installed")
    vprint(f"[Link] bw = {data.get('bandwidth')}, delay = {data.get('delay')}")

  try:
    with get_a_mininet(g) as net:
      install_path_rules(net, [0, 1])
      install_path_rules(net, [2, 1])
      
      # 预热
      # net.pingAll()
      h0, h1, h2 = net.get('h0', 'h1', 'h2')
      # 3. 运行流量测试
      vprint("[Test] Running Traffic Generation...")

      CLI(net)
  finally:
    pass

if __name__ == '__main__':
  run_test()