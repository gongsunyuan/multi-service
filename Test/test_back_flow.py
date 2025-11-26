import re
import time 
import networkx as nx
import MS.Env.VerbosePrint as vp
from mininet.cli import CLI
from mininet.log import setLogLevel, info
from MS.Env.TrafficManager import generate_congestion_iperf
from MS.Env.MininetController import get_a_mininet, install_path_rules

vprint = vp.vprint

def run_test():
  setLogLevel('info')
  
  # 1. 创建一个最简单的拓扑: h0 -- s0 -- h1
  vprint("[Test] Setting up minimal topology...")
  g = nx.Graph()
  g.add_nodes_from([0, 1])
  g.add_edges_from([(0, 1),])
  for u, v in g.edges():
    vprint(f"[Test] link ({u}, {v}) installed")
  try:
    with get_a_mininet(g) as net:
      install_path_rules(net, [0, 1])
      
      # 预热
      net.pingAll()
      h0, h1 = net.get('h0', 'h1')
      g[0][1]['delay'] = 2
      g[0][1]['bandwidth'] = 600
      # 3. 运行流量测试
      vprint("[Test] Running Traffic Generation...")
      import json
      output = generate_congestion_iperf(h0, h1, duration=10, target_bw_mbps=900)
      
      # 4. 验证结果
      try:
        mbps = 0.0
        lost_percent = 0.0
        # 尝试查找 JSON 开始和结束的大括号，防止混入其他日志噪音
        json_start = output.find('{')
        json_end = output.rfind('}') + 1
        
        if json_start != -1 and json_end != -1:
          clean_json = output[json_start:json_end]
          data = json.loads(clean_json)
          
          # 获取 Summary 部分
          # iperf3 的 JSON 结构在 UDP 和 TCP 下略有不同
          end_sum = data.get('end', {}).get('sum', {})
          
          # 如果是 UDP，有时候在 sum_received 里，有时候在 sum 里
          # 这是一个防御性写法
          if 'bits_per_second' not in end_sum and 'sum_received' in data['end']:
              end_sum = data['end']['sum_received']

          # 提取数据
          bps = end_sum.get('bits_per_second', 0.0)
          mbps = bps / 1e6
          lost_percent = end_sum.get('lost_percent', 0.0)
          
          vprint( "="*40)
          vprint(f"[Test] Perf3 Result (Parsed):")
          vprint(f"[Test] Throughput: {mbps:.2f} Mbps")
          vprint(f"[Test] Loss Rate:  {lost_percent:.2f} %")
          vprint("="*40)
        else:
          vprint(f"[Error] Could not find valid JSON in output.")

        CLI(net)
      except Exception as e:
        vprint(f"[Parse Error] Failed to parse iPerf3 JSON: {e}")
        vprint(f"Raw Output: {output}") # 调试时可以打开
  finally:
    pass

if __name__ == '__main__':
  run_test()