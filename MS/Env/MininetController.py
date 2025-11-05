import os
import subprocess
from mininet.net import Mininet
from mininet.topo import topo

def run_traffic_capture(S_host, D_host, flow_profile, N_PACKETS=50):
    
  temp_pcap_file = f"/tmp/fingerprint_{os.getpid()}.pcap"
  mode = flow_profile['iperf_mode']
  rate = flow_profile['target_rate']
  
  # --- 根据模式启动对应的服务器 ---
  iperf_server_cmd = f'iperf -s -p 5001'
  if mode == 'udp':
    iperf_server_cmd += ' -u' # 如果是 UDP，添加 -u
  
  D_host.cmd(f'{iperf_server_cmd} &')
  # -----------------------------------

  # 启动 tcpdump
  capture_command = f'tcpdump -i {S_host.intfNames()[0]} -c {N_PACKETS} -w {temp_pcap_file}'
  S_host.cmd(f'{capture_command} &')
  
  time.sleep(0.5) 
  
  # 启动 iperf client (这部分是正确的)
  client_cmd = f'iperf -c {D_host.IP()} -p 5001 -{mode[0]} -b {rate} -t 5'
  S_host.cmd(client_cmd) 
  
  # 清理 iperf server 进程
  D_host.cmd('kill %iperf')
  
  return temp_pcap_file

def get_flow_fingerprint(S_host, D_host, flow_profile):
    
  # 运行捕获
  pcap_path = run_traffic_capture(S_host, D_host, flow_profile)
  
  if not os.path.exists(pcap_path):
    print(f"⚠️ 捕获文件未找到: {pcap_path}")
    return None
      
  # 使用 Scapy 或 dpkt 读取文件
  # 这里的 `extract_flow_fingerprint` 就是我们之前讨论的 Scapy 函数
  fingerprint_matrix = extract_flow_fingerprint(pcap_path)
  
  # 清理临时文件 (重要!)
  os.remove(pcap_path)
  
  return fingerprint_matrix


def measure_path_qos(S_host, D_host, path_route, flow_profile):
  # 1. 配置 OpenFlow 规则 (将流量强制导向 path_route)
  # todo()
  # ... (这一步依赖于你的 SDN 控制器或 OVS 流表操作) ...

  # 2. 启动 iperf server
  D_host.cmd(f'iperf -s -p 5002 -i 1 &') # -i 1: 每秒输出一次报告
  time.sleep(1) 
  
  # 3. 运行 iperf client 并捕获输出
  mode = flow_profile['iperf_mode']
  rate = flow_profile['target_rate']
  
  # 注意：使用 Mininet.cmd(..., verbose=True) 或直接在 Mininet 控制台中运行
  # 这里我们使用 Python 的 subprocess 或 Mininet 的 cmd 捕获输出
  
  # 运行 client 10秒，并将结果保存在 result 变量中
  result = S_host.cmd(f'iperf -c {D_host.IP()} -p 5002 -{mode[0]} -b {rate} -t 10')
  
  # 4. 清理 server 进程
  D_host.cmd('kill %iperf')
  
  # 5. 解析结果 (从 result 字符串中提取数值)
  # 这需要编写一个解析器来处理 iperf 输出格式
  # 假设解析器能返回一个字典  
  qos_metrics = parse_iperf_output(result) 
  
  # 6. 计算 Reward
  reward = calculate_qoe_reward(qos_metrics, flow_profile)
  
  return reward