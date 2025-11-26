import re
import time 
import networkx as nx
from time import sleep
from mininet.net import Mininet
import MS.Env.VerbosePrint as vp
from mininet.log import setLogLevel, info
from MS.Env.MininetController import get_a_mininet, install_path_rules

vprint = vp.vprint

def generate_background_traffic(client, server, duration=10, target_bw_mbps=600):
  """
  启动并发 D-ITG 流，等待完成，并自动解码显示结果。
  """
  single_flow_limit = 150.0 
  num_flows = int(target_bw_mbps / single_flow_limit) + 1
  
  # 1460 Bytes payload + headers -> ~1500 Bytes wire size
  # D-ITG -C rate is in packets/second
  # Rate = (Target_Mbps * 1000 * 1000) / (1460 * 8)
  pkt_rate = int((target_bw_mbps * 1000000 / (1460 * 8)) / num_flows)

  vprint(f"[Traffic] Starting {num_flows} flows to target {target_bw_mbps} Mbps...")
  
  # 使用 /tmp/ 目录以保持整洁，并防止权限问题
  log_prefix = "itg_flow"

  # --- 1. 启动所有流 ---
  for i in range(num_flows):
    sig_port = 9000 + i*5 
    data_port = 8000 + i*5
    log_file = f"receiver{i+1}.log"
    
    # 启动 Server (接收端)
    # 关键：使用 -l 指定日志路径
    recv_out = server.cmd(f"ITGRecv -Sp {sig_port} -l {log_file} &")
    # 稍作停顿防止竞争
    time.sleep(0.5)

    vprint(f"[Recv]\n{recv_out}\n")
    # 关键：不需要 -x，因为 Server 已经指定了 -l
    client_cmd = f"ITGSend -a {server.IP()} -Sdp {sig_port} -T UDP -t {duration*1000} -c 1460 -C {pkt_rate} -l sender{i+1}.log&"
    # 绑核优化 (可选)
    # client_cmd = f"taskset -c {i+1} " + client_cmd
    client_out = client.cmd(client_cmd)
    time.sleep(0.2)
    vprint(f"[Send]\n{client_out}\n")
    vprint(f"   -> Flow {i} started (Sig={sig_port}, Data={data_port})")

  # --- 2. 等待传输完成 ---
  vprint(f"[Traffic] Running for {duration} seconds...")
  # 多等 2 秒确保数据包发完
  time.sleep(duration + 2)

  # --- 3. 清理进程 (这是生成完整日志的关键) ---
  # 必须杀掉 ITGRecv，确保它把缓冲区的数据写入文件
  server.cmd("killall -9 ITGRecv")
  client.cmd("killall -9 ITGSend")
  vprint("[Traffic] Processes cleaned up. Decoding logs...")

  # --- 4. 自动解码并统计 ---
  total_bitrate = 0.0
  
  for i in range(num_flows):
    log_file = f"receiver{i+1}.log"
    
    # 在 Server 端运行解码器
    output = server.cmd(f"ITGDec {log_file}")
    vprint(f"[Decoder] \n{output}\n")
    # 解析输出获取码率
    # 匹配行: Average bitrate = 12345.67 Kbit/s
    match = re.search(r"Average bitrate\s+=\s+([\d\.]+)\s+Kbit/s", output)
    
    if match:
      kbps = float(match.group(1))
      mbps = kbps / 1000.0
      total_bitrate += mbps
      vprint(f"   -> Flow {i}: {mbps:.2f} Mbps")
    else:
      vprint(f"   -> Flow {i}: [Error] No data found (Check log file)")

  vprint("="*40)
  vprint(f"🔥 Total Aggregate Throughput: {total_bitrate:.2f} Mbps")
  vprint(f"   Target was: {target_bw_mbps} Mbps")
  vprint("="*40)

  # 清理临时文件
  server.cmd(f"rm -f {log_prefix}_*.log")

def generate_congestion_iperf(client, server, duration=5, target_bw_mbps=600):
  vprint(f"[Back] Generating {target_bw_mbps} Mbps on {client.name}->{server.name} ---")
  
  # 1. 启动 Server (后台)
  # -1: 处理一次连接后退出
  server.cmd("iperf3 -s -p 5201 -1 &") 
  time.sleep(0.5) # 等待 Server 就绪
  
  # 2. 启动 Client (发送)
  # -t: 时间
  # -b: 带宽
  # -u: UDP
  # -J: 输出 JSON 格式 (方便 Python 解析验证)
  cmd = f"iperf3 -c {server.IP()} -p 5201 -u -b {target_bw_mbps}M -t {duration} -J"
  
  vprint(f"[Back] Executing: {cmd}")
  result_json = client.cmd(cmd)
  
  return result_json

