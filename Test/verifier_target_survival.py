import os
import sys
import time
import networkx as nx
import re
import signal
import subprocess
import time
import numpy as np
import subprocess
from mininet.log import setLogLevel, info
from mininet.cli import CLI
# 确保路径正确
sys.path.append(os.getcwd())

from MS.Env.FlowGenerator import FlowGenerator
from MS.Env.NetworkGenerator import TopologyGenerator
from MS.Env.MininetController import get_a_mininet, install_path_rules, clean_flow_rules, NetworkMonitor

class Config:
  MAX_BW = 90.0
  MIN_BW = 7.5
  MIN_DELAY = 1.0
  MAX_DELAY = 200.0
  MAX_NODES_NUM = 14

def run_itg_safe(h_src, cmd, timeout_sec=20):
  """
  安全运行 ITGSend：如果超时，发送 SIGINT 让其写入日志后退出。
  """
  print(f"🚀 [Sender] 启动命令 (超时限制 {timeout_sec}s)...")
  
  # 注意：这里我们使用 h_src.popen 的底层 subprocess 对象
  # start_new_session=True 是为了能够向进程组发送信号 (可选，视 Mininet 实现而定)
  proc = h_src.popen(cmd.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
  
  try:
    # 1. 正常等待 (Wait)
    # 如果 D-ITG 在规定时间内跑完，这里会正常返回
    stdout, stderr = proc.communicate(timeout=timeout_sec)
      
  except subprocess.TimeoutExpired:
    # 2. 发生超时 (Hang) -> 触发熔断
    print(f"⏰ [Timeout] TCP 发送卡死 (> {timeout_sec}s)！正在强制结算...")
    
    # 【关键步骤】发送 SIGINT (Ctrl+C)
    # 这告诉 D-ITG: "别发了，赶紧写日志收工！"
    proc.send_signal(signal.SIGINT)
      
    try:
      # 给它 1-2 秒时间处理后事 (写文件)
      stdout, stderr = proc.communicate(timeout=2)
      print("✅ [Safe Kill] 进程已优雅退出，日志应已保存。")
    except subprocess.TimeoutExpired:
      # 3. 敬酒不吃吃罚酒 -> 强杀
      print("💀 [Force Kill] 进程无响应，执行 SIGKILL。")
      proc.kill()
      stdout, stderr = proc.communicate()
          
  # 打印输出以便调试
  if stderr:
    print(f"❌ Stderr: {stderr.decode('utf-8')}")
      
  return stdout, stderr

def verify_offload_status(net):
  """
  遍历网络中的所有交换机接口，打印 TCP Segmentation Offload 状态。
  用于验证 '幽灵带宽' 问题是否已解决。
  """
  from MS.Env.VerbosePrint import vprint # 假设你有这个打印函数，或者直接用 print
  
  print("\n" + "="*40)
  print("🔍 [DEBUG] Verifying TSO/GSO Status on Switches")
  print("="*40)
  
  # 抽样检查：如果接口太多，可以只检查前几个交换机
  for sw in net.switches:
    for intf in sw.intfList():
      if intf.name == 'lo': continue # 跳过回环接口
      
      # 在该节点(Switch)内部执行 shell 命令
      # grep 过滤只显示 tcp-segmentation 相关的行
      cmd = f"ethtool -k {intf.name} | grep 'tcp-segmentation-offload'"
      result = sw.cmd(cmd).strip()
      
      # 解析结果：如果是 'on'，用红色或显眼标记；如果是 'off'，表示成功
      status_mark = "✅ OFF (Safe)" if ": off" in result else "❌ ON (Ghost Bandwidth Risk!)"
      
      print(f"[{sw.name}][{intf.name}]: {status_mark}")
      print(f"    Raw: {result}") # 如果需要详细信息可取消注释

  print("="*40 + "\n")

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
    verify_offload_status(net)
    print("[System] Mininet started.")
    monitor = NetworkMonitor(net) # 1. 初始化 Monitor
    
    clean_flow_rules(net, cookie=0xB000, mask=0xFFFF)
    clean_flow_rules(net, cookie=0xA001, mask=0xFFFF)
    
    # --- Step 1: 制造拥塞 ---
    LOAD_FLOW=200
    print(f"\n[Step 1] 注入背景流 (Target: {LOAD_FLOW} Mbps)...")
    tm = flow_gen.generate_traffic_matrix(G_nx.nodes(), total_load_mbps=LOAD_FLOW)
    
    proccesses = flow_gen.apply_traffic_matrix_to_mininet(
      net, tm, G_nx, install_path_rules, duration=100
    )
    
    print(f"[Ghost] There is {len(proccesses)} num of proc")
    print("[Wait] 等待 5 秒让背景流稳定...")
    time.sleep(5)
    
    # --- [NEW] Step 1.5: 感知与输出状态 ---
    print("\n[Monitor] 同步网络状态...")
    monitor.sync_state_to_graph(G_nx) # 读取真实物理状态写入 G_nx
    
    # 1. 打印全网状态 (看看是不是真的堵了)
    print_network_status(G_nx)
    # CLI(net)
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

    install_path_rules(net, path, tos=32, dst_port=12000, cookie=0xA001, do_ping=True)
      
    # 2.2 [诊断] 检查现有的 ITGRecv 是否存活
    # 既然背景流在跑，h_dst 上应该必须有一个 ITGRecv 在运行
    print(f"🔄 在 {h_dst.name} 上启动 VIP ITGRecv (Port 9001)...")
    # 清理旧的 VIP 进程
    h_dst.cmd("pkill -f 'ITGRecv -Sp 9001'") 
    time.sleep(0.2)
    # 启动新进程
    output = h_dst.cmd("ITGRecv -Sp 9001 > /dev/null 2>&1 &")
    print("start ITGRecv 9001: ")
    print(output)
    time.sleep(0.5)

    # 2.5 诊断 VIP 通道
    dst_ip = h_dst.IP()
    print(f"🕵️  [Diag] 诊断 VIP 通道 (TCP 9001)...")
    nc_res = h_src.cmd(f"nc -z -v -w 4 {dst_ip} 9001")
    if "succeeded" not in nc_res:
      print(f"❌ VIP 通道不通: {nc_res.strip()}")
    else:
      print("✅ VIP 通道 (9001) 畅通无阻！")

    # 2. [NEW] 打印目标路径状态 (看看这车是不是往火坑里开)
    print_path_status(G_nx, path)

    # 2.2 下发规则 & 启动接收端
    
    send_log = "/tmp/sender_output.log"
    recv_log = "/tmp/receiver_output.log"
    if os.path.exists(recv_log): os.remove(recv_log)
    
    time.sleep(0.5)

    print(f"[Ghost] There is {len(proccesses)} num of proc")
    # 2.3 启动发送端
    cmd = (f"ITGSend -a {h_dst.IP()} "
           f"-T TCP "
           f"-rp 12000 "
           f"-C 500 -c 1600 -t 15000 "
           f"-Sdp 9001 "
           f'-b 32 '
           f"-l {send_log} "
           f"-x {recv_log}") # 护身符
           
    run_itg_safe(h_src, cmd, timeout_sec=18)

    time.sleep(2)
    # --- Step 3: 验证结果 ---
    print(f"[Ghost] There is {len(proccesses)} num of proc")
    monitor.sync_state_to_graph(G_nx) # 读取真实物理状态写入 G_nx
    # 1. 打印全网状态 (看看是不是真的堵了)
    print_network_status(G_nx)
    print("\n[Step 3] 验证 D-ITG 接收结果:")
    print(f"[Ghost] There is {len(proccesses)} num of proc")
    
    if "No such file" in h_src.cmd(f"ls {send_log}"):
      print("❌ 发送端日志都没生成，ITGSend 彻底崩了。")
    else:
      print("✅ 发送端日志已生成 (说明 D-ITG 运行正常)。")

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