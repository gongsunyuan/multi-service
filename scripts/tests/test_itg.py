import os
import sys
import time
from mininet.net import Mininet
from mininet.node import OVSKernelSwitch
from mininet.link import TCLink
from mininet.log import setLogLevel, info

def debug_itg_mechanism():
    setLogLevel('info')
    print("====== [Debug] D-ITG Deep Diagnostic ======")
    
    # 1. 强力清理 (清除所有残留进程)
    print("[1] Cleaning up environment...")
    os.system("sudo killall -9 ITGSend ITGRecv 2>/dev/null")
    os.system("sudo mn -c >/dev/null 2>&1")
    
    # 2. 搭建最小拓扑 (h1 <--> s1 <--> h2)
    print("[2] Starting Mininet...")
    net = Mininet(switch=OVSKernelSwitch, link=TCLink)
    h1 = net.addHost('h1')
    h2 = net.addHost('h2')
    s1 = net.addSwitch('s1')
    net.addLink(h1, s1, bw=100, delay='1ms') # 100Mbps 链路
    net.addLink(s1, h2, bw=100, delay='1ms')
    
    try:
        net.start()
        h1.cmd("ovs-ofctl add-flow s1 actions=normal") # 确保连通
        
        # 3. 启动接收端 (带详细日志)
        print("[3] Starting ITGRecv on h2...")
        # 关键：把 ITGRecv 的日志也打出来，看看它有没有报错 (比如 Bind failed)
        h2.cmd("ITGRecv > /tmp/recv_debug.log 2>&1 &")
        time.sleep(1)
        
        # 检查 ITGRecv 是否活着
        pid_recv = h2.cmd("pgrep -f ITGRecv").strip()
        if not pid_recv:
            print("❌ ITGRecv failed to start! Check /tmp/recv_debug.log")
            os.system("cat /tmp/recv_debug.log")
            return
        else:
            print(f"✅ ITGRecv running (PID: {pid_recv})")

        # 4. 构造聚合流命令 (模拟你 FlowGenerator 里的参数)
        # 假设我们聚合出了一条 30Mbps 的大流
        # KBps = 30 * 1000 / 8 = 3750 KB/s
        kbps = 3750 
        duration_ms = 5000
        
        dst_ip = h2.IP()
        send_log_path = "/tmp/sender_test.log"
        debug_log_path = "/tmp/send_debug.log"
        
        # 确保文件不存在
        os.system(f"rm -f {send_log_path} {debug_log_path}")
        
        print(f"[4] Starting ITGSend on h1 (Target: {kbps} KBps)...")
        
        # [关键测试] 
        # 1. 使用 -l 生成二进制日志
        # 2. 显式重定向 stdout/stderr
        # 3. 这里的参数完全复刻你的 Generator
        cmd = (
            f"ITGSend -a {dst_ip} "
            f"-T UDP "
            f"-k {kbps} "
            f"-c 1000 "
            f"-t {duration_ms} "
            f"-l {send_log_path} "
            f"> {debug_log_path} 2>&1 &"
        )
        
        print(f"   Command: {cmd}")
        h1.cmd(cmd)
        
        # 5. 监控运行状态
        print("[5] Monitoring process for 3 seconds...")
        for i in range(3):
            time.sleep(1)
            pid_send = h1.cmd("pgrep -f ITGSend").strip()
            if pid_send:
                print(f"   Time {i+1}s: ITGSend is ALIVE (PID: {pid_send})")
            else:
                print(f"   Time {i+1}s: ITGSend is DEAD ❌")
                break
        
        # 6. 结果分析
        print("\n[6] Analysis:")
        
        # 检查 Debug Log
        if os.path.getsize(debug_log_path) > 0:
            print(f"   [Debug Log Content]:")
            os.system(f"cat {debug_log_path}")
        else:
            print(f"   [Debug Log] is EMPTY (No standard output).")
            
        # 检查 Binary Log
        if os.path.exists(send_log_path) and os.path.getsize(send_log_path) > 0:
            print(f"   [Send Log] FOUND ({os.path.getsize(send_log_path)} bytes). Decoding...")
            # 尝试解码
            decode_out = h1.cmd(f"ITGDec {send_log_path}")
            print(decode_out)
            
            if "Total packets" in decode_out:
                print("✅ TEST PASSED: Traffic generated successfully.")
            else:
                print("⚠️ TEST WARNING: Log exists but decoding weird.")
        else:
            print("❌ TEST FAILED: Binary log file not created or empty.")
            
    except Exception as e:
        print(f"Error: {e}")
    finally:
        net.stop()
        os.system("sudo killall -9 ITGSend ITGRecv 2>/dev/null")

if __name__ == "__main__":
    if os.getuid() != 0:
        print("Run with sudo")
    else:
        debug_itg_mechanism()