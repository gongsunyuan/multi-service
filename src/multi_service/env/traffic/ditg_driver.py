import os
import subprocess
import shlex
import signal
import uuid
import torch
from time import sleep
from typing import Optional
from ...utils import logger
from ...utils.verbose_logger import vprint_qos
from ..flow_generator import FlowType, FLOW_PROFILES
from ..qos.evaluator import parse_ditg_output, calculate_qoe_reward, calculate_qos_reward

# 根据流类型，返回不同的 D-ITG 命令。
def get_flow_command(
    flow_type: FlowType, 
    target_ip: str, 
    duration_sec: int, 
    sig_port: int = 15000,
    log_file: str | None = None,
    **kwargs
    ) -> str:
    """
    Generates a high-priority D-ITG command string.
    Supports both TCP (Streaming) and UDP (VoIP/Gaming).
    """
    # Map friendly names to FlowType Enum or dict keys
    # Assuming FLOW_PROFILES is a global dict defined elsewhere
    profile = FLOW_PROFILES.get(flow_type) 
  
    assert profile is not None, f"FlowType {flow_type} not found in FLOW_PROFILES"
    protocol = profile['protocol'] # 'UDP' or 'TCP'

    assert protocol in ['UDP', 'TCP'], f"Invalid protocol: {protocol}"
    duration_ms = int(duration_sec * 1000)
  
    # Log file argument
    log_str = f"-x {log_file}" if log_file else ""
    flow_tos = 32
    # 1. Base ITGSend Arguments
    # -a: Target IP
    # -rp: Remote Port (Must match server!)
    # -Sdp: Signaling Port (Must match server!)
    # -t: Duration in ms
    # -T: Transport Protocol
    itg_args = (
        f"-a {shlex.quote(target_ip)} "
        f"-rp 12000 "
        f"-b {flow_tos} "
        f"-Sdp {sig_port} "  # Use same port for signaling to keep it simple
        f"{log_str} "
        f"-t {duration_ms} "
        f"-T {protocol}")

    # 2. Add Payload Parameters (Size/Rate)
    # Priority: Manual args > Preset profile > Default
    if 'ditg_manual' in profile:
        specific_args = profile['ditg_manual']
    elif 'ditg_preset' in profile:
        specific_args = profile['ditg_preset']
    else:
        specific_args = "-C 100 -c 100 " # Safe default

    # 3. Apply "Nuclear Option" (Real-Time Priority)
    # chrt -r 99: Run as Real-Time Round-Robin process with max priority
    # This ensures the marker flow isn't starved by background traffic.
    # wrapper = "chrt -r 99" 
    wrapper = "" 
  
    # 4. Construct Final Command
    final_cmd = f"{wrapper} ITGSend {itg_args} {specific_args}"
  
    return final_cmd

# 启动itg命令
def ensure_server_surgical(host_node, start_port=15000, max_retries=3):
    """
    Ensures an ITGRecv instance is listening on a specific port.
    If the port is busy (TIME_WAIT), it tries the next one (9002, 9003...).
    Does NOT kill all ITGRecv processes, preserving background traffic.
    """
    current_port = start_port
  
    for attempt in range(max_retries):
        # A. Surgical Clean: Kill only the process holding this port
        # netstat flags: -n(numeric) -l(listening) -p(show pid)
        # awk vprints the "PID/ProgramName" column
        check_cmd = f"netstat -nlp | grep :{current_port} | awk '{{print $7}}'"
        pid_info = host_node.cmd(check_cmd).strip()
    
        if pid_info:
            pid = pid_info.split('/')[0]
            if pid.isdigit():
                logger.log(f"Port {current_port} busy by PID {pid}. Cleaning...", tag="Recv Start")
                host_node.cmd(f"kill -9 {pid}")
                sleep(0.1) # Yield to OS

        # B. Start New Server (High Priority)
        try:
            # Note: -Sp defines the signaling port. ITGRecv uses this for setup.
            # chrt is used here too so the receiver doesn't drop packets due to CPU load.
            cmd = f"chrt -r 99 ITGRecv -Sp {current_port}"
      
            # Start process via Mininet's popen
            proc = host_node.popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
      
            # C. Verification (The most important step)
            sleep(0.2) # Allow bind
      
            # Check 1: Is process alive?
            if proc.poll() is not None:
                # Process died immediately
                continue 

            # Check 2: Is port actually listening?
            # -u (udp) -t (tcp) -l (listening) -n (numeric)
            out = host_node.cmd(f"netstat -an | grep :{current_port}")
            if str(current_port) in out:
                return proc, current_port # Success!
      
            # If we got here, process is alive but port isn't open? Kill and retry.
            proc.kill()
        
        except Exception as e:
            logger.log(f"Start failed on {current_port}: {e}", tag="Recv Err")
        
        # Increment port and retry
        current_port += 1

    raise RuntimeError(f"Failed to start ITGRecv on {host_node.name} after {max_retries} attempts.")

# --- 3. Safe Client Execution ---
def run_itg_safe(client_node, server_node, log_file, flow_type, duration_sec, timeout_sec, retry_count=0):
    """
    Orchestrates the measurement:
    1. Starts Server (Surgical) -> Gets Port
    2. Starts Client (Safe Popen) -> Sends to that Port
    3. Handles Timeouts -> Sends SIGINT to save logs
    """
    server_proc = None

    try:
        # --- Step 1: Start Server ---
        server_proc, actual_port = ensure_server_surgical(server_node)
        logger.log(f"Server {server_node.name} listening on {actual_port}", tag="Debug")

        # --- Step 2: Generate Client Command ---
        # Crucial: Client must send to 'actual_port'
        target_ip = server_node.IP()
        cmd = get_flow_command(
            flow_type=flow_type,
            target_ip=target_ip,
            duration_sec=duration_sec,
            sig_port=actual_port, # Sync ports!
            log_file=log_file )

        logger.log(f"{client_node.name} -> {target_ip}:{actual_port} ({flow_type}); Timeout: {timeout_sec}", tag="Debug")
        logger.log(f"Send command: {cmd}", tag="Debug")
        # --- Step 3: Start Client ---
        # os.setsid creates a new process group, allowing us to kill the whole tree later
        client_proc = client_node.popen(
            cmd, 
            shell=True, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            preexec_fn=os.setsid 
        )

        # --- Step 4: Wait with Timeout ---
        try:
            stdout, stderr = client_proc.communicate(timeout=timeout_sec)
            # Check for immediate D-ITG errors in stderr
            logger.log(f"{stdout}", tag="Debug")
            if stderr:
                err_str = stderr.decode('utf-8', errors='ignore')
                if "Connection refused" in err_str or "Connect error" in err_str:
                    raise ConnectionError(err_str)
      
            return True # Success

        except subprocess.TimeoutExpired as e:
            logger.log(f"Flow timed out (> {timeout_sec}s). Saving logs...", tag="Send Fail")
            logger.log(f"ITGSend output: {e.stdout}", tag="Send Fail")
            # Graceful Shutdown: Send SIGINT to the Process Group
            # This tells D-ITG to stop sending and flush logs to disk
            try:
                os.killpg(os.getpgid(client_proc.pid), signal.SIGINT)
                client_proc.communicate(timeout=2) # Give it 2s to write file
            except:
                logger.log("Process unresponsive.", tag="Send Fail")
                os.killpg(os.getpgid(client_proc.pid), signal.SIGKILL)
      
            # For TCP, a timeout is a valid result (congestion), not necessarily a crash.
            # We return True so the log parser can see the packet loss/delay.
            return True 

    except ConnectionError as e:
        # --- Step 5: Retry Logic ---
        if retry_count < 2: # Retry once
            logger.log(f"Connection failed. Retrying...", tag="Send Retry")
            # Clean up server before retrying
            if server_proc: 
                server_proc.terminate()
                server_proc.wait()
            return run_itg_safe(client_node, server_node, log_file, flow_type, duration_sec, timeout_sec, retry_count + 1)
        else:
            logger.log(f"Connection refused after retries.", tag="Send Err")
            return False

    except Exception as e:
        logger.log(f"Execution failed: {e}", tag="Send Err")
        return False

    finally:
        # --- Step 6: Cleanup Server ---
        if server_proc:
            try:
                server_proc.terminate()
                server_proc.wait(timeout=1)
            except:
                server_proc.kill()

def measure_path_qos(server, client, path_route, flow_type, config, resend=False):
    """
    [完美健壮版] 解决时序竞争与解析失败
    """
    # --- 1. 给 OVS 流表下发一点“呼吸时间” ---
    # 解决代码跑得比交换机快的问题
    sleep(0.2) 

    # --- 3. 准备 D-ITG 参数 ---
    random_id = uuid.uuid4().hex[:8]
    log_prefix = f"/dev/shm/itg_{client.name}_{server.name}_{random_id}"
    recv_log = f"{log_prefix}.recv"  
  
    # target_duration = 6 if flow_type==FlowType.STREAMING else 2

    target_duration = 1.5

    if flow_type == FlowType.STREAMING:
        # TCP 给 2.5 倍余量，防止拥塞误杀
        safe_timeout = target_duration* 2.5 + 4  # 6*2.5 + 2 = 17s
    else:
        # UDP 给 2 秒余量即可
        safe_timeout = target_duration + 4  # 2 + 4 = 6s
  
    success = run_itg_safe(
        client_node=client,
        server_node=server,
        log_file=recv_log,
        flow_type=flow_type,
        duration_sec =target_duration,
        timeout_sec=int(safe_timeout))

    if not success:
        # 如果 run_itg_safe 返回 False (连接彻底失败)，直接返回 -1
        return -1.0, -1.0

    # if stderr: return 0
    # 检查文件是否存在 (防止传输完全失败导致无日志)
    check_log = server.cmd(f"ls {recv_log}")
    if "No such file" in check_log and not resend:
        logger.log("No log generated. Resend same cmd again", tag="Sender Err")
        client.cmd(f"rm -f {recv_log}")
        return measure_path_qos(server, client, path_route, flow_type, config, resend=True)

    # 运行解码器拿到文本结果 
    # 解析结果 
    # Meta-DRL 
    try:
        logger.log(f"Running ITGDec on {recv_log}...", tag="Debug")

        # 1. 启动进程
        with client.popen(
            f"ITGDec {recv_log}", 
            shell=True,
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            text=True 
        ) as dec_proc:

            # 2. 等待进程结束并获取输出 (Block until finished)
            # communicate 会读取 stdout 直到 EOF，确保拿到了所有输出
            stdout, stderr = dec_proc.communicate(timeout=6) # 设置个超时防止卡死

            if dec_proc.returncode != 0 and not resend:
                dec_proc.kill()
                logger.log(f"ITGDec failed with code {dec_proc.returncode}", tag="Debug")
                return measure_path_qos(server, client, path_route, flow_type, config, resend=True)
            else:
                dec_output = stdout
                logger.log("Success dec recieve file", tag="Debug")

    except subprocess.TimeoutExpired:
        logger.log("ITGDec Timed out!", tag="Debug")
        dec_proc.kill()
        dec_output = ""

    finally:
        # 清理所有临时文件 
        client.cmd(f"rm -f {recv_log}")

    # 2. 解析 D-ITG 输出
    qos_metrics, no_packet_arrive = parse_ditg_output(dec_output)    
    if no_packet_arrive:
        if not resend :
            logger.log(f"No packet arrive : Resend cmd again", tag="Sender Err")
            return measure_path_qos(server, client, path_route, flow_type, config, resend=True)
        else :
            logger.log(f"Fail to send packet, bad path", tag="Sender Err")
            return -1.0, -1.0

    # 假设 qos_metrics 里的数值已经拿到了
    d = qos_metrics['delay']
    j = qos_metrics['jitter']
    b = qos_metrics['bandwidth']
    l = qos_metrics['loss_rate']
    
    vprint_qos(flow_type.name, delay=d, jitter=j, bw=b, loss=l, tag="QoS")

    # 计算 Reward
    qoe_reward = calculate_qoe_reward(qos_metrics, FLOW_PROFILES[flow_type])
    qos_reward = calculate_qos_reward(
        delay_ms=qos_metrics['delay'],
        loss_percent=qos_metrics['loss_rate'],
        jitter_ms=qos_metrics['jitter'],
        flow_type_str=flow_type.name,
        config=config
    )

    return qos_reward, qoe_reward

# 发送流量并捕获包特征
def send_packet_and_capture(
    server, 
    client, 
    flow_type: FlowType, 
    duration_sec=15, 
    n_packets_to_capture=30, 
    **flow_params):
    """
    在 Mininet 中运行 D-ITG 流量, 并同时使用 tshark 管道实时捕获特征。
    [Fix] 集成了 ensure_server_surgical 以防止端口冲突。
    """

    server_ip = server.IP()
    client_ip = client.IP()
  
    # 1. 查找监听接口
    server_intf = None
    for intf in server.intfList():
        if intf.name != 'lo' and intf.link:
            server_intf = intf
            break
    if server_intf is None:
        raise Exception(f"在 {server.name} 上找不到已连接的数据接口!")

    switch_intf = server_intf.link.intf2 if server_intf.link.intf1 == server_intf else server_intf.link.intf1
    switch_intf_name = switch_intf.name
  
    feature_matrix = []
    client_proc = None
    tshark_proc = None
    server_proc = None

    try:
        # 2. [Action 1] 安全启动服务端 (获取动态端口)
        # 这会清理旧进程并返回一个干净的端口
        server_proc, actual_port = ensure_server_surgical(server)

        # 3. [Action 2] 获取客户端命令 (使用动态端口)
        client_cmd = get_flow_command(
            flow_type=flow_type,
            target_ip=server_ip,
            duration_sec=duration_sec,
            sig_port=actual_port, # [Fix] 使用实际端口
            **flow_params)

        MARK_TOS = 32
    
        # 4. [Action 3] 启动 tshark
        display_filter = f"src host {client_ip} and dst host {server_ip} and ip[1] == {MARK_TOS}"
        # 增加一点超时余量
        timeout_duration = duration_sec + 5

        tshark_cmd = [
            'sudo', 'tshark',
            '-c', str(n_packets_to_capture),
            '-a', f'duration:{timeout_duration}',
            '-i', switch_intf_name,
            '-l', 
            '-T', 'fields',
            '-e', 'frame.len',
            '-e', 'frame.time_delta',
            '-e', 'ip.src',
            '-e', 'ip.dst',
            '-E', 'separator=,',
            '-f', display_filter]
  
        tshark_proc = subprocess.Popen(
            tshark_cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.DEVNULL,
            text=True
        )

        # 5. [Action 4] 启动客户端流量
        sleep_time = 1.0 # 给 tshark 一点启动时间
        sleep(sleep_time)
    
        # 使用 popen 启动客户端
        client_proc = client.popen(client_cmd, shell=True)

        # 6. [核心] 实时读取
        assert tshark_proc.stdout is not None, "tshark stdout is None"
  
        for line in tshark_proc.stdout:
            line = line.strip()
            if not line: continue
            try:
                size_str, iat_str, src_ip, dst_ip = line.split(',')
                size = float(size_str)
                try:
                    iat = float(iat_str)
                except ValueError:
                    iat = 0.0
        
                feature_vector = [size, iat]
                feature_matrix.append(feature_vector)
            except ValueError:
                logger.log("Grap packet failed!", tag="Tshark Err")
                pass # 忽略解析错误
      
    except Exception as e:
        logger.log(f"采集指纹出错: {e}", tag="Tshark Err")
  
    finally:
        # [Fix] 统一清理资源
        if tshark_proc:
            tshark_proc.kill()
    
        if client_proc:
            # 向进程组发送信号，确保杀死 chrt 启动的子进程
            try:
                client_proc.kill()
                # 如果使用了 os.setsid (虽然这里没显式用，但为了保险)
                # os.killpg(os.getpgid(client_proc.pid), signal.SIGKILL)
            except:
                pass
          
        if server_proc:
            try:
                server_proc.terminate()
                server_proc.wait(timeout=0.5)
            except:
                server_proc.kill()

    # 如果没抓到包，返回全0或者随机噪声防止报错，但在训练初期这可能导致冷启动问题
    if len(feature_matrix) == 0:
        return torch.zeros((n_packets_to_capture, 2), dtype=torch.float32)

    fingerprint_tensor = torch.tensor(feature_matrix, dtype=torch.float32)
    return fingerprint_tensor

# 将特征向量归一化
def normalize_fingerprint(tensor: torch.Tensor) -> torch.Tensor:
    """
    对流量指纹 Tensor 进行归一化处理。
    输入形状: (N, 3) -> [Size, IAT, Direction]
    """
    # 1. 克隆 Tensor 以免修改原始数据 (可选)
    norm_tensor = tensor.clone()
  
    # --- 列 0: 包大小 (Size) ---
    # 使用 Min-Max 归一化。
    # 网络包最大通常是 1514 (MTU + Ethernet Header)。
    # 将其缩放到 [0, 1] 范围内。
    norm_tensor[:, 0] = norm_tensor[:, 0] / 1600.0
    
    iat_cap = 0.1  # 100ms
    iat_clamped = torch.clamp(norm_tensor[:, 1], max=iat_cap)   # 截断 (Clamp)
    norm_tensor[:, 1] = iat_clamped / iat_cap                   # 归一化
    
    return norm_tensor

# 获取一个流量特征张量
def get_a_fingerprint(
    server, 
    client, 
    flow_type: FlowType, 
    n_packets_to_capture=30, 
    **flow_params):

    duration_sec = 15 

    final_tensor = send_packet_and_capture(
        server=server,
        client=client,
        flow_type=flow_type,
        duration_sec=duration_sec,
        n_packets_to_capture=n_packets_to_capture)
  
    while final_tensor.size(0) < 30:
        sleep(1)
        final_tensor = send_packet_and_capture(
            server=server,
            client=client,
            flow_type=flow_type,
            duration_sec=duration_sec,
            n_packets_to_capture=n_packets_to_capture)
  
    return normalize_fingerprint(final_tensor).unsqueeze(0)