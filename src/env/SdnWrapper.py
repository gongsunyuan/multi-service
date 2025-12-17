import time
import networkx as nx
from src.env import MininetController as mc
from src.env.FlowGenerator import FlowGenerator
from src.env.NetworkGenerator import TopologyGenerator, get_pyg_data_from_nx
from src.utils.VerbosePrint import vprint, vprint_network_status, vprint_path_status, vprint_qos

class SdnWrapper:
  def __init__(self, config):
    self.config = config
    
    # --- 1. 基础组件初始化 ---
    self.flow_gen = FlowGenerator()
    self.topo_gen = TopologyGenerator()
    self.blueprint_G = self.topo_gen.load_topology(config.graph_path)
    self.current_G = self.blueprint_G.copy()

    # --- 2. 启动 Mininet (修复顺序 Bug) ---
    # 必须先创建生成器对象，再 enter
    self.net_gen = mc.get_a_mininet(self.blueprint_G, is_test=False)
    self.net = self.net_gen.__enter__()
    
    # 启动后才能初始化监控
    self.monitor = mc.NetworkMonitor(self.net)
    
    # --- 3. 状态管理 ---
    self.bg_processes = []
    self.bg_start_time = 0
    self.active_cookies = set()

    # --- 4. Cookie 与 配置 ---
    self.cookie_mask = 0xF000
    self.bg_cookie_start = getattr(config, 'bg_cookie', 0xB000)
    self.agent_cookie_start = getattr(config, 'flow_cookie', 0xA000)
    self.bg_duration = getattr(config, 'bg_duration', 60) 
    
    # 任务状态
    self.s_node = None
    self.d_node = None
    self.current_flow_type = None
    self.step_count = 0
    self.path_so_far = [] 
    self.current_node = None

  def reset(self, force_hard=False, current_load_mbps=None):
    """
    智能重置函数。
    自动判断是执行 "任务重置" 还是 "环境重置"。
    
    Args:
      force_hard (bool): 强制执行 Hard Reset
      current_load_mbps (float): [课程学习] 指定当前的背景流量负载
    """
    
    # 1. 检查是否需要 Hard Reset (换地图)
    current_time = time.time()
    time_elapsed = current_time - self.bg_start_time
    
    # 检查存活进程比例
    live_procs = [p for p in self.bg_processes if p.poll() is None]
    is_bg_dead = len(live_procs) < len(self.bg_processes) * 0.5 if self.bg_processes else True
    
    # 判定逻辑：强制 OR 超时 OR 进程死亡 OR (关键)负载需求变了
    # 如果外部传入了新的 load 且与 config 不一致(通常意味着 Epoch 变了)，建议也 Hard Reset
    need_hard = force_hard or (time_elapsed > self.bg_duration) or is_bg_dead
    
    if need_hard:
      # 如果外部没传 load，就用 config 默认值
      target_load = current_load_mbps if current_load_mbps else self.config.total_load_mbps
      self._hard_reset_background_traffic(target_load)
    else:
      # Soft Reset: 只清理 Agent 留下的痕迹
      mc.clean_flow_rules(self.net, cookie=self.agent_cookie_start, mask=self.cookie_mask)

    # 2. 执行 Soft Reset (换任务)
    self._soft_reset_task()

    # 3. 初始化游走状态
    self.current_node = self.s_node
    self.path_so_far = [self.s_node]
    self.step_count = 0

    return self.get_observation()

  def get_observation(self):
    # --- 1. 物理同步 ---
    self.current_G = self.monitor.sync_state_to_graph(self.blueprint_G, duration=0.05)
    
    # --- 2. 特征提取 ---
    # Hop-by-Hop 关键：S_node 填 self.current_node
    data, _ = get_pyg_data_from_nx(
      self.current_G, 
      self.current_node, 
      self.d_node,      
      self.config
    )
    
    self.observation_data = data
    return data
  
  def _hard_reset_background_traffic(self, load_mbps):
    """
    环境重置：注入指定负载的背景流
    """
    vprint(f">>> Hard Reset | Load: {load_mbps:.1f} Mbps <<<", tag="Env Reset")
    
    # 1. 清理
    self._kill_bg_processes()
    mc.clean_flow_rules(self.net, cookie=self.bg_cookie_start, mask=self.cookie_mask)
    mc.clean_flow_rules(self.net, cookie=self.agent_cookie_start, mask=self.cookie_mask)
    self.active_cookies.clear()
    
    # 2. 生成 (使用传入的动态负载)
    tm = self.flow_gen.generate_traffic_matrix(
      self.blueprint_G.nodes(), 
      self.blueprint_G, 
      total_load_mbps=load_mbps # <--- 动态负载
    )
    
    # 3. 注入 (1.5倍冗余时长)
    safe_duration = int(self.bg_duration * 1.5)
    self.bg_processes = self.flow_gen.apply_traffic_matrix_to_mininet(
      self.net, tm, self.blueprint_G, 
      install_rules_func=mc.install_path_rules,
      duration=safe_duration
    )
    
    # 4. 更新时间戳
    self.bg_start_time = time.time()
    
    # 5. 等待流量稳定 (给 TCP 爬坡和队列积压一点时间)
    time.sleep(3)

    for _ in range(3):
      self.monitor.sync_state_to_graph(self.blueprint_G.copy())
      time.sleep(0.5)

    vprint_network_status(self.current_G)

  def _soft_reset_task(self):
    self.s_node, self.d_node = self.topo_gen.select_source_destination()
    self.current_flow_type, _ = self.flow_gen.get_random_flow()

  def _kill_bg_processes(self):
    for proc in self.bg_processes:
      try:
        proc.terminate() # 先尝试温和终止
        proc.wait(timeout=0.1)
      except:
        try:
          proc.kill() # 强制杀死
        except:
          pass
    self.bg_processes = []

  def close(self):
    vprint("Closing environment...", tag="Env Close")
    self._kill_bg_processes()
    # 退出前最后清理一次，保持宿主机干净
    mc.clean_flow_rules(self.net, cookie=self.bg_cookie_start, mask=self.cookie_mask)
    mc.clean_flow_rules(self.net, cookie=self.agent_cookie_start, mask=self.cookie_mask)
    
    if hasattr(self, 'net_gen') and self.net_gen:
      try:
        self.net_gen.__exit__(None, None, None)
      except Exception as e:
        print(f"Error closing mininet context: {e}")

  def step(self, action_edge_idx):
    """
    Returns: next_state, reward, done, info
    """
    # 1. 解析动作
    try:
        # 增加防越界保护
        u = self.observation_data.edge_index[0, action_edge_idx].item()
        v = self.observation_data.edge_index[1, action_edge_idx].item()
    except IndexError:
        vprint("Action index out of bounds!", tag="Env Err")
        return self.get_observation(), -2.0, True, {'error': 'index_error'}

    # 2. 合法性检查
    if u != self.current_node:
      vprint(f"Illegal Move: {self.current_node} -> {u} impossible. Valid edges start from {self.current_node}", tag="Env Err")
      # 严重惩罚，这通常意味着 Mask 没做好
      return self.get_observation(), -5.0, True, {'error': 'illegal_move'}

    # 3. 执行移动
    self.current_node = v
    self.path_so_far.append(v)
    self.step_count += 1
    
    done = False
    reward = -0.01 # 步数惩罚 (Step Penalty)，鼓励短路径
    info = {'flow_type': self.current_flow_type.name}

    # 4. 状态判定
    if v == self.d_node:
      # === Success ===
      done = True
      
      # 生成 Cookie
      cookie = self.agent_cookie_start + len(self.active_cookies) % 4096
      self.active_cookies.add(cookie)

      # 打印路径状态 (用于调试)
      # vprint_path_status(self.current_G, self.path_so_far)

      # A. 下发规则
      mc.install_path_rules(
        self.net, self.path_so_far, 
        tos=32, dst_port=12000, cookie=cookie
      )
      
      # B. 测量 QoS
      src_host = self.net.get(f'h{self.s_node}')
      dst_host = self.net.get(f'h{self.d_node}')
      
      qos_reward, qoe_reward = mc.measure_path_qos(
        server=dst_host,
        client=src_host,
        path_route=self.path_so_far,
        flow_type=self.current_flow_type,
        config=self.config
      )
      
      # [鲁棒性修复] 处理测量失败的情况
      if qos_reward == -1 or qos_reward is None:
          vprint("Measurement Failed! Punishment applied.", tag="Env Warn")
          reward = -1.0 # 测量失败视为路径不可达
      else:
          vprint(f"QoS: {qos_reward:.4f} | QoE: {qoe_reward:.4f} | Path: {self.path_so_far}", tag="Env Done")
          reward = qos_reward # 这里你可以选择返回 qos_reward 还是 qoe_reward

      info['qos'] = qos_reward
      info['path'] = self.path_so_far

    elif v in self.path_so_far[:-1]:
      # === Loop ===
      done = True
      reward = -1.0 
      info['error'] = 'loop_detected'

    elif self.step_count >= self.config.max_steps:
      # === Timeout ===
      done = True
      reward = -2.0 # 超时惩罚
      info['error'] = 'max_steps'

    else:
      # === 情况 D: 继续赶路 (Running) ===
      done = False
      reward = -0.01 # 可选：给予微小的步数惩罚，鼓励最短路

    # 5. 更新观察 (Observation)
    # 下一跳的 Observation 中，Source 特征可能需要变为 Current Node
    # 或者你需要让 GNN 知道 "Current Position" 在哪里
    next_state = self.get_observation()
    return next_state, reward, done, info
  