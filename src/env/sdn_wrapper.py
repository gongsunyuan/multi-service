import time
import torch
import networkx as nx
import random
from mininet.cli import CLI
from ..utils import logger
from . import sdn_controller as mc
from .flow_generator import FlowType
from .flow_generator import FlowGenerator
from .network_generator import TopologyGenerator, get_pyg_data_from_nx

class SdnWrapper:

  # 初始化背景流
  def __init__(self, config):
    """
    __init__ 的 Docstring
    
    :param config: 
      graph_path: 图绝对路径
      max_steps: 最大寻路跳数
      bg_cookie: 背景流cookie
      flow_cookie: 业务流cookie
      bg_duration: 背景流持续时间
      total_flow_load: 默认背景流负载
    """
    self.config = config

    # --- 1. 基础组件初始化 ---
    self.flow_gen = FlowGenerator()
    self.topo_gen = TopologyGenerator()
    self.blueprint_G = self.topo_gen.load_topology(config.path.graph_path)
    self.current_G = self.blueprint_G.copy()

    # --- 2. 启动 Mininet (修复顺序 Bug) ---
    # 必须先创建生成器对象，再 enter
    self.net_gen = mc.get_a_mininet(self.blueprint_G.copy(), is_test=False)
    self.net = self.net_gen.__enter__()
    
    # 启动后才能初始化监控
    self.monitor = mc.NetworkMonitor(self.net)
    
    # --- 3. 状态管理 ---
    self.bg_processes = []
    self.bg_start_time = 0
    self.active_cookies = set()

    # --- 4. Cookie 与 配置 ---
    self.cookie_mask = 0xF000
    self.bg_cookie_start = getattr(config.env, 'bg_cookie', 0xB000)
    self.agent_cookie_start = getattr(config.env, 'flow_cookie', 0xA000)
    self.bg_duration = getattr(config.env, 'bg_duration', 60) 
    
    # 任务状态
    self.s_node = None
    self.d_node = None
    self.current_flow_type = None
    self.step_count = 0 
    self.path_so_far = [] 
    self.current_node = None 
    self.dist_matrix = dict(nx.all_pairs_shortest_path_length(self.blueprint_G))
    self.shaping_weight = getattr(config.env, 'shaping_weight', 0.1)
  
  # 重新启动背景流量/重新发送业务流
  def reset_bg(self, current_load_mbps=None):
    """
    Args:
      force_hard (bool): 强制执行 Hard Reset
      current_load_mbps (float): [课程学习] 指定当前的背景流量负载
    """
    target_load = current_load_mbps if current_load_mbps else self.config.env.total_load_mbps
    self._hard_reset_background_traffic(target_load)

    return self.get_observation()

  def reset_flow(self):
    mc.clean_flow_rules(self.net, cookie=self.agent_cookie_start, mask=self.cookie_mask)
    return self.get_observation()
  
  # 动作掩码
  def get_action_mask(self):
    """
    生成动作掩码 (True 代表合法动作)
    """
    edge_index = self.observation_data.edge_index
    num_edges = edge_index.shape[1]
    mask = torch.zeros(num_edges, dtype=torch.bool)
    
    # 获取所有边的起点
    source_nodes = edge_index[0, :]
    
    # 1. 基础规则：边的起点必须是当前节点
    is_neighbor = (source_nodes == self.current_node)
    
    # 2. 进阶规则：禁止走回头路 (除非只有回头路可走)
    if len(self.path_so_far) > 1:
      prev_node = self.path_so_far[-2]
      target_nodes = edge_index[1, :]
      is_backtrack = (target_nodes == prev_node)
      # 合法 = 是邻居 AND 不是回头路
      valid = is_neighbor & (~is_backtrack)
      
      # 兜底：如果剔除回头路后无路可走（死胡同），则允许回头
      if valid.any():
        mask = valid
      else:
        mask = is_neighbor
    else:
      mask = is_neighbor
        
    return mask
  
  # 更新图状态，返回图状态
  def get_observation(self):
    # --- 1. 物理同步 ---
    self.current_G = self.monitor.sync_state_to_graph(self.blueprint_G.copy(), duration=0.05)
    
    # --- 2. 特征提取 ---
    # Hop-by-Hop 关键：S_node 填 self.current_node
    data, _ = get_pyg_data_from_nx(
      self.current_G.copy(), 
      self.current_node, 
      self.d_node,      
      self.config )
    
    self.observation_data = data
    data.action_mask = self.get_action_mask()

    return data
  
  # 结束环境
  def close(self):
    logger.log("Closing environment...", tag="Env Close")
    self._kill_bg_processes()
    # 退出前最后清理一次，保持宿主机干净
    mc.clean_flow_rules(self.net, cookie=self.bg_cookie_start, mask=self.cookie_mask)
    mc.clean_flow_rules(self.net, cookie=self.agent_cookie_start, mask=self.cookie_mask)
    
    if hasattr(self, 'net_gen') and self.net_gen:
      try:
        self.net_gen.__exit__(None, None, None)
      except Exception as e:
        print(f"Error closing mininet context: {e}")

  # 重置背景流，打印新的网络状态
  def _hard_reset_background_traffic(self, load_mbps):
    """
    环境重置：注入指定负载的背景流
    """
    logger.log(f">>> Hard Reset | Load: {load_mbps:.1f} Mbps <<<", tag="Env Reset")
    
    # 1. 清理
    self._kill_bg_processes()
    mc.clean_flow_rules(self.net, cookie=self.bg_cookie_start, mask=self.cookie_mask)
    mc.clean_flow_rules(self.net, cookie=self.agent_cookie_start, mask=self.cookie_mask)
    self.active_cookies.clear()
    
    # 2. 生成 (使用传入的动态负载)
    tm = self.flow_gen.generate_traffic_matrix(
      self.blueprint_G.nodes(), 
      self.blueprint_G.copy(), 
      total_load_mbps=load_mbps 
    )
    
    # 3. 注入 (1.5倍冗余时长)
    safe_duration = int(self.bg_duration * 1.5)
    self.bg_processes = self.flow_gen.apply_traffic_matrix_to_mininet(
      self.net, tm, self.blueprint_G.copy(), 
      install_rules_func=mc.install_path_rules,
      duration=safe_duration
    )
    
    # 4. 更新时间戳
    self.bg_start_time = time.time()
    
    # 5. 等待流量稳定 (给 TCP 爬坡和队列积压一点时间)
    time.sleep(3)

    for _ in range(3):
      self.get_observation()
      time.sleep(0.5)

    logger.log_network_status(self.current_G.copy())

  # 清空背景流
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

  # 执行一跳动作，获取环境奖励
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
      logger.log("Action index out of bounds!", tag="Env Err")
      return self.get_observation(), -2.0, True, {'error': 'index_error'}

    # 2. 合法性检查
    if u != self.current_node:
      logger.log(f"Illegal Move: {self.current_node} -> {u} impossible. Valid edges start from {self.current_node}", tag="Env Err")
      import traceback
      traceback.print_exc()
      exit()
      return self.get_observation(), -5.0, True, {'error': 'illegal_move'}

    # 3. 执行移动
    self.current_node = v
    self.path_so_far.append(v)
    self.step_count += 1
    
    done = False
    old_dist = self.dist_matrix[u][self.d_node]
    new_dist = self.dist_matrix[v][self.d_node]
    progress = old_dist - new_dist

    shaping_reward = progress * self.shaping_weight
    step_reward = -0.01 + shaping_reward
    info = {'flow_type': self.current_flow_type.name}

    # 4. 状态判定
    if v == self.d_node:
      # === Success ===
      done = True
      
      # 生成 Cookie
      cookie = self.agent_cookie_start + len(self.active_cookies) % 4096
      self.active_cookies.add(cookie)

      # 打印路径状态 (用于调试)
      logger.log(f"Path: {self.path_so_far}, FlowType: {self.current_flow_type.name.upper()}", tag="Agent Path")
      
      protocol_str = 'TCP' if self.current_flow_type == FlowType.STREAMING else 'UDP'
      # A. 下发规则
      mc.install_path_rules(
        self.net, self.path_so_far, 
        tos=32, dst_port=12000, cookie=cookie, protocol=protocol_str)

      # B. 测量 QoS
      src_host = self.net.get(f'h{self.s_node}')
      dst_host = self.net.get(f'h{self.d_node}')
      
      qos_reward, qoe_reward = mc.measure_path_qos(
        server=dst_host,
        client=src_host,
        path_route=self.path_so_far,
        flow_type=self.current_flow_type,
        config=self.config )
      
      # 处理测量失败的情况
      if qos_reward == -1 or qos_reward is None:
        logger.log("Measurement Failed! Punishment applied.", tag="Env Warn")
        total_reward = -1.0 # 测量失败视为路径不可达
      else:
        logger.log(f"QoS: {qos_reward:.4f} | QoE: {qoe_reward:.4f} | Path: {self.path_so_far}", tag="Env Done")
        total_reward = qos_reward # 这里你可以选择返回 qos_reward 还是 qoe_reward

      info['qos'] = qos_reward
      info['path'] = self.path_so_far

      return self.get_observation(), total_reward, True, info

    elif v in self.path_so_far[:-1]:
      # === Loop ===
      done = True
      reward = -1.0 
      info['error'] = 'loop_detected'

    elif self.step_count >= self.config.train.max_steps:
      # === Timeout ===
      done = True
      reward = -2.0 # 超时惩罚
      info['error'] = 'max_steps'

    # 5. 更新观察 (Observation)
    # 下一跳的 Observation 中，Source 特征可能需要变为 Current Node
    # 或者你需要让 GNN 知道 "Current Position" 在哪里
    next_state = self.get_observation()
    return next_state, step_reward, done, info

