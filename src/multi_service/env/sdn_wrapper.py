from omegaconf import DictConfig, ListConfig
from torch_geometric.data.data import Data
from loguru import logger
import time
from typing import Any, Literal
import torch
import networkx as nx
import random
from mininet.cli import CLI
from omegaconf import DictConfig
from . import sdn_controller as mc
from .flow_generator import FLOW_PROFILES, FlowType
from .flow_generator import FlowGenerator
from .network_generator import TopologyGenerator
from ..utils import AttrDict, get_graph_data



class SdnWrapper:
    # 初始化背景流
    def __init__(
        self, 
        topo: nx.Graph, 
        src_nodes: list,
        dst_nodes: list,
        max_steps: int,
        train_stats: dict | DictConfig,
        eval_stats: dict | DictConfig,
        qos_reward: dict | DictConfig,
        bg_cookie: int = 0xB000,
        flow_cookie: int = 0xA000,
        bg_duration: int = 60,
        shaping_weight: float = 0.1,
        total_load_mbps: float = 100.0,
        bg_traffic_nodes: list | None = None) -> None:
        """
        __init__ 的 Docstring
        
        :param 
            topo: 拓扑图
            src_nodes: 源节点列表
            dst_nodes: 目的节点列表
            max_steps: 最大寻路跳数
            topo_stats: 拓扑统计信息 (for get_graph_data)
            qos_reward: QoS 奖励配置 (for measure_path_qos)
            bg_cookie: 背景流cookie
            flow_cookie: 业务流cookie
            bg_duration: 背景流持续时间
            shaping_weight: 流量整形权重
            total_load_mbps: 默认背景流负载
            bg_traffic_nodes: 允许生成背景流的节点列表 (默认为 None，表示全图)
        """
        self.bg_traffic_nodes = bg_traffic_nodes
        self.total_load_mbps = total_load_mbps
        self.max_steps = max_steps
        
        self.topo_stats = train_stats
        self.train_stats = train_stats
        self.eval_stats = eval_stats

        self.qos_reward_config = qos_reward

        # --- 1. 基础组件初始化 ---
        self.flow_gen = FlowGenerator()
        self.topo_gen = TopologyGenerator()

        self.blueprint_G = topo
        self.current_G = self.blueprint_G.copy()

        self.src_nodes = src_nodes
        self.dst_nodes = dst_nodes

        # --- 2. 启动 Mininet (修复顺序 Bug) ---
        # 必须先创建生成器对象，再 enter
        self.net_gen = mc.get_a_mininet(self.blueprint_G.copy(), is_test=False)
        self.net = self.net_gen.__enter__()
        
        # 启动后才能初始化监控
        self.monitor = mc.NetworkMonitor(self.net)
        
        # --- 3. 状态管理 ---
        self.bg_start_time = 0

        # --- 4. Cookie 与 配置 ---
        self.cookie_mask = 0xF000
        self.bg_cookie_start = bg_cookie
        self.agent_cookie_start = flow_cookie
        self.bg_duration = bg_duration
        
        # 任务状态
        self.s_node = None
        self.d_node = None
        self.current_flow_type = None
        self.step_count = 0 
        self.path_so_far = []
        self.current_node = None 
        self.dist_matrix = dict(nx.all_pairs_shortest_path_length(self.blueprint_G))
        self.shaping_weight = shaping_weight
    
    # 重新启动背景流量/重新发送业务流
    def reset_bg(self, current_load_mbps: float | None = None) -> Data:
        """
        Args:
            force_hard (bool): 强制执行 Hard Reset
            current_load_mbps (float): [课程学习] 指定当前的背景流量负载
        """
        target_load = current_load_mbps if current_load_mbps else self.total_load_mbps
        self._hard_reset_background_traffic(target_load)

        return self.get_observation()

    def reset_flow(self):
        mc.clean_flow_rules(self.net, cookie=self.agent_cookie_start, mask=self.cookie_mask)
        return self.get_observation()
    
    # 动作掩码
    def get_action_mask(self):
        """
        生成最终可用的动作掩码 (True 代表合法动作)
        统一处理所有掩码规则：
            1. 边的起点必须是当前节点
            2. 禁止走回头路（除非只有回头路可走）
            3. 排除已访问过的所有节点（不仅是前一个）
        """

        edge_index = self.observation_data.edge_index

        assert(edge_index is not None)
        num_edges = edge_index.shape[1]
        mask = torch.zeros(num_edges, dtype=torch.bool)
        
        # 获取所有边的起点和终点
        source_nodes = edge_index[0, :]
        target_nodes = edge_index[1, :]
        
        # 1. 基础规则：边的起点必须是当前节点
        is_current_node_edge = (source_nodes == self.current_node)
        
        # 2. 获取已访问节点集合
        visited_nodes = set(self.path_so_far)
        
        # 3. 生成掩码：只允许到未访问节点的边
        valid_edges = torch.zeros(num_edges, dtype=torch.bool)
        for i in range(num_edges):
            if is_current_node_edge[i]:
                target_node = target_nodes[i].item()
                if target_node not in visited_nodes:
                    valid_edges[i] = True
        
        # 4. 严格规则：只返回有效边，如果是死胡同则全为False
        mask = valid_edges
            
        return mask
    
    # 更新图状态，返回图状态
    def get_observation(self):
        # --- 1. 物理同步 ---
        self.current_G = self.monitor.sync_state_to_graph(duration=0.05)
        
        # --- 2. 特征提取 ---
        # Hop-by-Hop 关键：S_node 填 self.current_node
        assert isinstance(self.current_node, int), f"self.current_node 必须是整数：但是 {self.current_node}"
        assert(self.current_node is not None)
        assert(self.d_node is not None)

        data, _ = get_graph_data(
            self.current_G.copy(), 
            self.current_node, 
            self.d_node,      
            self.topo_stats,
            self.max_steps
        )
        
        self.observation_data = data
        data.action_mask = self.get_action_mask()

        return data
    
    # 结束环境
    def close(self):
        logger.info("Closing environment...")
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
    def _hard_reset_background_traffic(self, load_mbps: float):
        """
        环境重置：注入指定负载的背景流
        """
        logger.info(f">>> Hard Reset | Load: {load_mbps:.1f} Mbps <<<")
        
        # 1. 清理
        self._kill_bg_processes()
        mc.clean_flow_rules(self.net, cookie=self.bg_cookie_start, mask=self.cookie_mask)
        mc.clean_flow_rules(self.net, cookie=self.agent_cookie_start, mask=self.cookie_mask)
        self.active_cookies.clear()
        
        # 2. 生成 (使用传入的动态负载)
        
        # 确定生成背景流的节点集合
        bg_nodes = self.bg_traffic_nodes if self.bg_traffic_nodes is not None else list(self.blueprint_G.nodes())
        
        tm = self.flow_gen.generate_traffic_matrix(
            bg_nodes, 
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
    def step(
        self, action_edge_idx: int
    ) -> tuple[Data, float | Any, Literal[True], dict[str, Any]] | tuple[Any | Data, Any, bool, dict[str, Any]]:
        """
        执行一跳动作，返回下一个状态、奖励、是否结束、额外信息
        params:
            action_edge_idx: 动作索引，对应 edge_index 中的边

        Returns: 
            next_state: 下一个状态
            reward: 奖励
            done: 是否结束
            info: 额外信息
        """
        info: dict[str, Any] = {}

        # 1. 解析动作
        assert(self.observation_data.edge_index is not None)

        try:
            # 增加防越界保护
            u = self.observation_data.edge_index[0, action_edge_idx].item()
            v = self.observation_data.edge_index[1, action_edge_idx].item()
        except IndexError:
            logger.error("Action index out of bounds!")
            import traceback
            traceback.print_exc()
            exit()
        
        # 2. 合法性检查
        if u != self.current_node:
            logger.error(f"Illegal Move: {self.current_node} -> {u} impossible. Valid edges start from {self.current_node}")
            import traceback
            traceback.print_exc()
            exit()

        self.current_node = v
        self.path_so_far.append(v)
        self.step_count += 1
        
        done = False

        step_reward = 0
        
        assert(self.current_flow_type is not None)
        
        info['flow_type'] = self.current_flow_type.name

        # 4. 状态判定
        if v == self.d_node:
            # === Success ===
            done = True
        
            # 生成 Cookie
            cookie = self.agent_cookie_start + len(self.active_cookies) % 4096
            self.active_cookies.add(cookie)
    
            protocol_str = FLOW_PROFILES[self.current_flow_type]['protocol']
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
                qos_reward_config=self.qos_reward_config )
            
            logger.trace(f"QoS: {qos_reward:.4f} | QoE: {qoe_reward:.4f} | Path: {self.path_so_far}")
            # Add potential reward to final reward too? 
            # Usually terminal reward replaces step reward. 
            total_reward = qos_reward + step_reward 

            info['qos'] = qos_reward
            info['path'] = self.path_so_far

            return self.get_observation(), total_reward, True, info

        elif v in self.path_so_far[:-1]:
            # === Loop ===
            done = True
            step_reward += -2.0  # Reduced from -10.0
            info['error'] = 'loop_detected'

        elif self.step_count >= self.max_steps:
            # === Timeout ===
            done = True
            step_reward += -2.0 # Reduced from -10.0
            info['error'] = 'max_steps'

        # 5. 更新观察 (Observation)
        # 下一跳的 Observation 中，Source 特征可能需要变为 Current Node
        # 或者你需要让 GNN 知道 "Current Position" 在哪里
        next_state = self.get_observation()
        return next_state, step_reward, done, info

    def get_path_reward(
        self, s_node: int, d_node: int, path_route: list[int], flow_type: FlowType
    ):
        cookie = self.agent_cookie_start + len(self.active_cookies) % 4096
        self.active_cookies.add(cookie)

        # 打印路径状态 (用于调试)
        logger.debug(f"Path: {path_route}, FlowType: {flow_type.name.upper()}")
        
        protocol_str = FLOW_PROFILES[flow_type]['protocol']
        # A. 下发规则
        mc.install_path_rules(
            self.net, path_route, 
            tos=32, dst_port=12000, cookie=cookie, protocol=protocol_str)

        # B. 测量 QoS
        src_host = self.net.get(f'h{s_node}')
        dst_host = self.net.get(f'h{d_node}')
        
        qos_reward, qoe_reward = mc.measure_path_qos(
            server=dst_host,
            client=src_host,
            path_route=path_route,
            flow_type=flow_type,
            qos_reward_config=self.qos_reward_config )
        
        logger.debug(f"QoS: {qos_reward:.4f} | QoE: {qoe_reward:.4f} | Path: {path_route}")
        return qos_reward, qoe_reward 
