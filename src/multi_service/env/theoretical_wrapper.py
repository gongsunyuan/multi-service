import torch
import networkx as nx
import time
from typing import Any, Literal
from loguru import logger
from omegaconf import DictConfig, ListConfig
from torch_geometric.data.data import Data

# 保持与原文件一致的导入结构
from .flow_generator import FLOW_PROFILES, FlowType
from ..utils import get_graph_data
from multi_service.env.qos.evaluator import calculate_qos_reward, calculate_qoe_reward
class TheoreticalWrapper:
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
        
        # 基础配置同步
        self.bg_traffic_nodes = bg_traffic_nodes
        self.total_load_mbps = total_load_mbps
        self.max_steps = max_steps
        self.topo_stats = train_stats
        self.train_stats = train_stats
        self.eval_stats = eval_stats
        self.qos_reward_config = qos_reward

        # 图状态初始化
        self.blueprint_G = topo
        self.current_G = self.blueprint_G.copy()
        self.src_nodes = src_nodes
        self.dst_nodes = dst_nodes

        # 任务状态（镜像原版）
        self.s_node = None
        self.d_node = None
        self.current_flow_type = None
        self.step_count = 0 
        self.path_so_far = [] 
        self.current_node = None 
        self.dist_matrix = dict(nx.all_pairs_shortest_path_length(self.blueprint_G))
        self.shaping_weight = shaping_weight
        
        # 流量带宽要求 (Mbps)
        self.flow_bandwidth = {
            FlowType.VOIP: 0.2,
            FlowType.STREAMING: 20.0,
            # FlowType.GAMING: 1.0
        }

        # 理论模式下不需要这些组件，但为了接口兼容保留占位
        self.active_cookies = set()
        self.agent_cookie_start = flow_cookie
        self.cookie_mask = 0xF000

    def reset_bg(self, current_load_mbps: float | None = None) -> Data:
        """
        理论模式下不考虑背景流，直接更新图状态并返回观察值
        """
        # 更新 current_G 可以在这里加入一些随机的链路波动模拟（如果需要）
        self.current_G = self.blueprint_G.copy()
        return self.get_observation()

    def reset_flow(self):
        """理论模式下无流规则清理，仅作为接口保留"""
        return self.get_observation()

    def get_action_mask(self):
        """逻辑与 SdnWrapper 完全一致"""
        edge_index = self.observation_data.edge_index
        assert(edge_index is not None)
        num_edges = edge_index.shape[1]
        
        source_nodes = edge_index[0, :]
        target_nodes = edge_index[1, :]
        is_current_node_edge = (source_nodes == self.current_node)
        visited_nodes = set(self.path_so_far)
        
        valid_edges = torch.zeros(num_edges, dtype=torch.bool)
        for i in range(num_edges):
            if is_current_node_edge[i]:
                target_node = target_nodes[i].item()
                if target_node not in visited_nodes:
                    valid_edges[i] = True
        return valid_edges

    def get_observation(self):
        """逻辑与 SdnWrapper 一致，但不再从 monitor 同步，直接使用内部 current_G"""
        assert isinstance(self.current_node, int)
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

    def close(self):
        logger.info("Closing Theoretical Environment...")

    def step(self, action_edge_idx: int):
        info: dict[str, Any] = {}
        assert(self.observation_data.edge_index is not None)

        # 1. 动作解析
        u = self.observation_data.edge_index[0, action_edge_idx].item()
        v = self.observation_data.edge_index[1, action_edge_idx].item()
        
        if u != self.current_node:
            raise ValueError(f"Illegal Move: {self.current_node} -> {u}")

        self.current_node = v
        self.path_so_far.append(v)
        self.step_count += 1
        
        done = False
        step_reward = 0
        info['flow_type'] = self.current_flow_type.name

        # 2. 状态判定
        if v == self.d_node:
            # 成功到达终点：计算理论 QoS
            done = True
            qos_reward, qoe_reward, qos_metrics = self.get_path_reward(
                self.s_node, self.d_node, self.path_so_far, self.current_flow_type
            )
            
            total_reward = qos_reward + step_reward 
            info['qos'] = qos_reward
            info['path'] = self.path_so_far
            info['qos_metrics'] = qos_metrics
            return self.get_observation(), total_reward, True, info

        elif v in self.path_so_far[:-1]:
            # 检测到环路
            done = True
            step_reward = -2.0
            info['error'] = 'loop_detected'

        elif self.step_count >= self.max_steps:
            # 步数超限
            done = True
            step_reward = -2.0
            info['error'] = 'max_steps'

        return self.get_observation(), step_reward, done, info

    def get_path_reward(
        self, s_node: int, d_node: int, path_route: list[int], flow_type: FlowType
    ):
        """
        核心替换：将原先 mc.measure_path_qos 的物理测量替换为基于拓扑属性的理论计算
        """
        # 1. 获取业务流带宽需求
        required_bw = self.flow_bandwidth.get(flow_type)
        
        # 2. 遍历路径链路，累加理论指标
        total_delay = 0.0
        bottleneck_bw = float('inf')
        total_loss_rate = 0.0 # 简化为独立丢包率累加：1 - (1-p1)(1-p2)...
        success_prob = 1.0

        for i in range(len(path_route) - 1):
            u, v = path_route[i], path_route[i+1]
            edge_data = self.current_G[u][v]
            
            # 理论计算逻辑
            # 延迟 = 链路固有延迟 + 传输延迟(PacketSize/Bandwidth)
            link_delay = edge_data.get('delay') # 默认1ms
            link_cap = edge_data.get('bandwidth') # 默认100Mbps
            link_loss = edge_data.get('loss', 0) # 默认0.1%丢包
            
            import random
            total_delay += link_delay + random.uniform(0, 1) # 加入少量随机扰动模拟测量误差
            bottleneck_bw = min(bottleneck_bw, link_cap)
            success_prob *= (1.0 - link_loss)

        total_loss = 1.0 - bottleneck_bw / (required_bw + random.uniform(0, 3)) if bottleneck_bw < required_bw else 0.0 # 加入少量随机扰动模拟测量误差

        qos_reward = calculate_qos_reward(
            delay_ms=total_delay,
            loss_percent=total_loss*100,
            jitter_ms=0.0, 
            flow_type_str=flow_type.name.upper(),
            qos_reward_config=self.qos_reward_config,
        )
        # if FlowType.STREAMING == flow_type:
        #     print(f"Path: {path_route}, Total Delay: {total_delay:.2f}ms, Bottleneck BW: {bottleneck_bw:.2f}Mbps, Loss Rate: {total_loss*100:.2f}%, QoS Reward: {qos_reward:.4f}")
        #     exit()
        qos_metric_dict = {
            'total_delay_ms': total_delay,
            'loss_rate_percent': total_loss*100
        }
        return float(qos_reward), float(0), qos_metric_dict