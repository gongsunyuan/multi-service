from typing import Any, Generator
from torch_geometric.data.data import Data
import torch
import random
import numpy as np
import networkx as nx
from torch.utils.data import IterableDataset
from torch_geometric.data import Data, Batch
from .routing_kernels import RoutingKernels
from ..env.network_generator import TopologyGenerator
from ..utils import logger, AttrDict

class WarmupDataset(IterableDataset):
    def __init__(self, config: AttrDict, max_samples: int = 10000):
        super().__init__()
        self.config = config
        self.topo_gen = TopologyGenerator()
        self.max_samples = max_samples
        self.kernel = RoutingKernels()

    def _generate_sample(self) -> Data | None:
        # 1. 生成拓扑，增加拓扑大小范围（5-30个节点）
        min_nodes = 5
        max_nodes = 30
        G_nx = self.topo_gen.generate_topology(min_nodes=min_nodes, max_nodes=max_nodes)
        
        # 2. 注入多样化拥塞并计算混合权重
        edges = list(G_nx.edges())
        num_edges = len(edges)
        
        # 选择拥塞注入策略
        congestion_strategy = random.choice(['random', 'extreme', 'local', 'path'])
        
        # 初始化所有边的利用率
        for u, v in edges:
            cap = random.choice(['30', '120', '200'])  
            util = 0.0
            
            # 根据不同策略生成拥塞
            if congestion_strategy == 'random':
                # 随机拥塞分布
                util = random.random()
            elif congestion_strategy == 'extreme':
                # 极端拥塞策略：少量边极度拥塞，其他边正常
                if random.random() < 0.1:
                    util = random.uniform(0.9, 0.99)
                else:
                    util = random.uniform(0.0, 0.5)
            elif congestion_strategy == 'local':
                # 局部拥塞策略：随机选择一个中心节点，其周围边拥塞
                if num_edges > 0:
                    center_node = random.choice(list(G_nx.nodes()))
                    # 检查是否是中心节点的邻居边
                    if u == center_node or v == center_node:
                        util = random.uniform(0.7, 0.95)
                    else:
                        util = random.uniform(0.0, 0.4)
            elif congestion_strategy == 'path':
                # 路径拥塞策略：随机选择一条路径，使其拥塞
                if num_edges > 0 and G_nx.number_of_nodes() > 2:
                    # 随机选择两个不同的节点
                    path_nodes = random.sample(list(G_nx.nodes()), 2)
                    try:
                        # 找到一条路径
                        path = nx.shortest_path(G_nx, source=path_nodes[0], target=path_nodes[1])
                        # 检查这条边是否在路径上
                        if (u, v) in list(zip(path[:-1], path[1:])) or (v, u) in list(zip(path[:-1], path[1:])):
                            util = random.uniform(0.8, 0.99)
                        else:
                            util = random.uniform(0.0, 0.5)
                    except nx.NetworkXNoPath:
                        # 如果没有路径，退化为随机策略
                        util = random.random()
            
            # 计算混合权重
            w = self.kernel.calculate_hybrid_weight(util, penalty_factor=10.0)
            
            # 写入属性 (Input 特征需要 utilization，Path计算需要 weight)
            G_nx[u][v]['utilization'] = util
            G_nx[u][v]['bandwidth'] = cap
            G_nx[u][v]['hybrid_weight'] = w
            # 必须把 Delay 设为 0 或噪声，防止 GNN 依赖它，强迫 GNN 看 utilization
            G_nx[u][v]['delay'] = 0.0 + random.normalvariate(0, 0.01)  # 添加少量噪声
            G_nx[u][v]['loss'] = 0.0

        # 3. 随机选 Target
        target = random.choice(list(G_nx.nodes()))
        
        # 4. 获取指路标签
        # labels: {current_node_id: best_next_hop_id}
        next_hop_map = self.kernel.get_smart_path_labels(G_nx, target)
        
        if not next_hop_map: return None # pyright: ignore[reportReturnType]

        # 5. 转 PyG
        # D_node=target，这样 get_graph_data 会自动生成 Is_Dest 等特征
        from .networkx_watcher import get_graph_data
        data, _ = get_graph_data(G_nx, Cur_node=0, D_node=target, config=self.config)
        
        # 6. 构造监督信号：Edge Classification
        # 遍历所有边，如果这条边是 (u -> best_neighbor)，则 Label=1
        assert(data.edge_index is not None)

        y_guidance = torch.zeros(data.edge_index.shape[1])
        mask = torch.zeros(data.edge_index.shape[1], dtype=torch.bool)
        
        edge_list = data.edge_index.t().tolist()
        for i, (u, v) in enumerate(edge_list):
            if u in next_hop_map:
                if next_hop_map[u] == v:
                    y_guidance[i] = 1.0 # 正确的路
                    mask[i] = True
                else:
                    y_guidance[i] = 0.0 # 错误的路（或者是次优路）
                    mask[i] = True
            elif u == target:
                pass # 终点没有出边需要预测
        
        data.train_mask = mask
        data.target_node = target
        data.y_guidance = y_guidance
        
        return data

    def __iter__(self) -> Generator[Data, Any, None]:
        for _ in range(self.max_samples):
            data = self._generate_sample()
            while data is None:
                data = self._generate_sample()

            assert data is not None
            yield data