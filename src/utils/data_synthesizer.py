import torch
import random
import numpy as np
import networkx as nx
from torch.utils.data import IterableDataset
from torch_geometric.data import Data, Batch
from .routing_kernels import RoutingKernels
from ..env.network_generator import TopologyGenerator, get_pyg_data_from_nx
from ..utils import logger, AttrDict

class WarmupDataset(IterableDataset):
    def __init__(self, config: AttrDict, max_samples: int = 10000):
        super().__init__()
        self.config = config
        self.topo_gen = TopologyGenerator()
        self.max_samples = max_samples
        self.kernel = RoutingKernels()

    def _generate_sample(self) -> Data:
        # 1. 生成拓扑
        G_nx = self.topo_gen.generate_topology(min_nodes=10, max_nodes=20)
        
        # 2. 注入随机拥塞并计算混合权重
        for u, v in G_nx.edges():
            cap = 100 # 假设带宽统一，简化问题，只看利用率
            util = random.random() # 0.0 - 1.0
            
            # 故意制造一些极端拥塞来训练避障
            if random.random() < 0.2:
                util = random.uniform(0.9, 0.99)
            
            # 计算混合权重
            w = self.kernel.calculate_hybrid_weight(util, penalty_factor=10.0)
            
            # 写入属性 (Input 特征需要 utilization，Path计算需要 weight)
            G_nx[u][v]['utilization'] = util
            G_nx[u][v]['bandwidth'] = cap
            G_nx[u][v]['hybrid_weight'] = w
            # 必须把 Delay 设为 0 或噪声，防止 GNN 依赖它，强迫 GNN 看 utilization
            G_nx[u][v]['delay'] = 0.0 
            G_nx[u][v]['loss'] = 0.0

        # 3. 随机选 Target
        target = random.choice(list(G_nx.nodes()))
        
        # 4. 获取指路标签
        # labels: {current_node_id: best_next_hop_id}
        next_hop_map = self.kernel.get_smart_path_labels(G_nx, target)
        
        if not next_hop_map: return None # pyright: ignore[reportReturnType]

        # 5. 转 PyG
        # D_node=target，这样 get_pyg 会自动生成 Is_Dest 等特征
        data, _ = get_pyg_data_from_nx(G_nx, Cur_node=0, D_node=target, config=self.config)
        
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
        
        data.y_guidance = y_guidance
        data.train_mask = mask
        data.target_node = target
        
        return data

    def __iter__(self):
        for _ in range(self.max_samples):
            data = self._generate_sample()
            if data is not None:
                yield data