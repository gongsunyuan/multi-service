import networkx as nx

class RoutingKernels:
    @staticmethod
    def calculate_hybrid_weight(utilization: float, penalty_factor: float = 5.0) -> float:
        """
        计算混合权重：兼顾跳数代价(1.0)和拥塞代价(Penalty)
        Utilization越高，权重呈指数级爆炸，迫使算法绕路。
        """
        base_cost = 1.0 # 基础跳数代价
        
        # 简单的非线性惩罚
        if utilization < 0.8:
            congestion_cost = 0.0
        else:
            # 超过 80% 负载，代价开始飙升
            # 例如 0.9 -> 0.5 * 5 = 2.5
            # 例如 0.99 -> 0.95 * 5 = 4.75
            # 你可以调大 penalty_factor 来让 Agent 更怕堵
            congestion_cost = (utilization - 0.8) * 5.0 * penalty_factor
            
        return base_cost + congestion_cost

    @staticmethod
    def get_smart_path_labels(G_nx: nx.Graph, target_node: str):
        """
        基于混合权重计算全图导航标签
        """
        # 1. 使用 Dijkstra 计算到 Target 的加权最短路
        # 注意：这里的 weight 必须是我们刚才注入的 'hybrid_weight'
        try:
            weighted_dist = nx.single_source_dijkstra_path_length(
                G_nx, target_node, weight='hybrid_weight'
            )
        except:
            return {}, {}

        # 2. 生成动作概率标签 (Softmax Target)
        # 如果 v 是 u 的最优下一跳，则 prob=1，否则=0
        next_hop_labels = {} 
        
        for u in G_nx.nodes():
            if u == target_node or u not in weighted_dist:
                continue
            
            best_neighbor = None
            min_cost = float('inf')
            
            # 遍历邻居找最优解
            # 逻辑：Dist(u) = Weight(u,v) + Dist(v)
            for v in G_nx.neighbors(u):
                if v not in weighted_dist: continue
                
                # 获取边权重
                w = G_nx[u][v]['hybrid_weight']
                path_cost = w + weighted_dist[v]
                
                if path_cost < min_cost:
                    min_cost = path_cost
                    best_neighbor = v
            
            # 记录最优下一跳 (这里做简单的 One-hot，也可以做 Multi-hot)
            if best_neighbor is not None:
                next_hop_labels[u] = best_neighbor
                
        return next_hop_labels