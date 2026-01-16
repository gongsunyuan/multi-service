import networkx as nx

class RoutingKernels:
    @staticmethod
    def calculate_hybrid_weight(delay: float, bandwidth: float, bw_priority: float=1.0) -> float:
        """
        在不考虑利用率的情况下，综合延迟和带宽计算链路代价。
        
        Args:
            delay: 链路物理延迟 (ms) 
            bandwidth: 链路物理容量 (Mbps) 
            bw_priority: 带宽优先级系数。值越大，模型越倾向于避开低带宽路径。
        """
        
        delay_cost = delay / 20  
        
        # 2. 带宽代价 (反比逻辑)
        # 代价与带宽成反比：100M 链路贡献 1.0 代价，10M 链路贡献 10.0 代价
        # 这能强迫 Dijkstra 避开窄带宽路，即使它延迟很低
        bw_cost = bw_priority / (bandwidth/10 + 1e-6)
        
        # 3. 组合代价
        # 返回两者的加权和
        return delay_cost + bw_cost

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