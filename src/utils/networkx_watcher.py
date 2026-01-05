import torch
import networkx as nx
from torch_geometric.data import Data

def get_graph_data(G: nx.Graph, Cur_node: int, D_node: int, config):
    
    """
    get_graph_data 的 Docstring
    
    :param G: 要提取特征的图
    :param Cur_node: 当前节点 (Agent 所在位置)
    :param D_node: 目的节点
    :param config: 配置对象

    returns: PyG Data 对象和带有特征的 NetworkX 图
    :rtype: tuple[Data, nx.Graph]
    
    特征维度说明：
        - 节点特征 (9维):
        0. Degree (归一化)
        1. Is_Destination (0/1)
        2. Dist_To_Destination (归一化)
        3. Betweenness Centrality
        4. Clustering Coefficient
        5. PageRank
        6. Buffer Occupancy 
        7. Processing Delay
        8. Is_Current_Node
        - 边特征 (4维): 
        0. Delay (归一化)
        1. Bandwidth (归一化)
        2. Loss (归一化)
        3. Utilization (直接使用 0-1)
    """
    
    # --- 1. 结构特征缓存 (保持不变) ---
    try: 
        pagerank = G.graph['pagerank']
        clustering = G.graph['clustering']
        betweenness = G.graph['betweenness']
    except:
        betweenness = nx.betweenness_centrality(G)
        pagerank = nx.pagerank(G, alpha=0.85)
        clustering = nx.clustering(G)

    # --- 2. 边特征处理 (移除 Avail BW) ---
    source_nodes, target_nodes = [], []
    edge_attrs_raw = []
    
    for u, v, data in G.edges(data=True):
        source_nodes.extend([u, v])
        target_nodes.extend([v, u])
        
        l = float(data.get('loss', 0.0))
        d = float(data.get('delay', 1.0))
        b = float(data.get('bandwidth', 100.0))
        u_load = float(data.get('utilization', 0.0))

        # [Change] 只保留 4 个核心特征
        attr = [d, b, l, u_load]
        edge_attrs_raw.extend([attr, attr])

    edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long) 
    
    # 转换为 Tensor
    if len(edge_attrs_raw) > 0:
        edge_attr_tensor = torch.tensor(edge_attrs_raw, dtype=torch.float)
        
        # 归一化
        edge_attr = torch.zeros_like(edge_attr_tensor)
        
        # [Fix] 索引范围修正为 0~3
        # 0: Delay
        edge_attr[:, 0] = (edge_attr_tensor[:, 0] - config.env.min_delay) / (config.env.max_delay - config.env.min_delay + 1e-6)
        # 1: Bandwidth
        edge_attr[:, 1] = (edge_attr_tensor[:, 1] - config.env.min_bw) / (config.env.max_bw - config.env.min_bw + 1e-6)
        # 2: Loss
        edge_attr[:, 2] = edge_attr_tensor[:, 2] / config.env.max_loss # Loss (假设最大5%)
        # 3: Utilization (本身就是 0-1，不需要额外归一化，Clamp一下即可)
        edge_attr[:, 3] = edge_attr_tensor[:, 3].clamp(0.0, 1.0) 
        
        # [Removed] edge_attr[:, 4] = Avail_BW (已删除)
        
        edge_attr = edge_attr.clamp(0.0, 1.0)
    else:
        # 防止空图报错
        edge_attr = torch.tensor([], dtype=torch.float)

    # --- 3. 节点特征处理 (移除 Source 相关) ---
    # [Removed] dist_from_s 计算块已删除

    # 只保留 Dist_To_Destination
    try:
        dist_to_d = nx.single_source_shortest_path_length(G, D_node)
    except:
        dist_to_d = {n: 999 for n in G.nodes()}

    num_nodes = G.number_of_nodes()
    node_features_list = []
    deg_max = config.env.max_nodes_num

    for i in range(num_nodes):
        node_data = G.nodes[i]
        
        # [Change] 这里的特征列表精简为 8 维
        is_c = 1.0 if i == Cur_node else 0.0
        is_d = 1.0 if i == D_node else 0.0
        deg = G.degree(i) / (deg_max + 1e-6) # type: ignore
        dd = 1.0 * min(dist_to_d.get(i, 999), deg_max) / deg_max
        
        assert isinstance(clustering, dict), f"clustering is not dict: {clustering}"
        
        betw = betweenness.get(i, 0.0)
        clus = clustering.get(i, 0.0)
        pr = pagerank.get(i, 0.0) * 10.0
        
        bo = float(node_data.get('buffer_occupancy', 0.0))
        pd = float(node_data.get('proc_delay', 0.0))

        # 基础特征 (8维)
        # [Removed] is_s, ds (dist_from_src)
        basic_feat = [deg, is_d, dd, is_c, betw, clus, pr, bo, pd]
        
        node_features_list.append(basic_feat)
        
    x = torch.tensor(node_features_list, dtype=torch.float)
    
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr), G

