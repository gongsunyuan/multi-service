from loguru import logger
from omegaconf import DictConfig
import torch
import networkx as nx
from torch_geometric.data import Data


def get_graph_data(G: nx.Graph, Cur_node: int, D_node: int,
                   topo_stats: dict | DictConfig, max_steps: int):
    """
    get_graph_data 的 Docstring
    
    :param G: 要提取特征的图
    :param Cur_node: 当前节点 (Agent 所在位置)
    :param D_node: 目的节点
    :param topo_stats: 拓扑统计信息 (dict 或 DictConfig)
    :param max_steps: 最大寻路跳数

    returns: PyG Data 对象和带有特征的 NetworkX 图
    :rtype: tuple[Data, nx.Graph]
    
    特征维度说明：
        - 节点特征 (2维):
            - Is_Destination (0/1)
            - Is_Current_Node (0/1)
        - 边特征 (3维): 
            - Delay (归一化)
            - Bandwidth (归一化)
            - Loss (归一化)
    """

    # --- 2. 边特征处理 ---
    source_nodes, target_nodes = [], []
    edge_attrs_raw = []

    # 为了兼容 dict 和 DictConfig
    stats = topo_stats

    # 获取统计值 (处理 dict 和 DictConfig 的不同访问方式，这里假设它们都支持属性访问或都支持字典访问)
    # 实际上 OmegaConf 的 DictConfig 支持属性访问，而普通 dict 需要用 ['key']
    # 为了通用性，我们可以尝试转成 OmegaConf 或使用 getattr 风格，或者假设传入的是 DictConfig 兼容对象
    # 鉴于原始代码用的是属性访问 (config.topo.stats.delay.mu)，如果传入普通 dict，可能需要用 keys
    # 但最简单的方法是让调用者确保传入的是支持属性访问的对象，或者我们在内部处理
    # 为了保险，我们使用 helper 来获取值
    
    def get_stat(category, metric):
        if isinstance(stats, dict):
            return stats[category][metric]
        else:
            return getattr(getattr(stats, category), metric)

    for u, v, data in G.edges(data=True):
        source_nodes.extend([u, v])
        target_nodes.extend([v, u])

        l = float(data.get('loss', 0.0))
        d = float(data.get('delay', 1.0))
        b = float(data.get('bandwidth', 100.0))

        # [Change] 只保留 4 个核心特征
        attr = [d, b, l]
        edge_attrs_raw.extend([attr, attr])

    edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)

    # 转换为 Tensor
    if len(edge_attrs_raw) > 0:
        edge_attr_tensor = torch.tensor(edge_attrs_raw, dtype=torch.float)

        # 归一化
        edge_attr = torch.zeros_like(edge_attr_tensor)

        # [Fix] 索引范围修正为 0~3
        # 0: Delay
        delay_mu = get_stat('delay', 'mu')
        delay_sigma = get_stat('delay', 'sigma')
        edge_attr[:, 0] = (edge_attr_tensor[:, 0] - delay_mu) / (delay_sigma + 1e-6)

        # 1: Bandwidth
        bw_mu = get_stat('bw', 'mu')
        bw_sigma = get_stat('bw', 'sigma')
        edge_attr[:, 1] = (edge_attr_tensor[:, 1] - bw_mu) / (bw_sigma + 1e-6)

        # 2: Loss
        loss_mu = get_stat('loss', 'mu')
        loss_sigma = get_stat('loss', 'sigma')
        edge_attr[:, 2] = (edge_attr_tensor[:, 2] - loss_mu) / (loss_sigma + 1e-6)

        edge_attr = edge_attr.clamp(0.0, 1.0)
    else:
        # 防止空图报错
        edge_attr = torch.tensor([], dtype=torch.float)

    try:
        dist_to_d = nx.single_source_shortest_path_length(G, D_node)
    except:
        dist_to_d = {n: 999 for n in G.nodes()}

    node_features_list = []
    loggest_path_len = max_steps

    for u, data in G.nodes(data=True):

        # [Change] 这里的特征列表精简为 6 维
        is_c = 1.0 if u == Cur_node else 0.0
        is_d = 1.0 if u == D_node else 0.0
        dd = 1.0 * min(dist_to_d.get(u, 999),
                       loggest_path_len) / loggest_path_len

        # 基础特征 (2维)
        basic_feat = [is_d, is_c]

        node_features_list.append(basic_feat)

    x = torch.tensor(node_features_list, dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr), G
