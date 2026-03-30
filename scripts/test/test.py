import sys
import os
import random
import time
from pathlib import Path
from typing import Any, Sequence

import hydra
from hydra.utils import instantiate
import numpy as np
import pandas as pd
import torch
import networkx as nx
from omegaconf import DictConfig, OmegaConf
from loguru import logger
from mininet.log import setLogLevel

# Add src to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent / "src"))

from multi_service.utils import PPOMemory, BankTrafficManager
from multi_service.env.sdn_wrapper import SdnWrapper
from multi_service.env.flow_generator import FlowType
from multi_service.env.network_generator import TopologyGenerator

class RoutingBaselines:
    @staticmethod
    def get_widest_path(topo: nx.Graph, src: int, dst: int):
        """
        最大瓶颈带宽路径 (Widest Path)
        逻辑：寻找一条路径，使其最小链路带宽最大化。
        这是对 Streaming 业务最公平的基准线。
        """
        # 1. 提取所有唯一的带宽值并从大到小排序
        all_bw = sorted(list(set([d['bandwidth'] for u, v, d in topo.edges(data=True)])), reverse=True)

        # 2. 二分查找或迭代搜索：只保留带宽 >= threshold 的边，看是否连通
        best_path = None
        for threshold in all_bw:
            # 创建临时子图，只包含带宽大于等于阈值的边
            temp_view = nx.subgraph_view(
                topo,
                filter_edge=lambda u, v: topo[u][v]['bandwidth'] >= threshold
            )

            if nx.has_path(temp_view, source=src, target=dst):
                # 在满足瓶颈要求的子图中，找一条跳数最短的路径（作为 tie-breaker）
                best_path = nx.shortest_path(temp_view, source=src, target=dst)
                # 找到满足最大瓶颈的最高阈值后立即停止
                break

        return best_path

    @staticmethod
    def get_spf_delay_path(topo: nx.Graph, src: int, dst: int):
        """
        基准 2: SPF-Delay (基于延迟)
        逻辑: 直接使用链路 delay 作为权重，倾向于走低延迟链路 (Zone B)
        """
        # 直接使用 graphml 中定义的 delay 属性 [cite: 4, 6]
        try:
            return nx.shortest_path(topo, source=src, target=dst, weight='delay')
        except nx.NetworkXNoPath:
            return None

def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    logger.info(f"Set random seed to {seed}")

def set_logger(config: DictConfig):
    setLogLevel('critical')
    logger.configure(handlers=[{
        "sink": config.path.trace_file,
        "level": "TRACE"
    }, {
        "sink": config.path.debug_file,
        "level": "DEBUG"
    }, {
        "sink": sys.stderr,
        "level": "INFO"
    }])

def build_workspace(config: DictConfig):
    for key, val in config.new_dir.items():
        new_dir = Path(val)
        new_dir.mkdir(parents=True, exist_ok=True, mode=0o777)
        OmegaConf.update(config, f"path.{key}", str(new_dir), force_add=True)

    for key, val in config.new_file.items():
        new_file = Path(val)
        new_file.touch(mode=0o777)
        OmegaConf.update(config, f"path.{key}", str(new_file), force_add=True)

def resolve_paths(config: DictConfig):
    """Resolve relative paths in config to absolute paths."""
    original_cwd = hydra.utils.get_original_cwd()

    # Resolve Topology Path
    topo_path = config.path.topo_path
    if not os.path.isabs(topo_path):
        topo_path = os.path.join(original_cwd, topo_path)
        # Update config with absolute path
        OmegaConf.update(config, "path.topo_path", topo_path, force_add=True)
        logger.info(f"Resolved topology path to {topo_path}")

    # Resolve Fingerprint Path
    if hasattr(config.path, 'fgpt_path') and not os.path.isabs(config.path.fgpt_path):
        abs_fgpt_path = os.path.join(original_cwd, config.path.fgpt_path)
        OmegaConf.update(config, "path.fgpt_path", abs_fgpt_path, force_add=True)
        logger.info(f"Resolved fingerprint path to {abs_fgpt_path}")

    # Resolve Checkpoint Paths if they are relative
    if hasattr(config.path, 'checkpoint_path'):
        for mode, path in config.path.checkpoint_path.items():
            if not os.path.isabs(path):
                abs_ckpt_path = os.path.join(original_cwd, path)
                OmegaConf.update(config, f"path.checkpoint_path.{mode}", abs_ckpt_path, force_add=True)
                logger.info(f"Resolved checkpoint path for {mode} to {abs_ckpt_path}")

    return topo_path

def load_topo(topo_path: Path) -> nx.Graph:
    # 显式指定 node_type=int（如果 graphml 的 ID 是纯数字字符串）
    G = nx.read_graphml(topo_path, node_type=int)

    # G = nx.convert_node_labels_to_integers(G)
    assert isinstance(G, nx.Graph), "Topology file is not a valid graphml file."
    logger.info(f"Successfully load topology from {topo_path}")
    logger.debug(f"Topology has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges:")
    for u, v, data in G.edges(data=True):
        logger.debug(f"    {u:<3} -> {v:<3} | {data['bandwidth']:>6.2f} Mbps | {data['delay']:>6.2f} ms")
    return G

def remove_edges(topo: nx.Graph, edges_to_remove: list) -> nx.Graph:
    """
    从拓扑图中删除指定的边，并记录操作结果。
    """
    actual_removed = []

    for u, v in edges_to_remove:
        # 统一转为字符串，因为 GraphML 导入的 ID 默认为字符串
        logger.debug(f"Removed edges: ")
        if topo.has_edge(u, v):
            # 获取边的属性（如 bw, delay），方便记录日志
            edge_data = topo.get_edge_data(u, v)
            topo.remove_edge(u, v)
            actual_removed.append((u, v))
            logger.debug(f"    {u:<3} -> {v:<3} | {edge_data['bandwidth']:>6.2f} Mbps | {edge_data['delay']:>6.2f} ms")
    logger.info(f"Removed {len(actual_removed)} edges")
    return topo

def get_topology_stats(G: nx.Graph, config: DictConfig):
    """通用的拓扑统计计算函数。"""
    if G.number_of_edges() == 0:
        return {
            'bw': {'mu': 0.0, 'sigma': 0.0},
            'delay': {'mu': 0.0, 'sigma': 0.0},
            'loss': {'mu': 0.0, 'sigma': 0.0}
        }

    # 提取属性
    bw_list = [float(d.get('bandwidth', 0)) for u, v, d in G.edges(data=True)]
    delay_list = [float(d.get('delay', 0)) for u, v, d in G.edges(data=True)]
    loss_list = [float(d.get('loss', 0)) for u, v, d in G.edges(data=True)]

    stats = {
        'bw': {
            'mu': float(np.mean(bw_list)),
            'sigma': float(np.std(bw_list))
        },
        'delay': {
            'mu': float(np.mean(delay_list)),
            'sigma': float(np.std(delay_list))
        },
        'loss': {
            'mu': float(np.mean(loss_list)),
            'sigma': float(np.std(loss_list))
        }
    }
    OmegaConf.update(config, "topo.stats", stats, force_add=True)

def load_weight(model: Any, config: DictConfig, mode: str):
    checkpoint_path = Path(config.path.weight_dir) / Path(f"{mode}.pth")
    try:
        checkpoint_info = torch.load(checkpoint_path, map_location=model.device)
        if 'model_state_dict' in checkpoint_info:
            model.load_state_dict(checkpoint_info['model_state_dict'])
            logger.info(f"Successfully load weight from `{checkpoint_path}`")
        else:
            logger.error(f"Checkpoint file `{checkpoint_path}` does not contain `model_state_dict`")
            logger.error(f"Keys found: {checkpoint_info.keys()}")
    except Exception as e:
        logger.error(f"Failed to load weight from `{checkpoint_path}`: {e}")

def initialize_components(config, topo):
    """Initialize environment, agent, and traffic manager."""

    # 1. Remove edges if specified
    if hasattr(config.env, 'edges_to_remove') and config.env.edges_to_remove:
        remove_edges(topo, config.env.edges_to_remove)

    # 2. Initialize Environment
    logger.info("Initializing SDN Environment...")
    env = instantiate(config.env.wrapper, topo=topo)

    # 3. Initialize Traffic Manager
    traffic_gen = instantiate(config.components.fgprt_bank)

    full = instantiate(config.components.full)
    gnn = instantiate(config.components.gnn)
    drl = instantiate(config.components.drl)
    # 4. Initialize Agents
    load_weight(full, config, "film")
    load_weight(gnn, config, "gnn")
    load_weight(drl, config, "drl")
    agents = {"full": full, "gnn": gnn, "drl": drl}

    return env, traffic_gen, agents

def init(config: DictConfig):
    # 1. Build Workspace
    build_workspace(config)

    # 2. Set Seed
    set_seed(config.seed)

    # 3. Set Logger
    set_logger(config)

    # 4. Resolve Paths
    topo_path = resolve_paths(config)

    # 5. Load Topology
    topo = load_topo(Path(topo_path))

    # 6. Get Topology Stats
    get_topology_stats(topo, config)

    # 7. Initialize Components
    env, traffic_gen, agents = initialize_components(config, topo)

    return topo, env, traffic_gen, agents

def get_agent_path(agent: Any, env: SdnWrapper, flow, mode: str):
    """
    逐跳决策生成完整路径，并与环境交互获得 QoS 奖励
    """
    agent.eval()
    # 1. 环境初始化：重置流状态并设置当前任务
    env.s_node = flow.src
    env.d_node = flow.dst
    env.current_node = flow.src
    env.current_flow_type = flow.flow_type
    env.path_so_far = [flow.src]
    env.step_count = 0

    # 每次测试流前清理旧流表，获取初始状态
    state = env.reset_flow()
    state = env.get_observation()

    fingerprint = flow.fingerprint.to(agent.device)

    if fingerprint.dim() == 2:
        fingerprint = fingerprint.unsqueeze(0)
    done = False
    total_reward = -1.0

    # 2. 逐跳决策循环
    while not done:
        state = state.to(agent.device)
        curr_node = env.current_node
        target_node = env.d_node
        logger.trace(
            f"Agent {mode} is now at {curr_node}, target is {target_node}")
        with torch.no_grad():
            # A. 提取当前节点嵌入
            node_embeds = agent.get_node_embeddings(state, fingerprint)

            # B. 获取当前节点的有效邻居特征
            edge_index = state.edge_index
            edge_attr = state.edge_attr
            assert edge_index is not None and edge_attr is not None

            mask = (edge_index[0] == curr_node)

            neighbor_indices = edge_index[1][mask].tolist()
            neighbor_edge_attrs = edge_attr[mask]
            # 获取环境提供的动作掩码 (防止环路)
            action_mask = state.action_mask[mask]
            # C. 模型推理
            next_node, _, _, action_idx = agent.get_action(
                state,
                node_embeds,
                action_mask,
                curr_node,
                target_node,
                neighbor_indices,
                neighbor_edge_attrs,
                deterministic=True  # 测试使用确定性策略
            )

            # D. 死胡同兜底处理
            if next_node is None:
                logger.warning(
                    f"Agent {agent.mode} stuck at {curr_node}, mission failed."
                )
                return -2.0, env.path_so_far

            # E. 映射回全局边索引并执行动作
            assert action_idx is not None
            selected_edge_idx = torch.where(mask)[0][int(action_idx)].item()
            state, reward, done, info = env.step(int(selected_edge_idx))

            # 在 SdnWrapper 中，最后一跳的 reward 即为 measure_path_qos 的结果
            if done:
                total_reward = reward

    return total_reward, info.get('path', []), info.get('qos_metrics', {})

def get_expert_reward(env: SdnWrapper, flow: Any, path: list[int] | None = None):
    """评估专家路径的 QoS 奖励"""
    if not path: return -1.0
    env.reset_flow()
    qos_reward, _, qos_metrics = env.get_path_reward(flow.src, flow.dst, path, flow.flow_type)
    return qos_reward, qos_metrics

@hydra.main(version_base=None, config_path="../../configs/test", config_name="test.yaml")
def test(config: DictConfig):
    # Use new init function
    topo, env, traffic_gen, agents = init(config)

    test_flows = traffic_gen.generate_batch(config.test.num_flows)

    # 用于收集数据的临时列表
    data_records = []

    # 3. 运行测试
    logger.info(f"Testing {len(test_flows)} flows for each model...")
    from tqdm import tqdm
    for i, flow in enumerate(tqdm(test_flows, desc="Flow Progress")):
        flow_type_str = flow.flow_type.name.lower()
        logger.debug(f">>>Flow {i} - {flow_type_str} - Src: {flow.src} - Dst: {flow.dst}<<<")
        for mode, agent in agents.items():
            logger.debug(f"*** Model {mode} ***")
            reward, path, qos_metrics = get_agent_path(agent, env, flow, mode)

            # 记录到列表，方便转为 DataFrame
            data_records.append({
                "Flow_ID": i,
                "Src": flow.src,
                "Dst": flow.dst,
                "Flow_Type": flow_type_str,
                "Model_Mode": mode,
                "Reward": reward,
                "Path_Length": len(path),
                "Delay_ms": qos_metrics.get('total_delay_ms', None),
                "Loss_percent": qos_metrics.get('loss_rate_percent', None)
            })

        # B. 测试专家路径 (Widest Path)
        logger.debug("*** Widest Path ***")
        widest_path = RoutingBaselines.get_widest_path(topo, flow.src, flow.dst)
        widest_reward, widest_qos_metrics = get_expert_reward(env, flow, widest_path)
        logger.debug(f"Widest Path: {widest_path} - Reward: {widest_reward}")
        data_records.append({
            "Flow_ID": i,
            "Src": flow.src,
            "Dst": flow.dst,
            "Flow_Type": flow_type_str,
            "Model_Mode": "Widest Path",
            "Reward": widest_reward,
            "Path_Length": len(widest_path) if widest_path else 0,
            "Delay_ms": widest_qos_metrics.get('total_delay_ms', None),
            "Loss_percent": widest_qos_metrics.get('loss_rate_percent', None)
        })

        # C. 测试专家路径 (SPF-Delay)
        logger.debug("*** SPF-Delay ***")
        delay_path = RoutingBaselines.get_spf_delay_path(topo, flow.src, flow.dst)
        delay_reward, delay_qos_metrics = get_expert_reward(env, flow, delay_path)
        logger.debug(f"SPF-Delay Path: {delay_path} - Reward: {delay_reward}")

        data_records.append({
            "Flow_ID": i,
            "Src": flow.src,
            "Dst": flow.dst,
            "Flow_Type": flow_type_str,
            "Model_Mode": "SPF-Delay",
            "Reward": delay_reward,
            "Path_Length": len(delay_path) if delay_path else 0,
            "Delay_ms": delay_qos_metrics.get('total_delay_ms', None),
            "Loss_percent": delay_qos_metrics.get('loss_rate_percent', None)
        })

    # 4. 转换与统计处理 [核心改进]
    results_df = pd.DataFrame(data_records)

    # 计算统计摘要
    summary = results_df.groupby(["Model_Mode", "Flow_Type"])[["Reward", "Delay_ms", "Loss_percent"]].agg(['mean', 'std', 'min', 'max']).reset_index()
    summary.to_csv(Path(config.path.eval_dir) / "robust_test_summary.csv")
    # 5. 保存结果
    output_path = Path(config.path.eval_dir) / "robust_test_results.csv"
    results_df.to_csv(output_path, index=False)
    logger.info(f"Full results saved to: {output_path}")

    env.close()

if __name__ == "__main__":
    test()
