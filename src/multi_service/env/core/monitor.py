import re
import networkx as nx
from time import time, sleep
from loguru import logger

# 监视器，负责动态提取mininet拓扑状态
class NetworkMonitor:

    def __init__(self, net):
        self.net = net

    def _read_sys_file(self, path):
        """快速读取系统文件"""
        try:
            with open(path, 'r') as f:
                return int(f.read().strip())
        except:
            return 0

    def _get_queue_limit(self, node_name, intf_name):
        """
        动态读取接口的队列上限 (Capacity)
        返回: limit (int, 单位: packets)
        """
        try:
            node = self.net.get(node_name)
            # -d 参数用于显示详细配置 (details)，包含 limit
            cmd = f"tc -d qdisc show dev {intf_name}"
            output = node.cmd(cmd)

            # 1. 尝试匹配 'limit 100p' (pfifo/bfifo_fast 等)
            # 这里的 p 代表 packets
            match_p = re.search(r'limit\s+(\d+)p', output)
            if match_p:
                return int(match_p.group(1))

            # 2. 尝试匹配 'limit 10000b' (bfifo，基于字节)
            # 如果是字节，我们需要估算包数。假设平均包大小 1500B
            match_b = re.search(r'limit\s+(\d+)b', output)
            if match_b:
                bytes_limit = int(match_b.group(1))
                return max(1, bytes_limit // 1500)

            # 3. 如果没找到 limit，通常是默认的 txqueuelen (通常是 1000)
            # 可以读取 /sys/class/net/.../tx_queue_len
            cmd_tx = f"cat /sys/class/net/{intf_name}/tx_queue_len"
            tx_len = int(node.cmd(cmd_tx).strip())
            return tx_len

        except Exception as e:
            logger.error(f"Error reading queue limit: {e}")
            return 100  # 兜底默认值

    def _get_all_interfaces_stats(self, G):
        """
        获取图中所有涉及接口的 tx_bytes 和 backlog
        返回: dict { (u, v): {'bytes': int, 'qlen': int} }
        """
        stats = {}
        for u, v in G.edges():
            # 映射节点名
            s_u_name = f"Ts{u}" if f"Ts{u}" in self.net else f"s{u}"
            s_v_name = f"Ts{v}" if f"Ts{v}" in self.net else f"s{v}"

            if s_u_name not in self.net: continue
            s_u = self.net.get(s_u_name)
            s_v = self.net.get(s_v_name)

            # 查找接口
            links = self.net.linksBetween(s_u, s_v)
            if not links: continue
            link = links[0]
            intf = link.intf1 if link.intf1.node == s_u else link.intf2

            # 1. 读取 Bytes (直接读宿主机文件，假设是在 Root Namespace 或路径正确)
            # 注意: Mininet 的虚拟接口通常在 Root NS 可见 (/sys/class/net/s1-eth2/...)
            # 如果读不到，可能需要用 node.cmd('cat ...')，但那样太慢。
            # 这里假设是标准 Mininet OVS 环境，接口都在 Root NS。
            tx_bytes = self._read_sys_file(
                f"/sys/class/net/{intf.name}/statistics/tx_bytes")

            # 2. 读取队列 (只能通过 tc 命令，较慢，但在 sampling 期间只读一次也行)
            # 为了速度，这里我们可以只在采样结束时读一次队列
            # 这里先存名字，稍后处理
            stats[(u, v)] = {'bytes': tx_bytes, 'intf': intf.name, 'node': s_u}

        return stats

    def _batch_get_qdisc_stats(self, nodes):
        """
        批量获取节点的 TC 统计信息，减少 shell 调用次数。
        返回: {node_name: {intf_name: q_len_packets}}
        """
        results = {}
        for node in nodes:
            results[node.name] = {}
            try:
                # 一次性获取该节点所有接口的 qdisc 信息
                output = node.cmd("tc -s qdisc show")

                # 解析输出
                current_intf = None
                for line in output.split('\n'):
                    line = line.strip()
                    # 匹配 qdisc 行获取接口名
                    match_dev = re.search(r'qdisc \w+ .* dev ([^\s]+) ', line)
                    if match_dev:
                        current_intf = match_dev.group(1)
                        continue

                    # 匹配 backlog 行获取队列长度
                    if current_intf:
                        match_backlog = re.search(r'backlog\s+\d+b\s+(\d+)p',
                                                  line)
                        if match_backlog:
                            results[node.name][current_intf] = int(
                                match_backlog.group(1))
            except Exception as e:
                logger.error(f"Batch TC error on {node.name}: {e}")
        return results

    def get_topology(self):
        """
        获取当前 Mininet 的实时拓扑结构和链路属性。
        只包含交换机，名字转换为整型 (e.g. 's1' -> 1)。
        链路属性包含: bw, delay, loss。
        """
        G = nx.Graph()

        # 1. 添加交换机节点
        for sw in self.net.switches:
            try:
                # 提取数字ID
                match = re.search(r'\d+', sw.name)
                if match:
                    sw_id = int(match.group())
                    G.add_node(sw_id, label=sw.name)
            except Exception as e:
                logger.warning(f"Error parsing switch name {sw.name}: {e}")
                continue

        # 2. 添加链路
        for link in self.net.links:
            node1 = link.intf1.node
            node2 = link.intf2.node

            # 仅处理交换机之间的链路
            if node1 not in self.net.switches or node2 not in self.net.switches:
                continue

            try:
                u_match = re.search(r'\d+', node1.name)
                v_match = re.search(r'\d+', node2.name)

                if not u_match or not v_match:
                    continue

                u = int(u_match.group())
                v = int(v_match.group())
            except Exception:
                continue

            # 获取链路参数 (优先取 intf1)
            # Mininet 的 TCLink 会将参数存储在 intf.params 中
            params = link.intf1.params

            bw = params.get('bw', 0)
            loss = params.get('loss', 0)
            delay_str = params.get('delay', '0ms')

            # 处理 delay 字符串 (e.g., '5ms' -> 5)
            # 如果是数字则直接使用
            delay = 0.0
            if isinstance(delay_str, str):
                match = re.search(r'([\d\.]+)', delay_str)
                if match:
                    delay = float(match.group(1))
            elif isinstance(delay_str, (int, float)):
                delay = float(delay_str)

            # 添加边及属性 (capacity 对应 bw)
            G.add_edge(u, v, bw=bw, delay=delay, loss=loss, capacity=bw)

        # Debug: 打印拓扑信息
        if G.number_of_edges() > 0:
            # 表头
            header = f"{'Link':<15} | {'BW (Mbps)':<12} | {'Delay (ms)':<12} | {'Loss (%)':<10}"
            sep = "-" * len(header)

            log_lines = ["\nTopology Links:"]
            log_lines.append(sep)
            log_lines.append(header)
            log_lines.append(sep)

            for u, v, data in sorted(G.edges(data=True)):
                link_name = f"s{u}-s{v}"
                bw = data.get('bw', 0)
                delay = data.get('delay', 0)
                loss = data.get('loss', 0)

                row = f"{link_name:<15} | {bw:<12} | {delay:<12} | {loss:<10}"
                log_lines.append(row)

            log_lines.append(sep)
            logger.trace("\n".join(log_lines))

        return G

    def sync_state_to_graph(self, duration=0.05):
        """
        同步网络状态到拓扑图 (Active Sampling)

        采用主动采样模式，通过两次快照计算链路的瞬时速率和利用率，
        并批量获取队列状态更新节点负载信息。

        Args:
            G (nx.Graph): NetworkX 拓扑图对象，边属性需包含 'capacity' (Mbps)。
            duration (float): 采样窗口时长 (秒)，默认为 0.05s。
                              该时间段内的流量差值用于计算速率。

        Returns:
            nx.Graph: 更新状态后的拓扑图。
                - Edge attributes updated: 'utilization', 'measured_speed'
                - Node attributes updated: 'buffer_occupancy', 'proc_delay'
        """
        G = self.get_topology()
        SAMPLE_WINDOW = duration  # 采样窗口 50ms

        # 1. 第一次快照
        snapshot1 = self._get_all_interfaces_stats(G)
        t1 = time()
        # 2. 等待
        sleep(SAMPLE_WINDOW)
        # 3. 第二次快照
        snapshot2 = self._get_all_interfaces_stats(G)
        t2 = time()

        delta_t = t2 - t1
        if delta_t <= 0: delta_t = 1e-6

        # [Optimization] 批量获取队列状态
        unique_nodes = set()
        for v in snapshot2.values():
            unique_nodes.add(v['node'])
        batch_q_stats = self._batch_get_qdisc_stats(list(unique_nodes))

        node_stats_buffer = {n: [] for n in G.nodes()}
        node_stats_util = {n: [] for n in G.nodes()}

        # 4. 计算并更新
        for u, v, data in G.edges(data=True):
            key = (u, v)
            if key not in snapshot1 or key not in snapshot2:
                continue

            # --- A. 计算瞬时利用率 ---
            b1 = snapshot1[key]['bytes']
            b2 = snapshot2[key]['bytes']

            # 速率 (Mbps)
            speed_mbps = ((b2 - b1) * 8) / (delta_t * 1_000_000)

            capacity = data.get('capacity', 30.0)
            assert (capacity >= 10.0 and capacity <= 100.0)
            util = min(speed_mbps / (capacity + 1e-6), 1.0)

            # 写入图
            # 更新 (平滑一点)

            data['utilization'] = 0.3 * data.get('utilization',
                                                 util) + 0.7 * util
            data['measured_speed'] = speed_mbps

            # --- B. 获取瞬时队列 (Buffer) ---
            # 队列长度只需要读一次（取最新状态）
            node = snapshot2[key]['node']
            intf_name = snapshot2[key]['intf']

            q_len = 0
            if node.name in batch_q_stats and intf_name in batch_q_stats[
                    node.name]:
                q_len = batch_q_stats[node.name][intf_name]

            max_q = 2000  # 假设最大队列长度为 2000 packets
            buf_occ = min(q_len / max_q, 1.0)
            if u in node_stats_buffer:
                node_stats_buffer[u].append(buf_occ)
                node_stats_util[u].append(util)

        # 写入节点特征 (源节点 Buffer)
        # 更新源节点 u 的状态
        for n in G.nodes():
            # --- 聚合 Buffer (取最大值) ---
            # 含义：如果有一个方向堵了，这个节点就标红
            buffer_list = node_stats_buffer[n]
            if buffer_list:
                max_buffer = max(buffer_list)
            else:
                max_buffer = 0.0

            util_list = node_stats_util[n]
            if util_list:
                max_util = max(util_list)
            else:
                max_util = 0.0

            # 简单估算 Proc Delay
            proc_delay = 0.01 * (1 + 5 * max_util**2)  # 拥塞时 CPU 处理变慢

            G.nodes[n]['buffer_occupancy'] = 0.3 * G.nodes[n].get(
                'buffer_occupancy', 0) + 0.7 * max_buffer
            G.nodes[n]['proc_delay'] = 0.3 * G.nodes[n].get(
                'proc_delay', 0) + 0.7 * proc_delay

        return G
