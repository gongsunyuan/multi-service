import torch
import networkx as nx
from collections import defaultdict
from time import sleep
from loguru import logger

# 根据gnn输出的 logits来生成路径--贪婪/概率选择：
# 替换 MS/Env/MininetController.py 中的 sample_path 函数
def sample_path(edge_logits, edge_index, s_node, d_node, max_steps=100, G_fallback=None, greedy=False):
    """
    增强版路径采样：支持 Masking (防环) + Dijkstra Fallback (兜底)
    返回: path, log_prob_sum, ai_success, path_complete
    """
    # 1. 构建邻接表
    adj = {}
    num_edges = edge_index.shape[1]
    for i in range(num_edges):
        u = edge_index[0, i].item()
        v = edge_index[1, i].item()
        if u not in adj: adj[u] = []
        adj[u].append((v, i)) 

    current = s_node
    path = [current]
    visited = {current}
    log_probs = [] 
  
    # ---------------------------------------------------------
    # A. AI 游走阶段
    # ---------------------------------------------------------
    for _ in range(max_steps):
        if current == d_node: break
        if current not in adj: break
    
        # Action Masking: 禁止走回头路
        neighbors = adj[current]
        valid_options = [n for n in neighbors if n[0] not in visited]
        if not valid_options: break # 死胡同
    
        # 提取 Logits
        candidate_logits = []
        candidate_nodes = []
        for next_node, edge_idx in valid_options:
            candidate_logits.append(edge_logits[edge_idx])
            candidate_nodes.append(next_node)
      
        logits_tensor = torch.stack(candidate_logits)
        if greedy:
            action_idx = torch.argmax(logits_tensor).item()
            log_prob = torch.tensor(0.0).to(logits_tensor.device)
        else:
            probs = torch.softmax(logits_tensor, dim=0)
            dist = torch.distributions.Categorical(probs)
            action_tensor = dist.sample()
            action_idx = action_tensor.item()
            log_prob = dist.log_prob(action_tensor)
    
        log_probs.append(log_prob)
    
        next_hop = candidate_nodes[int(action_idx)]
        path.append(next_hop)
        visited.add(next_hop)
        current = next_hop

    # ---------------------------------------------------------
    # B. 状态判定与兜底 (Safety Net) [核心修复]
    # ---------------------------------------------------------
    ai_success = False      # AI 是否独立完成
    path_complete = False   # 最终路径是否连通 (含兜底)

    if path[-1] == d_node:
        # Case 1: AI 独立成功
        ai_success = True
        path_complete = True
    else:
        # Case 2: AI 失败，尝试 Dijkstra 兜底
        if G_fallback is not None:
            try:
                # 从断点找最短路补全
                remaining_path = nx.shortest_path(G_fallback, source=path[-1], target=d_node, weight='delay')
                path.extend(remaining_path[1:]) # 拼接到路径后
        
                ai_success = False    # AI 没完成
                path_complete = True  # 但路通了
            except nx.NetworkXNoPath:
                # 物理隔离，彻底失败
                ai_success = False
                path_complete = False
        else:
            # 没有兜底机制，且 AI 没走通
            ai_success = False
            path_complete = False
  
    # 汇总 Log Probs
    if log_probs:
        log_prob_sum = torch.stack(log_probs).sum()
    else:
        log_prob_sum = torch.tensor(0.0).to(edge_logits.device)

    # 返回 4 个值：路径，梯度概率，AI是否成功，最终是否连通
    return path, log_prob_sum, ai_success, path_complete
  
def verify_cleanup(net, cookie=0xA000):
    """
    检查网络中是否还有残留的指定 cookie 的流表。
    返回: True (清理干净了), False (还有残留)
    """
    has_residue = False
  
    for sw in net.switches:
        # 获取该交换机的所有流表
        # 注意: grep 是 shell 命令，通过 sw.cmd 调用
        # result 包含查询结果字符串
        result = sw.cmd(f'ovs-ofctl -O OpenFlow13 dump-flows {sw.name} | grep "cookie={hex(cookie)}"')
    
        # 如果 result 不为空，说明找到了残留
        if result.strip():
            logger.info(f"Residue flows found on {sw.name}:\n{result.strip()}")
            has_residue = True
        
    if not has_residue:
        # logger.log(f"Flow cleanup verified. No rules with cookie={hex(cookie)} found.", tag="Clean OK")
        return True
    else:
        logger.error(f"Flow cleanup failed. Error cookie: {cookie}")
        return False

# 清除流表规则
def clean_flow_rules(net, cookie=0xA000, mask=0xF000):
    """
    清理流表。
    :param cookie: 要匹配的 Cookie 值 (前缀)
    :param mask: 掩码。0xF000 表示只匹配前 4 位 (即 0xA...)，忽略后面具体的步数。
    """
  
    # 构造匹配字符串: cookie=0xA000/0xF000
    # 这告诉 OVS: "只要 Cookie 以 A 开头 (高4位是A)，不管后面几位是什么，统统删掉！"
    match_str = f"cookie={hex(cookie)}/{hex(mask)}"
  
    # logger.log(f"Cleaning flows matching {match_str}...", tag="Rule Clean")

    for sw in net.switches:
        # 注意：这里去掉了 /-1，改用我们计算出的掩码
        cmd = f'ovs-ofctl -O OpenFlow13 del-flows {sw.name} "{match_str}"'
        sw.cmd(cmd)
      
    sleep(0.5) # 给 OVS 一点反应时间
    verify_cleanup(net, cookie)

# 下发流表规则
def install_path_rules(net, path_nodes, cookie, actual_sig_port=15000, tos=None, dst_port=None, protocol='UDP'):
    """
    安装端到端路径流表 (严格模式)
    140: 只放行 ARP、Ping 和 TCP 握手 (SYN)
    150: 放行信令数据和带 ToS 的业务数据
    100: 反向回包保底
    """
    # 1. 获取基础网络对象与 IP
    src_id, dst_id = path_nodes[0], path_nodes[-1]
    h_src, h_dst = net.get(f'h{src_id}'), net.get(f'h{dst_id}')
    src_ip, dst_ip = h_src.IP(), h_dst.IP()

    # [Optimization] 批量规则缓冲区
    batch_rules = defaultdict(list)

    for i, current_node_id in enumerate(path_nodes):
        sw_name = f's{current_node_id}'
        switch = net.get(sw_name)

        # ==========================================
        # A. 端口预计算 (必须先算好正向和反向，才能下流表)
        # ==========================================
    
        # 1. 正向输出端口 (Out Port)
        out_port = None
        if i == len(path_nodes) - 1:
            links = net.linksBetween(switch, h_dst)
            if links:
                link = links[0]
                out_intf = link.intf1 if link.intf1.node == switch else link.intf2
                out_port = switch.ports[out_intf]
        else:
            next_sw = net.get(f's{path_nodes[i+1]}')
            links = net.linksBetween(switch, next_sw)
            if links:
                link = links[0]
                out_intf = link.intf1 if link.intf1.node == switch else link.intf2
                out_port = switch.ports[out_intf]

        # 2. 反向输出端口 (Reverse Port)
        rev_port = None
        if i == 0:
            links_rev = net.linksBetween(switch, h_src)
            if links_rev:
                link_rev = links_rev[0]
                rev_intf = link_rev.intf1 if link_rev.intf1.node == switch else link_rev.intf2
                rev_port = switch.ports[rev_intf]
        else:
            prev_sw = net.get(f's{path_nodes[i-1]}')
            links_rev = net.linksBetween(switch, prev_sw)
            if links_rev:
                link_rev = links_rev[0]
                rev_intf = link_rev.intf1 if link_rev.intf1.node == switch else link_rev.intf2
                rev_port = switch.ports[rev_intf]

        if out_port is None or rev_port is None:
            continue # 容错处理

        # ==========================================
        # B. 收集流表 (批量模式)
        # ==========================================

        # [Priority 140: 基础设施层] 
    
        # [ARP 规则]: 允许二层地址解析。匹配: dl_type=0x0806 (ARP 协议)。
        batch_rules[sw_name].append(f"priority=140,dl_type=0x0806,actions=NORMAL")

        # [Ping 正向]: 匹配: dl_type=0x0800 (IP), nw_proto=1 (ICMP), 目标 IP。
        batch_rules[sw_name].append(f"cookie={cookie},priority=140,dl_type=0x0800,nw_proto=1,nw_dst={dst_ip},actions=output:{out_port}")
        # [Ping 反向]: 匹配: 目标 IP 为源主机。
        batch_rules[sw_name].append(f"cookie={cookie},priority=140,dl_type=0x0800,nw_proto=1,nw_dst={src_ip},actions=output:{rev_port}")

        # [TCP 握手正向]: 匹配: nw_proto=6 (TCP), tcp_flags=0x002/0x002 (仅 SYN 位为 1)。
        batch_rules[sw_name].append(f"cookie={cookie},priority=140,dl_type=0x0800,nw_proto=6,tcp_flags=0x002/0x002,nw_dst={dst_ip},actions=output:{out_port}")
        # [TCP 握手反向]: 匹配: 反向路径的 SYN/ACK。
        batch_rules[sw_name].append(f"cookie={cookie},priority=140,dl_type=0x0800,nw_proto=6,tcp_flags=0x002/0x002,nw_dst={src_ip},actions=output:{rev_port}")
        # tcp_flags=0x010/0x010 匹配 ACK 位。
        batch_rules[sw_name].append(f"cookie={cookie},priority=140,tcp,nw_dst={dst_ip},tcp_flags=0x010/0x010,actions=output:{out_port}")
        
        # [Priority 150: 业务通道层] 
        if tos is not None:
            # [信令通道 15000]: 匹配: TCP 协议, 目的端口 15000。
            batch_rules[sw_name].append(f"cookie={cookie},priority=150,dl_type=0x0800,nw_proto=6,tp_dst={actual_sig_port},nw_dst={dst_ip},actions=output:{out_port}")
            # [信令回包]: 匹配: 源端口 15000 (ACK)。
            batch_rules[sw_name].append(f"cookie={cookie},priority=150,dl_type=0x0800,nw_proto=6,tp_src={actual_sig_port},nw_dst={src_ip},actions=output:{rev_port}")

            # [VIP 数据流]: 匹配: 指定协议, 指定 ToS, 目标 IP。
            p_num = 6 if protocol.upper() == 'TCP' else 17
            match_data = f"cookie={cookie},priority=150,dl_type=0x0800,nw_proto={p_num},nw_tos={tos},nw_dst={dst_ip}"
            # 仅 UDP 精确匹配数据端口，TCP 建议放宽端口以兼容。
            if protocol.upper() == 'UDP' and dst_port:
                match_data += f",tp_dst={dst_port}"
      
            batch_rules[sw_name].append(f"{match_data},actions=output:{out_port}")

        # [Priority 100: 反向回包保底] 
    
        # [反向保底]: 匹配: 只要发往源主机的 IP 包。用于承载所有 ACK/Reply，优先级最低。
        batch_rules[sw_name].append(f"cookie={cookie},priority=100,dl_type=0x0800,nw_dst={src_ip},actions=output:{rev_port}")

    # [Optimization] 批量下发流表
    for sw_name, rules in batch_rules.items():
        switch = net.get(sw_name)
        if not rules: continue
        
        # 使用 printf + pipe 一次性下发所有规则
        # 注意: rules 中不能包含单引号，否则会破坏 printf 语法
        rules_str = "\\n".join(rules)
        cmd = f"printf '{rules_str}' | ovs-ofctl -O OpenFlow13 add-flows {sw_name} -"
        switch.cmd(cmd)
