import networkx as nx
from functools import partial
from contextlib import contextmanager
from mininet.net import Mininet
from mininet.topo import Topo
from mininet.node import OVSKernelSwitch, RemoteController
from mininet.link import TCLink
from mininet.log import setLogLevel, info
from ...utils import logger

# Generator :
# 生成一个mininet网络

# mininet 定义
class GraphTopo(Topo):
    def __init__(self, blueprint_g: nx.Graph, is_test: bool = False, **opts):
        """
        初始化一个基于蓝图图的 Mininet 拓扑。
        params:
            blueprint_g: 蓝图图，节点属性包含 'bandwidth', 'delay', 'loss', 'queue_size'
            is_test: 是否为测试环境，若为 True 则在节点 ID 前添加 'T' 前缀
            opts: Topo 类的其他参数
        """
        Topo.__init__(self, **opts)
        test_str = "T" if is_test else ""

        for node_id in blueprint_g.nodes():
            self.addSwitch(f'{test_str}s{node_id}', protocols='OpenFlow13')
            self.addHost(f'{test_str}h{node_id}')
            self.addLink(f'{test_str}h{node_id}', f'{test_str}s{node_id}', delay='0ms')

        for u, v, data in blueprint_g.edges(data=True):
            bw = data.get('bandwidth', 30.0) # Mbps
            delay = f"{data.get('delay', 10)}ms"
            loss = data.get('loss', 0)
            
            if 'queue_size' in data:
                q_limit = data['queue_size']
            else:
                # 估算每秒包转发率 (pps)
                pps = (bw * 1000000) / (1500 * 8)
                # 设置 20ms 的缓冲
                q_limit = int(max(50, pps * 0.02))

            rate_bytes = bw * 1000000 / 8
            # 2. 计算最佳 r2q (确保 quantum ≈ 1500)
            r2q = int(max(1, rate_bytes / 1500))
            # 这里沿用 Mininet 构造函数中设置的 r2q
            logger.log(
                f"Link {u:>2} <-> {v:<2} | "
                f"BW: {bw:>5} Mbps | "
                f"Delay: {delay:>6} | "
                f"Loss: {loss:>3}% | "
                f"R2Q: {r2q:>5} | "
                f"QLimit: {q_limit:>4}", 
                tag="Mini Init"
            )
            self.addLink(
                f'{test_str}s{u}', f'{test_str}s{v}', 
                cls=TCLink, 
                bw=bw, 
                delay=delay, 
                loss=loss, 
                r2q=r2q, 
                use_htb=True, 
                max_queue_size=q_limit
            ) 

# mininet 启动
@contextmanager
def get_a_mininet(g: nx.Graph, is_test=False, remote_port=None):
    if remote_port:
        controller = partial(RemoteController, ip='127.0.0.1', port=remote_port)
    else:
        controller = None

    # if not vp.MININET_VERBOSE:
    # setLogLevel('critical')

    net = Mininet(
        topo=GraphTopo(g, is_test),
        switch=OVSKernelSwitch,
        link=TCLink,
        controller=controller,
        autoSetMacs=True, 
        autoStaticArp=True)

    try:
        logger.log("Disabling TCP Offload (TSO/GSO/GRO) on all switches...", tag="Mini Init")
        for h in net.hosts:
            for intf in h.intfList():
                if intf.name != 'lo':
                    # 使用 ethtool 关闭卸载
                    h.cmd(f"ethtool -K {intf.name} tso off gso off gro off > /dev/null 2>&1")
        for sw in net.switches:
            for intf in sw.intfList():
                if intf.name != 'lo':
                    sw.cmd(f"ethtool -K {intf.name} tso off gso off gro off > /dev/null 2>&1")
            
        net.start()
        yield net
    finally:
        logger.log("stopping mininet ...", tag="Mini Stop")
        net.stop()

    return net
