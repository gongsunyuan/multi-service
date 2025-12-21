from mininet.net import Mininet   
from mininet.cli import CLI
from src.env.sdn_controller import send_packet_and_capture, get_a_mininet, install_path_rules, NetworkMonitor
from src.env.network_generator import TopologyGenerator
from src.env.flow_generator import(
  FlowGenerator
)
from src.utils import (
  SdnParaser, logger
)
from src.utils.verbose_logger import vprint
import networkx as nx

class Test_config:
  M_BA = 2
  MIN_BW = 5.0
  MAX_BW = 50.0
  MIN_DELAY = 1.0
  MAX_DELAY = 300.0
  MIN_NODES_NUM = 15
  MAX_NODES_NUM = 30
  
def function_test():

  topo_parser=SdnParaser()
  args = topo_parser.parse_args()

  flow_gen = TopologyGenerator()
  load_graph_path = "data/topologies/nsfnet_gravity.graphml"
  g = flow_gen.load_topology(load_graph_path)

  with get_a_mininet(g, remote_port=args.remote_port) as net: 
    monitor = NetworkMonitor(net)
    tm_gen = FlowGenerator()
    tm_dict = tm_gen.generate_traffic_matrix(g.nodes, g.copy(), 10.00)
    tm_gen.apply_traffic_matrix_to_mininet(net, tm_dict, g.copy(), install_rules_func=install_path_rules)
    for _ in range(3):
      cur_G = monitor.sync_state_to_graph(g.copy())
    logger.log_network_status(cur_G.copy())
    for u, v in g.edges():
      print(f"[{u}, {v}] bandwidth={g[u][v]['bandwidth']} delay={g[u][v]['delay']}")
    install_path_rules(net, [0, 1], cookie=0xA000)
    CLI(net)

if __name__ == '__main__':
  function_test()