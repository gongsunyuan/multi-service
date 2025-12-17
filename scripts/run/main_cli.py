from mininet.net import Mininet   
from mininet.cli import CLI
from env.sdn_controller import send_packet_and_capture, get_a_mininet, install_path_rules
from env.network_generator import TopologyGenerator
from utils.sdn_paraser import TopoParaser
from utils import verbose_logger as vp
import networkx as nx

vprint = vp.vprint

class Test_config:
  M_BA = 2
  MIN_BW = 5.0
  MAX_BW = 50.0
  MIN_DELAY = 1.0
  MAX_DELAY = 300.0
  MIN_NODES_NUM = 15
  MAX_NODES_NUM = 30
  
def function_test():

  vp.MININET_VERBOSE = True
  topo_parser=TopoParaser()
  args = topo_parser.parse_args()

  flow_gen = TopologyGenerator()
  load_graph_path = "nsfnet.graphml"
  g = flow_gen.load_topology(load_graph_path)

  with get_a_mininet(g, remote_port=args.remote_port) as net: 
    for u, v in g.edges():
      print(f"[{u}, {v}] bandwidth={g[u][v]['bandwidth']} delay={g[u][v]['delay']}")
    install_path_rules(net, [0, 1])
    CLI(net)

if __name__ == '__main__':
  function_test()