from mininet.net import Mininet   
from mininet.cli import CLI
from MS.Env.MininetController import send_packet_and_capture, get_a_mininet
from MS.Env.NetworkGenerator import TopologyGenerator
from MS.Env.ConstructParaser import TopoParaser
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
  
  topo_parser=TopoParaser()
  args = topo_parser.parse_args()

  N_PACKETS_TO_CAPTURE = 30
  FLOW_DURATION = 30 
  flow_gen = TopologyGenerator()
  g = flow_gen.generate_topology()

  with get_a_mininet(g, is_test=True, remote_port=args.remote_port) as net: 
    CLI(net)

if __name__ == '__main__':
  function_test()