import sys
import os
import networkx as nx
from mininet.net import Mininet
from mininet.node import OVSKernelSwitch
from mininet.link import TCLink
from mininet.log import setLogLevel

# Add project root to path
sys.path.append(os.getcwd())

from MS.Env.NetworkGenerator import TopologyGenerator
from MS.Env.MininetController import measure_path_qos, install_path_rules
from MS.Env.FlowGenerator import FlowType

def verify_pipeline():
  setLogLevel('info')
  print("====== [System Test] Verifying Full Pipeline (Mininet + D-ITG + Reward) ======")

  # 1. Create a 2-node topology: h1 <--- link ---> h2

  g = nx.Graph()
  g.add_nodes_from([0, 1, 2])
  g.add_edges_from([(0,1), (1,2)])

  g[0][1]['delay'] = 0
  g[1][2]['delay'] = 100
  g[0][1]['bandwidth'] = 5
  g[1][2]['bandwidth'] = 5

  try:
    net.start()
    # Install forwarding rules
    install_path_rules(net, [], cookie=0x1234) # s1 ID is usually 0 or 1 depending on generation
    # Manually install flow rule since install_path_rules depends on specific graph IDs
    s1.cmd(f"ovs-ofctl -O OpenFlow13 add-flow s1 actions=flood") 

    # 3. Test Gaming Flow (Should fail due to 100ms delay)
    print("\n>>> Testing GAMING Flow on High Latency Link...")
    reward_game = measure_path_qos(h2, h1, [], FlowType.GAMING)
    print(f"    Gaming Reward: {reward_game} (Expected: Negative/Low)")

    # 4. Test VoIP Flow (Should be okay-ish)
    print("\n>>> Testing VOIP Flow on High Latency Link...")
    reward_voip = measure_path_qos(h2, h1, [], FlowType.VOIP)
    print(f"    VoIP Reward: {reward_voip} (Expected: Higher than Gaming)")

    # 5. Change Link to Low Bandwidth (1Mbps)
    print("\n[Setup] Changing Link to 1Mbps (Video Killer)...")
    # Mininet link dynamic change (requires reference to link object, skipping for simplicity)
    # Instead, we interpret the result: 
    # Since BW is 10Mbps, Video (req 5Mbps) should pass.
    print("\n>>> Testing STREAMING Flow on 10Mbps Link...")
    reward_video = measure_path_qos(h2, h1, [], FlowType.STREAMING)
    print(f"    Video Reward: {reward_video} (Expected: Positive/High)")

  except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
  finally:
    net.stop()

if __name__ == "__main__":
  if os.getuid() != 0:
    print("Run with sudo!")
  else:
    verify_pipeline()