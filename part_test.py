from mininet.net import Mininet   
from mininet.cli import CLI
from MS.Env.MininetController import send_packet_and_capture, get_a_mininet
import networkx as nx

def function_test():
    
  N_PACKETS_TO_CAPTURE = 30
  FLOW_DURATION = 30 
  g = nx.Graph()

  g.add_nodes_from(["1", "2"])
  g.add_edge("1", "2")

  with get_a_mininet(g) as net: 
    ping = 100
    CLI(net)
    for host in net.hosts:
      print(f"主机名称: {host.name}, IP: {host.IP()}")
    while ping == 100:
      ping = net.pingAll()

    server, client = net.get("h1", "h2")
    print(server.IP())
    print(client.IP())
    # --- 实验 1: VoIP ---
    voip_tensor = send_packet_and_capture(
      server=server,
      client=client,
      flow_type='voip',
      duration_sec=FLOW_DURATION,
      n_packets_to_capture=N_PACKETS_TO_CAPTURE
    )
    print("--- VoIP 特征 Tensor ---")
    print(voip_tensor)
    print(f"(类型: {voip_tensor.dtype}, 形状: {voip_tensor.shape})")
  
    voip_tensor = send_packet_and_capture(
      server=server,
      client=client,
      flow_type='gaming',
      duration_sec=FLOW_DURATION,
      n_packets_to_capture=N_PACKETS_TO_CAPTURE
    )
    print("--- Gaming 特征 Tensor ---")
    print(voip_tensor)
    print(f"(类型: {voip_tensor.dtype}, 形状: {voip_tensor.shape})")

    voip_tensor = send_packet_and_capture(
      server=server,
      client=client,
      flow_type='streaming',
      duration_sec=FLOW_DURATION,
      n_packets_to_capture=N_PACKETS_TO_CAPTURE
    )
    print("--- Streaming 特征 Tensor ---")
    print(voip_tensor)
    print(f"(类型: {voip_tensor.dtype}, 形状: {voip_tensor.shape})")

if __name__ == '__main__':
  function_test()