import networkx as nx
import os

def generate_standard_nsfnet_graphml():
  # 1. 创建空的无向图
  G = nx.Graph()
  
  # 2. 添加 14 个节点 (带城市名称，方便 Gephi 显示)
  # 依据: NSFNet 1991 T1 Backbone 物理地图
  nodes = {
    0: "Seattle, WA",
    1: "Palo Alto, CA",
    2: "San Diego, CA",
    3: "Salt Lake City, UT",
    4: "Boulder, CO",
    5: "Lincoln, NE",
    6: "Champaign, IL",
    7: "Ann Arbor, MI",
    8: "Pittsburgh, PA",
    9: "Ithaca, NY",
    10: "Princeton, NJ",
    11: "College Park, MD",
    12: "Atlanta, GA",
    13: "Houston, TX"
  }
  
  for node_id, label in nodes.items():
    # label 属性用于 Gephi 显示标签
    G.add_node(node_id, label=label)

  # 3. 添加 21 条标准连接边 (Source, Target)
  edges = [
    (0, 1), (0, 3), (1, 2), (2, 3),  # 西海岸环
    (0, 5), (3, 4),                  # 西部 -> 中部
    (1, 9), (1, 6),                  # 【关键长链路】Palo Alto -> Ithaca/Champaign
    (4, 5), (4, 13), (5, 6), (5, 13), # 中部连接
    (6, 7), (6, 12),                 # 中部 -> 东部/南部
    (7, 8), (7, 9), (8, 10),         # 东北部密集区
    (9, 10), (10, 11),               # 东海岸走廊
    (11, 12), (12, 13)               # 东南部连接
  ]
  G.add_edges_from(edges)

  # 4. 初始化物理属性 (基准值)
  # 这些属性会被写入 GraphML，Gephi 可以读取
  for u, v in G.edges():
    G[u][v]['bandwidth'] = 100.0  # Mbps
    G[u][v]['delay'] = 10.0       # ms (基础延迟)
    G[u][v]['loss'] = 0.0         # %
    G[u][v]['capacity'] = 100.0   # Mbps

  # 5. 导出
  filename = "nsfnet_standard.graphml"
  try:
    nx.write_graphml(G, filename)
    print(f"✅ 成功生成标准 NSFNet 拓扑: {filename}")
    print(f"   - 节点: {G.number_of_nodes()} (0-13)")
    print(f"   - 边数: {G.number_of_edges()} (标准应为 21)")
    print(f"   - 平均度数: {2 * G.number_of_edges() / G.number_of_nodes():.2f} (标准约为 3.0)")
  except Exception as e:
    print(f"❌ 生成失败: {e}")

if __name__ == "__main__":
  generate_standard_nsfnet_graphml()