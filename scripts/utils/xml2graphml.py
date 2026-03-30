import xml.etree.ElementTree as ET
from xml.dom import minidom
import os

def convert_to_graphml_stable(input_str, output_file):
    # 确保输出目录存在
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 初始化 GraphML
    graphml = ET.Element("graphml", {
        "xmlns": "http://graphml.graphdrawing.org/xmlns",
        "xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance",
        "xsi:schemaLocation": "http://graphml.graphdrawing.org/xmlns http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd"
    })

    # 属性定义
    ET.SubElement(graphml, "key", {"id": "x", "for": "node", "attr.name": "x_coordinate", "attr.type": "double"})
    ET.SubElement(graphml, "key", {"id": "y", "for": "node", "attr.name": "y_coordinate", "attr.type": "double"})
    graph = ET.SubElement(graphml, "graph", {"id": "G", "edgedefault": "undirected"})

    lines = input_str.strip().split('\n')
    current_section = None
    node_count = 0
    link_count = 0

    for line in lines:
        line = line.strip()
        if not line or line == ")":
            continue

        # 判断区域
        if line.startswith("NODES"):
            current_section = "NODES"
            continue
        elif line.startswith("LINKS"):
            current_section = "LINKS"
            continue

        # 解析数据
        try:
            if current_section == "NODES":
                # 格式: Name ( Lon Lat )
                # 处理掉括号和多余空格
                parts = line.replace('(', '').replace(')', '').split()
                if len(parts) >= 3:
                    node_id = parts[0]
                    lon = parts[1]
                    lat = parts[2]

                    node_elem = ET.SubElement(graph, "node", {"id": node_id})
                    ET.SubElement(node_elem, "data", {"key": "x"}).text = lon
                    ET.SubElement(node_elem, "data", {"key": "y"}).text = lat
                    node_count += 1

            elif current_section == "LINKS":
                # 格式: LinkID ( Node1 Node2 ) ...
                parts = line.replace('(', '').replace(')', '').split()
                if len(parts) >= 3:
                    link_id = parts[0]
                    target = parts[1]
                    source = parts[2]

                    ET.SubElement(graph, "edge", {
                        "id": link_id,
                        "source": source,
                        "target": target
                    })
                    link_count += 1
        except Exception as e:
            print(f"解析行出错: {line}, 错误: {e}")

    # 保存文件
    if node_count > 0:
        raw_str = ET.tostring(graphml, 'utf-8')
        pretty_xml = minidom.parseString(raw_str).toprettyxml(indent="  ")
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(pretty_xml)
        print(f"转换成功！\n节点数: {node_count}\n连边数: {link_count}\n保存路径: {output_file}")
    else:
        print("错误：未能提取到任何数据，请检查输入字符串。")

# 你的原始数据
raw_data = """
NODES (
  at1.at ( 16.3729 48.2091 )
  be1.be ( 4.3518 50.8469 )
  ch1.ch ( 6.1399 46.2038 )
  cz1.cz ( 14.4423 50.0785 )
  de1.de ( 8.6842 50.1122 )
  es1.es ( -3.7033 40.4167 )
  fr1.fr ( 2.351 48.8566 )
  gr1.gr ( 23.5808 37.9778 )
  hr1.hr ( 15.9644 45.8071 )
  hu1.hu ( 19.0936 47.4976 )
  ie1.ie ( -6.2573 53.3416 )
  il1.il ( 34.8097 32.0714 )
  it1.it ( 9.19 45.4642 )
  lu1.lu ( 6.1296 49.6112 )
  nl1.nl ( 4.9407 52.3236 )
  ny1.ny ( -73.94384 40.6698 )
  pl1.pl ( 16.8874 52.3963 )
  pt1.pt ( -9.1363 38.7073 )
  se1.se ( 17.8742 59.3617 )
  si1.si ( 14.5148 46.0574 )
  sk1.sk ( 17.1297 48.1531 )
  uk1.uk ( -0.1264 51.5086 )
)

LINKS (
  at1.at_ch1.ch ( at1.at ch1.ch ) 0.00 0.00 0.00 0.00 ( 40000.00 804.00 )
  at1.at_de1.de ( at1.at de1.de ) 0.00 0.00 0.00 0.00 ( 40000.00 598.00 )
  at1.at_hu1.hu ( at1.at hu1.hu ) 0.00 0.00 0.00 0.00 ( 40000.00 218.00 )
  at1.at_ny1.ny ( at1.at ny1.ny ) 0.00 0.00 0.00 0.00 ( 40000.00 6802.00 )
  at1.at_si1.si ( at1.at si1.si ) 0.00 0.00 0.00 0.00 ( 40000.00 277.00 )
  be1.be_fr1.fr ( be1.be fr1.fr ) 0.00 0.00 0.00 0.00 ( 40000.00 264.00 )
  be1.be_lu1.lu ( be1.be lu1.lu ) 0.00 0.00 0.00 0.00 ( 40000.00 186.00 )
  be1.be_nl1.nl ( be1.be nl1.nl ) 0.00 0.00 0.00 0.00 ( 40000.00 169.00 )
  ch1.ch_fr1.fr ( ch1.ch fr1.fr ) 0.00 0.00 0.00 0.00 ( 40000.00 410.00 )
  ch1.ch_it1.it ( ch1.ch it1.it ) 0.00 0.00 0.00 0.00 ( 40000.00 250.00 )
  cz1.cz_de1.de ( cz1.cz de1.de ) 0.00 0.00 0.00 0.00 ( 40000.00 411.00 )
  cz1.cz_pl1.pl ( cz1.cz pl1.pl ) 0.00 0.00 0.00 0.00 ( 40000.00 309.00 )
  cz1.cz_sk1.sk ( cz1.cz sk1.sk ) 0.00 0.00 0.00 0.00 ( 40000.00 290.00 )
  de1.de_fr1.fr ( de1.de fr1.fr ) 0.00 0.00 0.00 0.00 ( 40000.00 478.00 )
  de1.de_gr1.gr ( de1.de gr1.gr ) 0.00 0.00 0.00 0.00 ( 40000.00 1794.00 )
  de1.de_ie1.ie ( de1.de ie1.ie ) 0.00 0.00 0.00 0.00 ( 40000.00 1088.00 )
  de1.de_it1.it ( de1.de it1.it ) 0.00 0.00 0.00 0.00 ( 40000.00 518.00 )
  de1.de_nl1.nl ( de1.de nl1.nl ) 0.00 0.00 0.00 0.00 ( 40000.00 358.00 )
  de1.de_se1.se ( de1.de se1.se ) 0.00 0.00 0.00 0.00 ( 40000.00 1184.00 )
  es1.es_fr1.fr ( es1.es fr1.fr ) 0.00 0.00 0.00 0.00 ( 40000.00 1054.00 )
  es1.es_it1.it ( es1.es it1.it ) 0.00 0.00 0.00 0.00 ( 40000.00 1189.00 )
  es1.es_pt1.pt ( es1.es pt1.pt ) 0.00 0.00 0.00 0.00 ( 40000.00 503.00 )
  fr1.fr_lu1.lu ( fr1.fr lu1.lu ) 0.00 0.00 0.00 0.00 ( 40000.00 287.00 )
  fr1.fr_uk1.uk ( fr1.fr uk1.uk ) 0.00 0.00 0.00 0.00 ( 40000.00 343.00 )
  gr1.gr_it1.it ( gr1.gr it1.it ) 0.00 0.00 0.00 0.00 ( 40000.00 1453.00 )
  hr1.hr_hu1.hu ( hr1.hr hu1.hu ) 0.00 0.00 0.00 0.00 ( 40000.00 304.00 )
  hr1.hr_si1.si ( hr1.hr si1.si ) 0.00 0.00 0.00 0.00 ( 40000.00 115.00 )
  hu1.hu_sk1.sk ( hu1.hu sk1.sk ) 0.00 0.00 0.00 0.00 ( 40000.00 163.00 )
  ie1.ie_uk1.uk ( ie1.ie uk1.uk ) 0.00 0.00 0.00 0.00 ( 40000.00 463.00 )
  il1.il_it1.it ( il1.il it1.it ) 0.00 0.00 0.00 0.00 ( 40000.00 2658.00 )
  il1.il_nl1.nl ( il1.il nl1.nl ) 0.00 0.00 0.00 0.00 ( 40000.00 3296.00 )
  nl1.nl_uk1.uk ( nl1.nl uk1.uk ) 0.00 0.00 0.00 0.00 ( 40000.00 359.00 )
  ny1.ny_uk1.uk ( ny1.ny uk1.uk ) 0.00 0.00 0.00 0.00 ( 40000.00 5575.00 )
  pl1.pl_se1.se ( pl1.pl se1.se ) 0.00 0.00 0.00 0.00 ( 40000.00 777.00 )
  pt1.pt_uk1.uk ( pt1.pt uk1.uk ) 0.00 0.00 0.00 0.00 ( 40000.00 1588.00 )
  se1.se_uk1.uk ( se1.se uk1.uk ) 0.00 0.00 0.00 0.00 ( 40000.00 1426.00 )
)
"""

if __name__ == "__main__":
    convert_to_graphml_stable(raw_data, output_file="data/topologies/geant.graphml")
