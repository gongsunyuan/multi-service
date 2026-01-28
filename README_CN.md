# 多业务异构流的差异化路由

## 概述

本研究旨在研究如何使用FiLM（线性映射层）+GNN在面对不同QoS业务流需求的网络流量时，根据不同的QoS业务流需求选择不同的路由，实现差异化路由。

## 实验设计

本实验在 Ubuntu22.04.5LST 操作系统中进行，使用Mininet来模拟网络流量的传输
实验考虑的流类型包括：

- VoIP：延迟敏感但是带宽需求低
- Stream：带宽要求高但是延迟不敏感

拓扑采用了合成拓扑[[data/topologies/train_film.graphml]]，该拓扑为了满足出不同业务流的差异化路由，满足了下面两个特点：

1. 延迟低的链路，带宽一定不高
2. 带宽高的链路，延迟也高

这保证了：

- 满足VoIP流QoS需求的路径，一定不满足Stream流的带宽需求
- 满足Stream流带宽需求的路径，一定不满足VoIP流的延迟需求

因此，只有模型学会了根据不同的流的特点来针对性选择路由，即，针对不同的流实现差异化路由，才能保证不同流的QoS需求被完全满足

## 模型设计

模型由`LSTM` + `FiLM`（线性映射层）+ `GAT`（图注意力网络）+ `Actor`&`Critci` 组成
其中：
- `LSTM`负责获取流量特征，将其转化为 FiLM 可以理解的流量特征向量
- `FiLM`负责生成线性调制的参数，它的输入就是`LSTM`生成的流量特征向量
- `GAT`负责图嵌入，在嵌入时，`FiLM`生成的线性调制参数负责调制隐藏特征向量，让GAT根据不同QoS需求的流量获取不同的图嵌入
- `Actor` 负责逐跳决策，通过 `GAT`的嵌入、当前节点相邻链路的链路属性来选择下一跳节点
- `Critic` 负责评估当前节点的价值

## 训练拓扑

```
<?xml version="1.0" encoding="UTF-8"?>
<graphml xmlns="http://graphml.graphdrawing.org/xmlns">
  <key id="label" for="ode" attr.name="label" attr.type="string"/>
  <key id="zone" for="ode" attr.name="zone" attr.type="string"/>
  <key id="bw" for="edge" attr.name="bandwidth" attr.type="double"/>
  <key id="delay" for="edge" attr.name="delay" attr.type="double"/>

  <graph edgedefault="undirected">
    
    <node id="0"><data key="label">Source_Entry</data><data key="zone">Gateway</data></node>
    <node id="13"><data key="label">Dest_Exit</data><data key="zone">Gateway</data></node>
    
    <node id="1"><data key="label">A_Core_1</data><data key="zone">High_BW</data></node>
    <node id="2"><data key="label">A_Core_2</data><data key="zone">High_BW</data></node>
    <node id="3"><data key="label">A_Core_3</data><data key="zone">High_BW</data></node>
    <node id="4"><data key="label">A_Core_4</data><data key="zone">High_BW</data></node>
    <node id="5"><data key="label">A_Core_5</data><data key="zone">High_BW</data></node>
    <node id="6"><data key="label">A_Core_6</data><data key="zone">High_BW</data></node>

    <node id="7"><data key="label">B_Edge_1</data><data key="zone">Low_Delay</data></node>
    <node id="8"><data key="label">B_Edge_2</data><data key="zone">Low_Delay</data></node>
    <node id="9"><data key="label">B_Edge_3</data><data key="zone">Low_Delay</data></node>
    <node id="10"><data key="label">B_Edge_4</data><data key="zone">Low_Delay</data></node>
    <node id="11"><data key="label">B_Edge_5</data><data key="zone">Low_Delay</data></node>
    <node id="12"><data key="label">B_Edge_6</data><data key="zone">Low_Delay</data></node>

    <edge source="0" target="1"><data key="bw">100.0</data><data key="delay">2.0</data></edge>
    <edge source="1" target="2"><data key="bw">100.0</data><data key="delay">20.0</data></edge>
    <edge source="2" target="13"><data key="bw">18.0</data><data key="delay">10.0</data></edge>
    <edge source="1" target="3"><data key="bw">10.0</data><data key="delay">2.0</data></edge>
    <edge source="3" target="4"><data key="bw">100.0</data><data key="delay">20.0</data></edge>
    <edge source="4" target="13"><data key="bw">100.0</data><data key="delay">20.0</data></edge>
    <edge source="2" target="5"><data key="bw">100.0</data><data key="delay">20.0</data></edge>
    <edge source="5" target="6"><data key="bw">100.0</data><data key="delay">20.0</data></edge>
    <edge source="6" target="4"><data key="bw">100.0</data><data key="delay">20.0</data></edge>

    <edge source="0" target="7"><data key="bw">100.0</data><data key="delay">2.0</data></edge>
    <edge source="7" target="8"><data key="bw">10.0</data><data key="delay">2.0</data></edge>
    <edge source="8" target="13"><data key="bw">100.0</data><data key="delay">20.0</data></edge>
    <edge source="7" target="9"><data key="bw">100.0</data><data key="delay">20.0</data></edge>
    <edge source="9" target="10"><data key="bw">10.0</data><data key="delay">2.0</data></edge>
    <edge source="10" target="13"><data key="bw">10.0</data><data key="delay">2.0</data></edge>
    <edge source="8" target="11"><data key="bw">10.0</data><data key="delay">2.0</data></edge>
    <edge source="11" target="12"><data key="bw">10.0</data><data key="delay">2.0</data></edge>
    <edge source="12" target="10"><data key="bw">10.0</data><data key="delay">2.0</data></edge>

    <edge source="3" target="9"><data key="bw">10.0</data><data key="delay">20.0</data></edge>
    <edge source="4" target="10"><data key="bw">30.0</data><data key="delay">10.0</data></edge>

  </graph>
</graphml>
```