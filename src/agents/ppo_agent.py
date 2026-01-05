
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch_geometric.data import Data

from ..utils import compute_advantages, AttrDict, logger
from ..models import FilmGenerator, FilmGNN, actor, critic

class FiLMPPOAgent(nn.Module):
    def __init__(self, config: AttrDict):

        super().__init__()
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")

        self.clip_eps = self.config.train.clip_eps
        self.film = FilmGenerator(config)
        self.gnn = FilmGNN(config)

        # Define actor and critic networks
        # Actor 输入：[当前节点, 目标节点, 邻居节点, 边特征]
        # Critic 输入：[当前节点, 目标节点]
        critic_input_dim =  config.model.hidden_dim*2 
        actor_input_dim =  config.model.hidden_dim*3 + config.model.edge_dim 
        self.actor = actor(input_dim= actor_input_dim)
        self.critic = critic(input_dim= critic_input_dim)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=config.train.lr)
        self.to(self.device)

    def forward(self, graph_data, fingerprint, curr_node_idx, target_node_idx):
        """
        前向传播 (用于训练时的 Batch 计算)
        注意：实际推理(Rollout)时通常拆开调用
        """
        node_embeds = self.get_node_embeddings(graph_data=graph_data, fingerprint=fingerprint)
        
        curr_feat = node_embeds[curr_node_idx]     # (H,)
        target_feat = node_embeds[target_node_idx] # (H,)
        
        # 4. 计算 Critic Value
        # 拼接: [Curr | Target]
        critic_input = torch.cat([curr_feat, target_feat], dim=-1)
        value = self.critic(critic_input)
        
        return node_embeds, value

    def get_action(self, state: Data, node_embeds: torch.Tensor, action_mask: torch.Tensor, curr_node_idx: int, target_node_idx: int, 
                    neighbor_indices: list[int], neighbor_edge_attrs: torch.Tensor, deterministic=False) -> tuple[int| None, torch.Tensor| None, torch.Tensor| None, int| None]: 
        """
        [推理核心] 逐跳决策 - 给邻居打分并采样
        
        Args:
            state: 图数据对象 (Data)
            node_embeds: GNN 输出的全局节点特征 (N, H)
            action_mask: 当前节点的有效邻居掩码 (K,)
            curr_node_idx: 当前节点 ID (int)
            target_node_idx: 目标节点 ID (int)
            neighbor_indices: 有效邻居 ID 列表 (list[int]) - 当前节点的所有邻居
            neighbor_edge_attrs: 对应邻居的边特征 Tensor (K, Edge_Dim)
        
        Returns:
            next_node: 选中的邻居ID (int)
            log_prob: 该步的 log_prob (float)
            logits: 所有邻居的未归一化分数 (Tensor, K)
            action_idx: 选中的邻居索引 (int)
        """
        if len(neighbor_indices) == 0:
            logger.log(f"当前节点 {curr_node_idx} 没有有效邻居", tag="Warning")
            return None, None, None, None # 死胡同，返回4个None
            
        # 1. 准备特征
        neighbor_indices_tensor = torch.tensor(neighbor_indices, device=node_embeds.device)
        neighbor_feats = node_embeds[neighbor_indices_tensor]       # (K, H)
        curr_feat = node_embeds[curr_node_idx].unsqueeze(0)     # (1, H)
        target_feat = node_embeds[target_node_idx].unsqueeze(0) # (1, H)
        
        K = len(neighbor_indices)
        
        # 2. 拼接 Actor 输入
        # 广播 Current 和 Target
        # Input: [Curr(K,H) | Target(K,H) | Neigh(K,H) | Edge(K,E)]
        actor_input = torch.cat([
            curr_feat.repeat(K, 1),
            target_feat.repeat(K, 1),
            neighbor_feats,
            neighbor_edge_attrs
        ], dim=-1)
        
        # 3. 打分
        logits = self.actor(actor_input).squeeze(-1) # (K,)
        
        # 4. 应用掩码
        if action_mask is not None:
            if not action_mask.any():
                # 所有动作都被屏蔽，视为死胡同
                return None, None, None, None
            logits[~action_mask] = -1e9
        

        # 4. 采样
        if deterministic:
            # 贪婪模式 (测试用)
            action_idx = torch.argmax(logits).item()
            log_prob = torch.tensor(0.0).to(logits.device) # 确定性策略无 log_prob
        else:
            # 随机模式 (训练用)
            probs = F.softmax(logits, dim=0)
            dist = torch.distributions.Categorical(probs)
            action_idx = dist.sample().item()
            action_tensor = torch.tensor(action_idx, device=logits.device)
            log_prob = dist.log_prob(action_tensor)
        
        # 返回: 选中的邻居ID, 该步的 log_prob
        assert isinstance(action_idx, int), f"action_idx 应该是整数，当前为 {type(action_idx)}"

        next_node = neighbor_indices[action_idx]
        return int(next_node), log_prob, logits, action_idx
  
    def get_node_embeddings(self, graph_data, fingerprint):
        fingerprint = fingerprint.float()
        film_params = self.film(fingerprint)
        batch_vec = getattr(graph_data, 'batch', None)
        node_embeds = self.gnn(graph_data.x, graph_data.edge_index, graph_data.edge_attr, film_params, batch_vector=batch_vec)

        return node_embeds
  
    def evaluate_value(self, node_embeds, curr_node_idx, target_node_idx):
        """用于 PPO 计算 Advantage"""
        curr_feat = node_embeds[curr_node_idx]
        target_feat = node_embeds[target_node_idx]
        return self.critic(torch.cat([curr_feat, target_feat], dim=-1))
  
    def evaluate_batch(self, batch_graph, fingerprints, curr_node_indices, target_node_indices, actions):
        """
        [PPO 核心训练模块] 批量重新评估动作价值和概率
        
        该函数在 PPO update 循环中被调用，用于计算新策略下的 log_probs 和 value，
        以便与 old_log_probs 对比计算 Ratio，并计算 Advantage 的梯度。

        Args:
            batch_graph (Batch): PyG 的大图对象。
            fingerprints (Tensor): 流量指纹。
            curr_node_indices (Tensor): 当前所在节点的局部索引。
            target_node_indices (Tensor): 目标节点的局部索引。
            actions (Tensor): 实际执行的动作（在邻居列表中的下标）。

        Returns:
            new_log_probs (Tensor): 新策略下采取该动作的对数概率 (用于计算 Actor Loss)。
            state_values (Tensor): Critic 对当前状态的打分 (用于计算 Critic Loss)。
            dist_entropy (Tensor): 动作分布的熵 (用于计算 Entropy Bonus，鼓励探索)。
        """
        
        # 1. 重新运行 GNN 提取特征 (Re-run GNN)
        node_embeds = self.get_node_embeddings(graph_data=batch_graph, fingerprint=fingerprints)

        # 2. 计算全局索引偏移 (Global Index Alignment)
        
        # [Source] batch_graph.ptr: PyG Batch 属性，记录了每个子图在全局大图中的起始索引位置
        # [Logic] 例如：图1有10个点，图2有14个点。ptr=[0, 10, 24...]。
        #         图2的第0号节点，在 node_embeds 里的实际位置是 10 + 0 = 10。
        # [Shape] (Batch_Size, )
        batch_ptr = batch_graph.ptr[:-1] 

        # [Action] 将局部索引 (0~13) 转换为全局嵌入索引 (10~23)
        # [Shape] (Batch_Size, )
        curr_global_idx = curr_node_indices + batch_ptr
        target_global_idx = target_node_indices + batch_ptr

        # =====================================================================
        # 3. 批量计算状态价值 (Critic Evaluation)
        # =====================================================================
        
        curr_feats = node_embeds[curr_global_idx]
        target_feats = node_embeds[target_global_idx]

        critic_input = torch.cat([curr_feats, target_feats], dim=-1)

        # [Action] 计算 V(s)
        # [Shape] (Batch_Size, )
        # [Source] self.critic: 价值网络
        state_values = self.critic(critic_input).squeeze(-1)

        # 4. 批量计算动作概率 (Actor Evaluation)
        # 难点：每个节点的邻居数量 (K) 不同，难以直接矩阵运算。
        # 方案：遍历 Batch 中的每个样本，动态查找邻居并计算 Logits。
        
        new_log_probs = []
        dist_entropies = []
        
        # 获取大图的边索引，用于查找邻居
        # [Shape] (2, Total_Edges)
        edge_attr = batch_graph.edge_attr
        edge_index = batch_graph.edge_index

        # 遍历 Batch 中的每一个样本 (Loop over Batch)
        for i in range(len(actions)):
            # 4.1 准备单样本数据 
            
            # [Var] u_global: 当前样本 i 在大图中的当前节点 ID
            u_global = curr_global_idx[i].item()
            
            # [Logic] 在 edge_index 中查找 u_global 的所有出边
            # mask 是一个布尔向量，标记了哪些边是从 u_global 出发的
            neighbor_mask = (edge_index[0] == u_global)
            
            # [Var] neighbor_global_indices: 所有邻居在大图中的 ID 
            # [Shape] (K, ) 其中 K 是邻居数量 (当前样本的度)
            neighbor_global_indices = edge_index[1][neighbor_mask]
            
            # [Var] neighbor_edge_attrs: 这些边的属性 (Delay, BW, Loss...)
            # [Shape] (K, Edge_Dim)
            neighbor_edge_attrs = edge_attr[neighbor_mask]

            # [Check] 死胡同处理 (理论上训练数据里不该有死胡同的 action，但为了鲁棒性)
            if len(neighbor_global_indices) == 0:
                # 填入默认值防止报错，这部分 Loss 会被 mask 掉或者 value 极低
                new_log_probs.append(torch.tensor(-1e9).to(node_embeds.device))
                dist_entropies.append(torch.tensor(0.0).to(node_embeds.device))
                continue

            # 4.2 准备 Actor 输入特征 
            
            # [Action] 提取邻居节点的特征
            # [Shape] (K, Hidden_Dim)
            neighbor_feats = node_embeds[neighbor_global_indices]
            
            # [Action] 提取当前节点和目标节点的特征 (复用上面的切片)
            # 这里的 unsqueeze(0) 是为了变成 (1, H) 以便广播
            c_feat = curr_feats[i].unsqueeze(0)   # (1, H)
            t_feat = target_feats[i].unsqueeze(0) # (1, H)
            
            # [Var] K: 当前节点的度 (邻居数)
            K = neighbor_feats.size(0)

            # [Action] 拼接 Actor 输入向量 (这就完全复现了 get_action 的逻辑)
            # [Shape] (K, Actor_Input_Dim)
            # Input: [Current(K,H) | Target(K,H) | Neighbors(K,H) | EdgeAttrs(K,E)]

            actor_input = torch.cat([
                c_feat.repeat(K, 1), # 广播当前节点特征
                t_feat.repeat(K, 1), # 广播目标节点特征
                neighbor_feats,      # 邻居特征
                neighbor_edge_attrs  # 边特征
            ], dim=-1)

            # 4.3 神经网络打分 
            # [Source] self.actor: 策略网络 
            # [Action] 计算每个邻居的 Logits (未归一化的分数) 
            logits = self.actor(actor_input).squeeze(-1) 

            # 4.4 动作掩码 (Action Masking)  
            # 我们利用 neighbor_mask 直接切片  
            mask = batch_graph.action_mask[neighbor_mask]  
            logits[~mask] = -1e9 # 屏蔽

            # 4.5 计算分布统计量 
            
            # [Action] 生成概率分布 
            probs = torch.softmax(logits, dim=0) 
            dist =  torch.distributions.Categorical(probs) 

            # [Var] act_idx: 实际采取的动作 (0 ~ K-1)
            act_idx = actions[i] 
            
            # [Output 1] 计算 log_prob (用于 Ratio) 
            log_prob = dist.log_prob(act_idx) 
            new_log_probs.append(log_prob) 
            
            # [Output 2] 计算 Entropy (用于鼓励探索)
            dist_entropies.append(dist.entropy())

        # 5. 堆叠与返回
        # [Action] 将列表转换为 Tensor

        return (
        torch.stack(new_log_probs),  # (Batch_Size, )
        state_values,                # (Batch_Size, )
        torch.stack(dist_entropies)) # (Batch_Size, )
    
    def update(self, memory):

        (states_batch, actions, old_log_probs, values, rewards, 
        is_terminals, fingerprints, curr_idxs, target_idxs) = memory.get_all()
        
        # 2. 计算优势函数
        # 注意：为了计算最后一步的 delta，我们需要多拿一个 next_value
        advantages = compute_advantages(rewards, values, is_terminals)
        advantages = advantages.to(self.device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

        returns = advantages + values # 计算目标回报
        
        # 标准化优势函数 (提升训练稳定性)

        total_actor_loss = 0
        total_critic_loss = 0

        # 3. PPO 更新循环
        for _ in range(self.config.train.ppo_epochs):
            # 重新评估当前的动作概率和价值 (因为参数在变)
            # 这里需要你在 SDNAgent 中实现一个 evaluate 方法
            new_log_probs, state_values, dist_entropy = self.evaluate_batch(
                states_batch, fingerprints, curr_idxs, target_idxs, actions
            )

            # 计算 Ratio (新旧策略比率)
            ratio = torch.exp(new_log_probs - old_log_probs)

            # PPO 截断损失 (Actor Loss)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()

            # 价值损失 (Critic Loss)
            critic_loss = 0.5 * F.mse_loss(state_values, returns)

            # 熵损失 (鼓励探索)
            entropy_loss = -self.config.train.entropy_coef * dist_entropy.mean()

            # 总 Loss
            loss = actor_loss + critic_loss + entropy_loss

            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.5)
            self.optimizer.step()

            total_actor_loss += actor_loss.item()
            total_critic_loss += critic_loss.item()

        memory.clear() # 清空记忆，准备下一轮采样

        return total_actor_loss / self.config.train.ppo_epochs, total_critic_loss / self.config.train.ppo_epochs


