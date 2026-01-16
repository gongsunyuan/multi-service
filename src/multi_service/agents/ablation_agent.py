import torch
import torch.nn as nn
from torch.nn import functional as F

from multi_service.agents.based_agent import BaseSDNAgent
from multi_service.utils import compute_advantages, logger
from multi_service.models import FilmGenerator, FilmGNN, actor, critic

class AblationAgent(BaseSDNAgent):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.mode = config.model.ablation_mode # 'full', 'vanilla_gnn', 'film_drl', 'vanilla_drl'
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self.clip_eps = self.config.train.clip_eps 

        # 1. 初始化意图生成器
        self.use_film = "film" in self.mode or self.mode == "full"
        self.film = FilmGenerator(config) 

        # 2. 初始化核心编码器
        self.use_gnn = "gnn" in self.mode or self.mode == "full"
        if self.use_gnn:
            self.encoder = FilmGNN(config)
        else:
            # DRL 模式：线性层将原始节点维度 (9) 映射到 hidden_dim [cite: 308, 317]
            self.encoder = nn.Linear(config.model.node_dim, config.model.hidden_dim)

        # 3. 初始化决策头 (维度始终对齐)
        critic_input_dim =  config.model.hidden_dim*2 
        actor_input_dim =  config.model.hidden_dim*3 + config.model.edge_dim 
        self.actor = actor(input_dim= actor_input_dim)
        self.critic = critic(input_dim= critic_input_dim)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=config.train.lr) 
        self.to(self.device)

    def get_node_embeddings(self, graph_data, fingerprint):
        """
        [核心适配] 无论哪种模式，最终输出 (N, H) 维度的嵌入 [cite: 35]
        """
        # 处理 FiLM 调制参数
        if self.use_film and self.mode != "vanilla_gnn":
            film_params = self.film(fingerprint.float()) 
        else:
            # 消融实验：强制 FiLM 参数为 0 (即 Identity 变换)
            batch_size = fingerprint.size(0)
            film_params = torch.zeros(
                batch_size, 
                self.config.model.hidden_dim * 2 * self.config.model.gnn_layers, 
                device=self.device
            )

        if self.use_gnn:
            return self.encoder(
                graph_data.x, graph_data.edge_index, graph_data.edge_attr, 
                film_params, getattr(graph_data, 'batch', None)
            ) 
        else:
            # DRL 模式：原始特征投影后，如果是 FiLM-DRL 则手动应用调制
            h = self.encoder(graph_data.x)
            if self.use_film and self.mode == "film_drl":
                params = film_params.view(film_params.size(0), self.config.model.gnn_layers, 2, -1)
                gamma = params[:, 0, 0, :] + 1.0 # 取第一层模拟调制
                beta = params[:, 0, 1, :]
                batch_vec = getattr(graph_data, 'batch', torch.zeros(h.size(0), dtype=torch.long, device=self.device))
                h = h * gamma[batch_vec] + beta[batch_vec]
            return h

    def get_action(self, state, node_embeds, action_mask, curr_node_idx, target_node_idx, 
                 neighbor_indices, neighbor_edge_attrs, deterministic=False):
        """
        [推理核心] 逻辑与 FiLMPPOAgent 保持一致
        """
        if len(neighbor_indices) == 0:
            return None, None, None, None

        neighbor_indices_tensor = torch.tensor(neighbor_indices, device=node_embeds.device)
        neighbor_feats = node_embeds[neighbor_indices_tensor] # (K, H) 
        curr_feat = node_embeds[curr_node_idx].unsqueeze(0) # (1, H)
        target_feat = node_embeds[target_node_idx].unsqueeze(0) # (1, H) 
        
        K = len(neighbor_indices)
        actor_input = torch.cat([
            curr_feat.repeat(K, 1), target_feat.repeat(K, 1), 
            neighbor_feats, neighbor_edge_attrs
        ], dim=-1) 
        
        logits = self.actor(actor_input).squeeze(-1) 
        if action_mask is not None:
            logits[~action_mask] = -1e9

        if deterministic:
            action_idx = torch.argmax(logits).item()
            log_prob = torch.tensor(0.0).to(logits.device)
        else:
            probs = F.softmax(logits, dim=0) 
            dist = torch.distributions.Categorical(probs) 
            action_idx = dist.sample().item()
            log_prob = dist.log_prob(torch.tensor(action_idx, device=logits.device))

        return int(neighbor_indices[action_idx]), log_prob, logits, action_idx 

    def evaluate_value(self, node_embeds, curr_node_idx, target_node_idx):
        """用于 PPO 计算 Advantage"""
        curr_feat = node_embeds[curr_node_idx]
        target_feat = node_embeds[target_node_idx]
        return self.critic(torch.cat([curr_feat, target_feat], dim=-1))
    
    def evaluate_batch(self, batch_graph, fingerprints, curr_node_indices, target_node_indices, actions):
        """
        [训练核心] 批量计算概率和价值 
        """
        node_embeds = self.get_node_embeddings(batch_graph, fingerprints)
        batch_ptr = batch_graph.ptr[:-1] 
        curr_global_idx = curr_node_indices + batch_ptr 
        target_global_idx = target_node_indices + batch_ptr 

        # Critic 评估 
        critic_input = torch.cat([node_embeds[curr_global_idx], node_embeds[target_global_idx]], dim=-1)
        state_values = self.critic(critic_input).squeeze(-1)

        new_log_probs, dist_entropies = [], []
        edge_index, edge_attr = batch_graph.edge_index, batch_graph.edge_attr 

        for i in range(len(actions)):
            u_global = curr_global_idx[i].item()
            neighbor_mask = (edge_index[0] == u_global)   
            neighbor_global_indices = edge_index[1][neighbor_mask] 
            
            neighbor_feats = node_embeds[neighbor_global_indices] 
            c_feat = node_embeds[u_global].unsqueeze(0)
            t_feat = node_embeds[target_global_idx[i]].unsqueeze(0)
        
            K = neighbor_feats.size(0)
            actor_input = torch.cat([
                c_feat.repeat(K, 1), t_feat.repeat(K, 1), 
                neighbor_feats, edge_attr[neighbor_mask]
            ], dim=-1) 
        
            logits = self.actor(actor_input).squeeze(-1)
            mask = batch_graph.action_mask[neighbor_mask]   
            logits[~mask] = -1e9
            
            dist = torch.distributions.Categorical(torch.softmax(logits, dim=0))   
            new_log_probs.append(dist.log_prob(actions[i]))
            dist_entropies.append(dist.entropy())

        return torch.stack(new_log_probs), state_values, torch.stack(dist_entropies)   

    def update(self, memory):
        """
            [策略更新] PPO 循环 
        """
        (states_batch, actions, old_log_probs, values, rewards, 
            is_terminals, fingerprints, curr_idxs, target_idxs) = memory.get_all()   
        
        advantages = compute_advantages(rewards, values, is_terminals).to(self.device)   
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

        returns = advantages + values

        for _ in range(self.config.train.ppo_epochs):   
            new_log_probs, state_values, dist_entropy = self.evaluate_batch(
                states_batch, fingerprints, curr_idxs, target_idxs, actions
            )

            ratio = torch.exp(new_log_probs - old_log_probs)   
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()      
            
            critic_loss = 0.5 * F.mse_loss(state_values, returns)   
            entropy_loss = -self.config.train.entropy_coef * dist_entropy.mean()      
            
            loss = actor_loss + critic_loss + entropy_loss
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.5)   
            self.optimizer.step()

        memory.clear()   
        return actor_loss.item() / self.config.train.ppo_epochs, critic_loss.item() / self.config.train.ppo_epochs

