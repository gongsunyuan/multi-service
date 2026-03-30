from omegaconf import DictConfig
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.optim import lr_scheduler
from loguru import logger
from multi_service.agents.based_agent import BaseSDNAgent
from multi_service.utils import compute_advantages
from multi_service.models import FilmGNN, actor, critic

class GnnAgent(BaseSDNAgent):
    """
    GNN Agent: Uses GNN for node encoding but without FiLM modulation (Identity/Zero params).
    """
    def __init__(self, 
                 node_dim, 
                 edge_dim, 
                 hidden_dim, 
                 gnn_layers, 
                 gnn_heads, 
                 dropout, 
                 lr, 
                 clip_eps, 
                 ppo_epochs, 
                 entropy_coef, 
                 device_name="cpu"):
        
        device = torch.device(device_name if torch.cuda.is_available() else "cpu")
        super().__init__(device)
        
        self.clip_eps = clip_eps
        self.ppo_epochs = ppo_epochs
        self.entropy_coef = entropy_coef
        self.hidden_dim = hidden_dim
        self.gnn_layers = gnn_layers
        
        # 1. Initialize Core Encoder (GNN only)
        # Note: FilmGNN is used but fed with zero params to act as standard GNN
        self.encoder = FilmGNN(
            node_dim=node_dim, 
            edge_dim=edge_dim, 
            hidden_dim=hidden_dim, 
            gnn_layers=gnn_layers, 
            gnn_heads=gnn_heads, 
            dropout=dropout
        )
        
        logger.debug("GnnAgent initialized with GNN (no FiLM)", tag="agent init")

        # 2. Initialize Decision Heads
        critic_input_dim = hidden_dim * 2 
        actor_input_dim = hidden_dim * 3 + edge_dim 
        self.actor = actor(input_dim=actor_input_dim)
        self.critic = critic(input_dim=critic_input_dim)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr) 
        self.scheduler = lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='max', 
            factor=0.5, 
            patience=100, 
            threshold=0.0005, 
            min_lr=1e-6,
        ) 
        self.to(self.device)

    def get_node_embeddings(self, graph_data, fingerprint):
        """
        Uses GNN with zero FiLM parameters (Identity transformation).
        """
        batch_size = fingerprint.size(0)
        # Create zero params
        film_params = torch.zeros(
            batch_size, 
            self.hidden_dim * 2 * self.gnn_layers, 
            device=self.device
        )
        
        return self.encoder(
            graph_data.x, graph_data.edge_index, graph_data.edge_attr, 
            film_params, getattr(graph_data, 'batch', None)
        ) 

    def get_action(self, state, node_embeds, action_mask, curr_node_idx, target_node_idx, 
                 neighbor_indices, neighbor_edge_attrs, deterministic=False):
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
        curr_feat = node_embeds[curr_node_idx]
        target_feat = node_embeds[target_node_idx]
        return self.critic(torch.cat([curr_feat, target_feat], dim=-1))
    
    def evaluate_batch(self, batch_graph, fingerprints, curr_node_indices, target_node_indices, actions):
        node_embeds = self.get_node_embeddings(batch_graph, fingerprints)
        batch_ptr = batch_graph.ptr[:-1] 
        curr_global_idx = curr_node_indices + batch_ptr 
        target_global_idx = target_node_indices + batch_ptr 

        # Critic Evaluation
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
        (states_batch, actions, old_log_probs, values, rewards, 
            is_terminals, fingerprints, curr_idxs, target_idxs) = memory.get_all()   
        
        advantages = compute_advantages(rewards, values, is_terminals).to(self.device)   
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

        returns = advantages + values

        for _ in range(self.ppo_epochs):   
            new_log_probs, state_values, dist_entropy = self.evaluate_batch(
                states_batch, fingerprints, curr_idxs, target_idxs, actions
            )

            ratio = torch.exp(new_log_probs - old_log_probs)   
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()      
            
            critic_loss = 0.5 * F.mse_loss(state_values, returns)   
            entropy_loss = -self.entropy_coef * dist_entropy.mean()      
            
            loss = actor_loss + critic_loss + entropy_loss
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.5)   
            self.optimizer.step()

        memory.clear()   
        return actor_loss.item() / self.ppo_epochs, critic_loss.item() / self.ppo_epochs

    def get_current_lr(self): 
        return self.optimizer.param_groups[0]['lr']
