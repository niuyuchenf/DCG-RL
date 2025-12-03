# 最终的Rewardnetwork
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import GradScaler, autocast
import json
from types import SimpleNamespace
import numpy as np
from gumbel import VQVAEGumbelMatrixLatent
from FCNN import FullConnectedLayers
import random
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
# from torch.cuda.amp import GradScaler, autocast


# 主训练函数参数定义
class RewardNetwork(nn.Module):
    def __init__(self, num_state_var, num_action_variable, params, training=True):
        super(RewardNetwork, self).__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ours_params = params.ours_params
        fc_dims = ours_params.feature_fc_dims
        self.local_causal_model = VQVAEGumbelMatrixLatent(params, num_state_var, num_action_variable, fc_dims, device,1)
        self.fc_layers = FullConnectedLayers(num_state_var + num_action_variable, output_dim=1)
        # self.state_embed = nn.Linear(num_state_var, num_state_var * 128)
        # self.action_embed = nn.Linear(num_action_variable, num_action_variable * 128)
    
    def forward(self, state_feature, action_feature, training=True):

        # ========== 步骤1：先统一到2维格式 ==========
        # 处理state_feature
        while state_feature.dim() > 2:
            if state_feature.size(1) == 1:
                state_feature = state_feature.squeeze(1)
            else:
                break

        if state_feature.dim() == 1:
            state_feature = state_feature.unsqueeze(0)

        # 处理action_feature
        if action_feature.dim() > 2:
            if action_feature.dim() == 4:
                action_feature = action_feature.squeeze(-1).squeeze(-1).squeeze(-1)
                if action_feature.dim() == 1:
                    action_feature = action_feature.unsqueeze(-1)
            elif action_feature.dim() == 3 and action_feature.size(2) == 1:
                action_feature = action_feature.squeeze(2)

        if action_feature.dim() == 1:
            action_feature = action_feature.unsqueeze(0)

        # ========== 步骤2：为VQVAEGumbelMatrixLatent准备3维输入 ==========
        # VQVAEGumbelMatrixLatent期望 [batch_size, 1, feature_dim] 格式
        state_for_causal = state_feature.unsqueeze(1)  # [batch, state_dim] -> [batch, 1, state_dim]
        action_for_causal = action_feature.unsqueeze(1)  # [batch, action_dim] -> [batch, 1, action_dim]

        # ========== 步骤3：调用因果模型 ==========
        local_mask, self.prob = self.local_causal_model(state_for_causal, action_for_causal, training=training)

        # ========== 步骤4：继续后续处理（使用2维张量） ==========
        input_vector = torch.cat([state_feature, action_feature], dim=-1)  # 使用2维张量
        local_mask = local_mask.squeeze(0) 
        input_vector = input_vector.unsqueeze(1).expand(-1, 1, -1)
        combined = local_mask * input_vector  
        combined_flattened = combined.view(state_feature.size(0), -1)  
        final_output = self.fc_layers(combined_flattened)
        return final_output

    def loss_state_prediction(self, predicted_reward, actual_reward):

        state_loss = F.mse_loss(predicted_reward, actual_reward)
        ours_loss = self.local_causal_model.total_loss()
        return state_loss + ours_loss

    # 保存模型参数
    def save_parameters(self, path):
        torch.save(self.state_dict(), path)

    # 加载模型参数
    def load_parameters(self, path, device):
        self.load_state_dict(torch.load(path, map_location=device))

def RewardPredict(num_state_var,num_action_variable,num_epochs,train_states,train_actions,train_rewards,training):
    # 加载参数配置
    with open('policy_params.json', 'r') as f:
        params = json.load(f, object_hook=lambda d: SimpleNamespace(**d))

    # 模型定义
    model = RewardNetwork(num_state_var, num_action_variable, params, training=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.load_parameters('.\\reward\\reward.pth', device)
    # train_states = torch.tensor(train_states, dtype=torch.float32).to(device)
    # train_actions = torch.tensor(train_actions, dtype=torch.float32).to(device)
    # train_rewards = torch.tensor(train_rewards, dtype=torch.float32).to(device)

    # 优化器和混合精度
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 构造 DataLoader
    dataset = TensorDataset(train_states, train_actions, train_rewards)
    train_loader = DataLoader(dataset, batch_size=128, shuffle=True)

    # 初始化 scaler
    scaler = GradScaler()
    for epoch in range(num_epochs):
        # current_tau = max(0.5, 1.0 * (1 - epoch / num_epochs))
        # epoch_loss = 0  # 每个epoch的总损失
        for states, actions, rewards in train_loader:
            states = states.unsqueeze(1)  # [B, 1, 28]
            actions = actions.unsqueeze(1).unsqueeze(1)  # [B, 1, 1]
            rewards = rewards  # [B, 28]
            states, actions, rewards = states.to(device), actions.to(device), rewards.to(device)
            optimizer.zero_grad()
            # with autocast():  # 混合精度上下文
            with torch.amp.autocast('cuda'):  # 修改后的混合精度上下文
                predicted_rewards = model(states, actions)
                loss = model.loss_state_prediction(predicted_rewards, rewards)
            scaler.scale(loss).backward()        # 反向传播（缩放梯度）
            scaler.step(optimizer)               # 更新参数
            scaler.update()                      # 更新缩放器
    model.save_parameters('.\\reward\\reward.pth')




