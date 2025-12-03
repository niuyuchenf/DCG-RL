import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D

class FullConnectedLayers(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(FullConnectedLayers, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)  # 第一个全连接层
        self.fc2 = nn.Linear(128, 128)       # 第二个全连接层
        self.fc3 = nn.Linear(128, output_dim)  # 第三个全连接层

    def forward(self, x):
        x = self.fc1(x)           # 通过第一个全连接层
        x = F.relu(x)             # 激活函数 ReLU
        x = self.fc2(x)           # 通过第二个全连接层
        x = F.relu(x)             # 激活函数 ReLU
        x = self.fc3(x)           # 通过第三个全连接层
        return x
    
    
class MDN(nn.Module):
    def __init__(self, input_dim,num_mixtures,output_dim):
        """
        input_dim: 输入特征的维度，例如3
        hidden_dim: 隐藏层神经元数量，例如64
        num_mixtures: 混合高斯成分数，例如5
        output_dim: 输出变量的维度，例如2
        """
        super(MDN, self).__init__()
        self.num_mixtures = num_mixtures
        self.output_dim = output_dim
        
        # 两层隐藏层
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        total_params = num_mixtures + num_mixtures * output_dim * 2
        self.fc_out = nn.Linear(128, total_params)
    
    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        params = self.fc_out(h)
        
        logits = params[:, :self.num_mixtures]
        weights = F.softmax(logits, dim=1)
        
        start = self.num_mixtures
        end = start + self.num_mixtures * self.output_dim
        means = params[:, start:end]
        means = means.view(-1, self.num_mixtures, self.output_dim)
        
        start = end
        stds = params[:, start:]
        stds = stds.view(-1, self.num_mixtures, self.output_dim)
        stds = F.softplus(stds) + 1e-6  # 防止数值为0
        
        return weights, means, stds
    
    def mdn_loss(self, y, weights, means, stds):
        y = y.unsqueeze(1).expand_as(means)

        dist = D.Normal(means, stds)
        log_probs = dist.log_prob(y).sum(dim=2)  
        weighted_log_probs = log_probs + torch.log(weights + 1e-6)
        log_prob = torch.logsumexp(weighted_log_probs, dim=1)
        
        return -log_prob.mean()
    
    def sample(self, x):
        """
        从MDN网络的输出中采样预测值。每个输入样本只生成一个预测值。

        :param x: 输入数据，形状为 [batch_size, input_dim]
        :return: 采样值，形状为 [batch_size, output_dim]
        """
        # 获取权重、均值和标准差
        weights, means, stds = self.forward(x)

        # 用权重选择一个混合成分
        component_indices = torch.multinomial(weights, num_samples=1, replacement=True)  # 每个样本选择一个分布

        # 根据选择的高斯成分，从均值和标准差中采样
        sampled_means = torch.gather(means, 1, component_indices.unsqueeze(-1).expand(-1, -1, self.output_dim))
        sampled_stds = torch.gather(stds, 1, component_indices.unsqueeze(-1).expand(-1, -1, self.output_dim))

        # 从选定的高斯分布中进行采样
        dist = D.Normal(sampled_means.squeeze(1), sampled_stds.squeeze(1))  # 扁平化成一个向量
        samples = dist.sample()  # 生成采样样本

        return samples  # 形状为 [batch_size, output_dim]




