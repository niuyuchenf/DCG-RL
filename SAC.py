from utils import Actor, Double_Q_Critic
from utils import str2bool, evaluate_policy, Action_adapter, Action_adapter_reverse, Reward_adapter
import torch.nn.functional as F
import numpy as np
import torch
import copy
import random
from reward_predict import RewardNetwork
from state_predict import StateNetwork
import json
from types import SimpleNamespace


class EmptyListError(Exception):
    pass

class SAC_countinuous():
    def __init__(self, **kwargs):
        # Init hyperparameters for agent, just like "self.gamma = opt.gamma, self.lambd = opt.lambd, ..."
        self.__dict__.update(kwargs)
        self.tau = 0.005

        self.actor = Actor(self.state_dim, self.action_dim, (self.net_width,self.net_width)).to(self.dvc)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.a_lr)

        self.q_critic = Double_Q_Critic(self.state_dim, self.action_dim, (self.net_width,self.net_width)).to(self.dvc)
        self.q_critic_optimizer = torch.optim.Adam(self.q_critic.parameters(), lr=self.c_lr)
        self.q_critic_target = copy.deepcopy(self.q_critic)
        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for p in self.q_critic_target.parameters():
            p.requires_grad = False

        self.replay_buffer = ReplayBuffer(self.state_dim, self.action_dim, max_size=int(1e6), dvc=self.dvc)

        if self.adaptive_alpha:
            # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
            self.target_entropy = torch.tensor(-self.action_dim, dtype=float, requires_grad=True, device=self.dvc)
            # We learn log_alpha instead of alpha to ensure alpha>0
            self.log_alpha = torch.tensor(np.log(self.alpha), dtype=float, requires_grad=True, device=self.dvc)
            self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=self.c_lr)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.modify_n = kwargs.get('modify_n', 10)
        state_dim=kwargs.get('state_dim', 11)
        action_dim=kwargs.get('action_dim', 3)
        # main_state=kwargs.get('main_state', 2)
        with open('policy_params.json', 'r') as f:
            self.params = json.load(f, object_hook=lambda d: SimpleNamespace(**d))
        self.reward_model = RewardNetwork(state_dim, action_dim, self.params, training=False)#
        self.reward_model = self.reward_model.to(self.device)
        self.state_model = StateNetwork(state_dim, action_dim, self.params, training=False)#
        self.state_model = self.state_model.to(self.device)
        self.update_frequency = 100
        self.state_dim=state_dim
        self.action_dim=action_dim
        
        
        self.state_model = StateNetwork(num_state_var=self.state_dim,
            num_action_variable=self.action_dim, params=self.params,training=False).to(self.dvc)
        self.reward_model = RewardNetwork(num_state_var=self.state_dim,
            num_action_variable=self.action_dim, params=self.params,training=False).to(self.dvc)

        # 加载权重
        state_checkpoint_path = '.\\state\\state.pth'
        reward_checkpoint_path = '.\\reward\\reward.pth'
        self.state_model.load_parameters(state_checkpoint_path, self.device)
        self.reward_model.load_parameters(reward_checkpoint_path, self.device)
        self.state_model.eval()
        self.reward_model.eval()

    def select_action(self, state, deterministic):
        # only used when interact with the env
        with torch.no_grad():
            state = torch.FloatTensor(state[np.newaxis,:]).to(self.dvc)
            a, _ = self.actor(state, deterministic, with_logprob=False)
        return a.cpu().numpy()[0]

    def train(self, episode):
        s, a, r, s_next, dw = self.replay_buffer.sample(self.batch_size)
        
        # 修改取样出的状态特征
        if len(s) >= self.modify_n:# eposide用来确认什么时候加载因果模型
            states, actions, rewards, next_states, dw = self.modify_state(s, a, r, s_next, dw, self.modify_n, episode)
        else:
            states, actions, rewards, next_states = s, a, r, s_next
            

        #----------------------------- ↓↓↓↓↓ Update Q Net ↓↓↓↓↓ ------------------------------#
        with torch.no_grad():
            a_next, log_pi_a_next = self.actor(next_states, deterministic=False, with_logprob=True)
            target_Q1, target_Q2 = self.q_critic_target(next_states, a_next)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = rewards + (~dw) * self.gamma * (target_Q - self.alpha * log_pi_a_next) #Dead or Done is tackled by Randombuffer

        # Get current Q estimates
        current_Q1, current_Q2 = self.q_critic(states, actions)

        q_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        self.q_critic_optimizer.zero_grad()
        q_loss.backward()
        self.q_critic_optimizer.step()

        #----------------------------- ↓↓↓↓↓ Update Actor Net ↓↓↓↓↓ ------------------------------#
        # Freeze critic so you don't waste computational effort computing gradients for them when update actor
        for params in self.q_critic.parameters(): params.requires_grad = False

        a_new, log_pi_a = self.actor(states, deterministic=False, with_logprob=True)
        current_Q1, current_Q2 = self.q_critic(states, a_new)
        Q = torch.min(current_Q1, current_Q2)

        a_loss = (self.alpha * log_pi_a - Q).mean()
        self.actor_optimizer.zero_grad()
        a_loss.backward()
        self.actor_optimizer.step()

        for params in self.q_critic.parameters(): params.requires_grad = True

        #----------------------------- ↓↓↓↓↓ Update alpha ↓↓↓↓↓ ------------------------------#
        if self.adaptive_alpha:
            # We learn log_alpha instead of alpha to ensure alpha>0
            alpha_loss = -(self.log_alpha * (log_pi_a + self.target_entropy).detach()).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            self.alpha = self.log_alpha.exp()

        #----------------------------- ↓↓↓↓↓ Update Target Net ↓↓↓↓↓ ------------------------------#
        for param, target_param in zip(self.q_critic.parameters(), self.q_critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
    def test(self, env, num_episodes=10):
        """
        Test the current policy in the given environment.
        :param env: Gym environment
        :param num_episodes: Number of episodes to run for testing
        """
        total_rewards = []
        total_speeds = []
        total_collision = []
        for _ in range(num_episodes):
            state, info = env.reset()
            collision=info['collision']
            done = False
            episode_reward = 0
            speed=[]
            collisions=0
            while not done:
                speed.append(state[6])
                action = self.select_action(state, deterministic=True)
                action = Action_adapter(action, 5)
                state, reward, done, info, _  = env.step(action)
                collision = info['collision']
                if collision:
                    collisions += 1
                episode_reward += reward
            total_speeds.append(sum(speed)/len(speed))
            total_rewards.append(episode_reward)
            total_collision.append(collisions)
        avg_reward = np.mean(total_rewards)
        avg_speed = np.mean(total_speeds)
        avg_collision = sum(total_collision)/num_episodes
        return avg_reward,avg_speed,avg_collision

    def save(self,EnvName, timestep,number):
        torch.save(self.actor.state_dict(), "./model/{}_actor{}_{}.pth".format(EnvName,timestep,number))
        torch.save(self.q_critic.state_dict(), "./model/{}_q_critic{}_{}.pth".format(EnvName,timestep,number))

    def load(self,EnvName, timestep,number):
        self.actor.load_state_dict(torch.load("./model/{}_actor{}_{}.pth".format(EnvName, timestep,number)))
        self.q_critic.load_state_dict(torch.load("./model/{}_q_critic{}_{}.pth".format(EnvName, timestep,number)))
        
    
    def modify_state(self, states, actions, rewards, next_states, dw, number, episode):
        device = states.device  # 确保设备一致
        state_dim = states.size(1)  # 获取状态的维度

        # 随机选择需要修改的索引和维度
        indices = random.sample(range(len(states)), number)  # 随机选择 `number` 个状态的索引
        selected_dimensions = torch.randint(0, state_dim-1, (number,))  # 随机选择需要修改的维度
        # selected_dimensions = torch.tensor([2]*number)

        # 提取选中状态和动作
        selected_states = states[indices]  # [number, state_dim]
        selected_actions = actions[indices]  # [number, action_dim]

        # 扩展状态用于计算 similarities 和 differences
        expanded_selected_states = selected_states.unsqueeze(1)  # [number, 1, state_dim]
        states_expanded = states.unsqueeze(0)  # [1, num_states, state_dim]
        if (episode % 100 == 0) and (episode > 0):
            self.state_model = StateNetwork(num_state_var=self.state_dim,
                num_action_variable=self.action_dim, params=self.params,training=False).to(self.dvc)
            self.reward_model = RewardNetwork(num_state_var=self.state_dim,
                num_action_variable=self.action_dim, params=self.params,training=False).to(self.dvc)

            # 加载权重
            state_checkpoint_path = '.\\state\\state.pth'
            reward_checkpoint_path = '.\\reward\\reward.pth'
            self.state_model.load_parameters(state_checkpoint_path, self.device)
            self.reward_model.load_parameters(reward_checkpoint_path, self.device)
            self.state_model.eval()
            self.reward_model.eval()


        # 计算相似性：排除选定维度后计算其他维度的欧几里得距离
        similarities = []
        for i, dim in enumerate(selected_dimensions):
            mask = torch.ones(state_dim, dtype=torch.bool, device=device)  # 创建一个掩码
            mask[dim] = False  # 排除选定的维度

            # 提取非选定维度的数据
            selected_state_filtered = expanded_selected_states[i, :, mask]  # [1, state_dim-1]
            states_filtered = states_expanded[0, :, mask]  # [num_states, state_dim-1]

            # 计算其他维度的欧几里得距离
            similarity = torch.norm(states_filtered - selected_state_filtered, dim=1)  # [num_states]
            similarities.append(similarity.unsqueeze(0))  # 保持批次维度

        similarities = torch.cat(similarities, dim=0)  # [number, num_states]

        # 计算差异性：仅计算选定维度上的绝对差值
        differences = []
        for i, dim in enumerate(selected_dimensions):
            diff = torch.abs(states_expanded[0, :, dim] - expanded_selected_states[i, 0, dim])  # [num_states]
            differences.append(diff.unsqueeze(0))  # 保持批次维度
        differences = torch.cat(differences, dim=0)  # [number, num_states]

        # 计算 data
        data = differences / (similarities + 1e-8)  # 防止除以 0

        # 找到最相似的状态索引
        k_indices = torch.argmax(data, dim=1)  # [number]

        # 修改选中状态的对应维度
        modified_states = selected_states.clone()
        for i in range(number):
            modified_states[i, selected_dimensions[i]] = states[k_indices[i], selected_dimensions[i]]

        # 转换为模型输入的格式
        batch_states = modified_states.to(device)
        batch_actions = selected_actions.to(device)
        # 批量预测 next_state 和 rewards
        with torch.no_grad():

            predicted_states = self.state_model(batch_states, batch_actions, training=False)
            predicted_rewards = self.reward_model(batch_states, batch_actions, training=False)

        # 批量更新 next_states 和 rewards
        
        next_states[indices] = predicted_states
        rewards[indices] = predicted_rewards

        return states, actions, rewards, next_states,dw



class ReplayBuffer():
    def __init__(self, state_dim, action_dim, max_size, dvc):
        self.max_size = max_size
        self.dvc = dvc
        self.ptr = 0
        self.size = 0

        self.s = torch.zeros((max_size, state_dim) ,dtype=torch.float,device=self.dvc)
        self.a = torch.zeros((max_size, action_dim) ,dtype=torch.float,device=self.dvc)
        self.r = torch.zeros((max_size, 1) ,dtype=torch.float,device=self.dvc)
        self.s_next = torch.zeros((max_size, state_dim) ,dtype=torch.float,device=self.dvc)
        self.dw = torch.zeros((max_size, 1) ,dtype=torch.bool,device=self.dvc)

    def add(self, s, a, r, s_next, dw):
        #每次只放入一个时刻的数据
        self.s[self.ptr] = torch.from_numpy(s).to(self.dvc)
        self.a[self.ptr] = torch.from_numpy(a).to(self.dvc) # Note that a is numpy.array
        self.r[self.ptr] = torch.tensor(r, device=self.dvc)
        self.s_next[self.ptr] = torch.from_numpy(s_next).to(self.dvc)
        self.dw[self.ptr] = dw

        self.ptr = (self.ptr + 1) % self.max_size #存满了又重头开始存
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):# 状态对取样
        ind = torch.randint(0, self.size, device=self.dvc, size=(batch_size,))
        return self.s[ind], self.a[ind], self.r[ind], self.s_next[ind], self.dw[ind]
    
    def modify_state(self, states, actions, rewards, next_states, number):
        index = random.sample(range(len(states)), number)
        out_states = []
        out_actions = []
        out_next_states = []
        out_rewards = []
        for s_t_index in index:
            selected_dimension = random.randint(0, len(states[0])-2)# 选择修改的维度
            s_t = states[s_t_index]
            similarities = torch.norm(s_t.unsqueeze(0).expand_as(states)[:, :selected_dimension] - 
                                      states[:, :selected_dimension], dim=1, p=2) + \
                           torch.norm(s_t.unsqueeze(0).expand_as(states)[:, selected_dimension+1:] - 
                                      states[:, selected_dimension+1:], dim=1, p=2)
            differences = torch.abs(s_t[selected_dimension] - states[:, selected_dimension])
            data = differences / similarities
            sorted_data, sorted_indices = torch.sort(data, descending=True)
            nan_indices = torch.isnan(sorted_data)
            first_non_nan_index = (nan_indices == False).nonzero(as_tuple=True)[0][0]
            original_indices = (data == sorted_data[first_non_nan_index]).nonzero(as_tuple=True)[0]
            k = original_indices[0].item()
            s_t_prime = s_t.clone()
            s_t_prime[selected_dimension] = states[k, selected_dimension]
            out_states.append(s_t_prime)
            out_next_states.append(next_states[s_t_index])
            out_actions.append(actions[s_t_index])
            out_rewards.append(rewards[s_t_index])
        return torch.stack(out_states), torch.stack(out_actions), torch.stack(out_rewards), torch.stack(out_next_states)
    
    def save(self, filename, save_threshold=0.01):
        """
        保存 ReplayBuffer 数据到文件
        :param filename: 保存文件路径
        :param save_threshold: 保存条件，ReplayBuffer 数据达到容量的比例时才保存 (0~1之间)
        """
        if self.size >= self.max_size * save_threshold:
            # 仅保存已使用部分的数据
            data = {
                'states': self.s[:self.size].cpu(),
                'actions': self.a[:self.size].cpu(),
                'rewards': self.r[:self.size].cpu(),
                'next_states': self.s_next[:self.size].cpu(),
                'dones': self.dw[:self.size].cpu()
            }
            torch.save(data, filename)
            print(f"Replay buffer saved to {filename}, size: {self.size}/{self.max_size}")
        else:
            print(f"Replay buffer size {self.size}/{self.max_size} does not meet the save threshold {save_threshold * self.max_size}.")

    def load(self, filename):
        """
        从文件加载 ReplayBuffer 数据
        :param filename: 加载文件路径
        """
        data = torch.load(filename)
        size = data['states'].shape[0]

        self.s[:size] = data['states'].to(self.dvc)
        self.a[:size] = data['actions'].to(self.dvc)
        self.r[:size] = data['rewards'].to(self.dvc)
        self.s_next[:size] = data['next_states'].to(self.dvc)
        self.dw[:size] = data['dones'].to(self.dvc)

        self.size = size
        self.ptr = size % self.max_size

        print(f"Replay buffer loaded from {filename}, size: {self.size}/{self.max_size}")
