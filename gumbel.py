import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from nearest_embed import NearestEmbed, NearestEmbedEMA


EPS = 1e-6

def kaiming_init(m):
    if isinstance(m, nn.Linear):
        init.kaiming_uniform_(m.weight, a=math.sqrt(3))
        if m.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(m.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(m.bias, -bound, bound)

def xavier_init(m):
    if isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)

def sample_logistic(shape, device):
    u = torch.rand(shape, dtype=torch.float32, device=device)
    u = torch.clip(u, EPS, 1 - EPS)
    return torch.log(u) - torch.log(1 - u)


def gumbel_sigmoid_batch(log_alpha, device, bs=None, tau=1, hard=False):
    if bs is None:
        shape = log_alpha.shape
    else:
        shape = tuple([bs] + list(log_alpha.size()))

    logistic_noise = sample_logistic(shape, device)
    y_soft = torch.sigmoid((log_alpha + logistic_noise) / tau)

    if hard:
        y_hard = (y_soft > 0.5).float()
        y = y_hard.detach() - y_soft.detach() + y_soft
    else:
        y = y_soft

    return y


def gumbel_sigmoid(log_alpha, device, bs=None, tau=1, hard=False):
    if bs is None or bs==1:
        shape = log_alpha.shape
    else:
        try: bs = bs[0]
        except: pass
        shape = tuple([bs] + list(log_alpha.size()))

    logistic_noise = sample_logistic(shape, device)
    y_soft = torch.sigmoid((log_alpha + logistic_noise) / tau)

    if hard:
        y_hard = (y_soft > 0.5).float()
        y = y_hard.detach() - y_soft.detach() + y_soft
    else:
        y = y_soft

    return y

class VQVAEGumbelMatrixLatent(torch.nn.Module):
    def __init__(self, params, num_state_var, num_action_var, fc_dims, device,reward=0):
        super(VQVAEGumbelMatrixLatent, self).__init__()
        self.local_mask_sampling_num = params.ours_params.local_mask_sampling_num
        self.eval_local_mask_sampling_num = params.ours_params.eval_local_mask_sampling_num
        self.num_action_var = num_action_var
        self.num_state_var = num_state_var
        self.device = device
        self.fc_dims = fc_dims
        self.reward = reward
        if self.reward == 1:# 状态因果模型和奖励因果模型的输出不同
            self.lcm_dim_1 = 1
            self.lcm_dim_2 = self.num_state_var + self.num_action_var
        else:
            self.lcm_dim_1 = self.num_state_var
            self.lcm_dim_2 = self.num_state_var + self.num_action_var
            self.lcm_dim_2 = self.lcm_dim_2 - 1
        self.adjust_dimension = self.adjust_dimension_default
        self.ours_type = params.training_params.inference_algo
        
        self.preprocess = self.preprocess_ours_mask

        self.input_dim = (self.num_state_var + self.num_action_var) * self.fc_dims[0]
        
        self.output_dim = self.lcm_dim_1 * self.lcm_dim_2#展平的矩阵

        self.ones = torch.arange(self.num_state_var)
        self.lower_inds = np.tril_indices(self.num_state_var, -1)
        self.upper_inds = np.triu_indices(self.num_state_var, 1)

        self.code_dim = params.ours_params.code_dim
        self.codebook_size = 3 #params.ours_params.codebook_size# 密码本尺寸

        enc_fc_dims = params.ours_params.vq_encode_fc_dims
        dec_fc_dims = params.ours_params.vq_decode_fc_dims
        self.state_embed = nn.Linear(self.num_state_var, self.num_state_var * self.fc_dims[0])     # 28 → 3584
        self.action_embed = nn.Linear(self.num_action_var, self.num_action_var * self.fc_dims[0])  # 1 → 128


        encs = nn.Sequential()# 定义编码器
        in_dim = self.input_dim
        for idx, out_dim in enumerate(enc_fc_dims):
            encs.add_module(f"fc_{idx}", nn.Linear(in_dim, out_dim, bias=False))
            encs.add_module(f"bn_{idx}", nn.BatchNorm1d(out_dim))
            encs.add_module(f"leaky_relu_{idx}", nn.LeakyReLU())
            in_dim = out_dim
        encs.add_module("fc_final", nn.Linear(in_dim, self.code_dim))
        self.encs = encs

        decs = nn.Sequential()# 定义解码器
        in_dim = self.code_dim
        for idx, out_dim in enumerate(dec_fc_dims):
            decs.add_module(f"fc_{idx}", nn.Linear(in_dim, out_dim, bias=False))
            decs.add_module(f"bn_{idx}", nn.BatchNorm1d(out_dim))
            decs.add_module(f"leaky_relu_{idx}", nn.LeakyReLU())
            in_dim = out_dim
        decs.add_module("fc_final", nn.Linear(in_dim, self.output_dim))
        self.decs = decs

        self.apply(kaiming_init)

        self.ema = params.ours_params.vqvae_ema
        if self.ema:# ema=True,所以最终选的是这个模型来更新密码本
            decay = params.ours_params.ema
            self.emb = NearestEmbedEMA(self.codebook_size, self.code_dim, decay=decay)
        else:
            self.emb = NearestEmbed(self.codebook_size, self.code_dim)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.reg_coef = 0.0005 # params.ours_params.reg_coef # 0.0005
        self.vq_coef = params.ours_params.vq_coef
        self.commit_coef = params.ours_params.commit_coef
        
        self.code_index = []
        # self.reset_loss()
        self.is_freeze = False

    def encode(self, x):
        return self.encs(x)
    
    def adjust_dimension_default(self, z):
        bs = z.size(0)
        if self.reward == 1:
            log_alpha = torch.zeros(bs, 1, self.num_state_var + self.num_action_var, dtype=torch.float32, device=self.device)
            z = z.reshape(bs, self.lcm_dim_1, self.lcm_dim_2)
            log_alpha=z
            # log_alpha[:, :, 1:1+self.lcm_dim_2] += torch.triu(z)
            # log_alpha[:, :, :self.lcm_dim_2] += torch.tril(z, diagonal=-1)
        else:
            log_alpha = torch.zeros(bs, self.num_state_var, self.num_state_var + self.num_action_var, dtype=torch.float32, device=self.device)
            z = z.reshape(bs, self.lcm_dim_1, self.lcm_dim_2)
            log_alpha[:, :, 1:1+self.lcm_dim_2] += torch.triu(z)
            log_alpha[:, :, :self.lcm_dim_2] += torch.tril(z, diagonal=-1)
            log_alpha[:, self.ones, self.ones] = 100

        return log_alpha

    def decode(self, z):
        return self.adjust_dimension(self.decs(z))# adjust_dimension调整维度并构造log_alpha张量
    
    def get_codebook_local_mask(self):# 这个在哪引用的，这个输入的是一个变量还是密码本中的全部变量
        return torch.sigmoid(self.decode(self.emb.weight.t()))

    def total_loss(self):
        if self.is_freeze:
            return self.get_total_loss().detach()
        else:
            return self.get_total_loss()
    
    def get_total_loss(self):
        self.reg_loss = torch.stack(self.reg_loss_list, dim=-1).float()# 稀疏损失
        self.commit_loss = torch.stack(self.commit_loss_list, dim=-1).float()# 向量化的损失
        if self.ema: 
            self.vq_loss = torch.zeros_like(self.commit_loss)
        else:
            self.vq_loss = torch.stack(self.vq_loss_list, dim=-1).float()
        return self.reg_coef*self.reg_loss.mean() + self.vq_coef*self.vq_loss.mean() + self.commit_coef*self.commit_loss.mean()

    def loss_function(self, prob, z_e, emb):
        self.learned_local_mask = []
        self.reg_loss_list = []
        self.vq_loss_list = []
        self.commit_loss_list = []
        self.code_index = []
        self.reg_loss = 0
        self.vq_loss = 0
        self.commit_loss = 0
        self.learned_local_mask.append(prob)
        self.reg_loss_list.append(prob.view(prob.size(0), -1).mean(dim=-1))
        
        if not self.ema: 
            self.vq_loss_list.append((emb - z_e.detach()).pow(2).mean())
        
        self.commit_loss_list.append((emb.detach() - z_e).pow(2).mean())
    
    def preprocess_ours_mask(self, feature, action):
        feature = F.relu(self.state_embed(feature))   # [B, 1, 3584]
        action = F.relu(self.action_embed(action))    # [B, 1, 128]
        x = torch.cat([feature, action], dim=2)       # [B, 1, 3712]
        x = x.view(x.size(0), -1)                     # [B, 3712]
        return x


    def forward_fcs(self, feature, action):
        x = self.preprocess(feature, action)
        z_e = self.encode(x)
        if self.ema:
            z_q, code_index = self.emb(z_e)
            emb = z_q.detach()
            z_q = z_e + (z_q - z_e).detach()
        else:
            z_q, code_index = self.emb(z_e, weight_sg=True)
            emb, _ = self.emb(z_e.detach())
        self.code_index.append(code_index)
        self.z_e = z_e
        return self.decode(z_q), z_e, emb
        # return z_q, z_e, emb

    def forward(self, feature, action, tau=1, drawhard=True, training=True):
        if training:
            assert self.training
            assert self.emb.training
        else:
            assert not self.training
            assert not self.emb.training
        if self.is_freeze:
            assert self.training is False
            assert self.emb.training is False
        log_alpha, z_e, emb = self.forward_fcs(feature, action)
        prob = torch.sigmoid(log_alpha)
        if training and not self.is_freeze:
            sample = gumbel_sigmoid_batch(log_alpha, self.device, bs=self.local_mask_sampling_num, tau=tau, hard=drawhard)
        else:
            sample = (prob > 0.5).float().unsqueeze(0)
        current_code_index = self.code_index[-1] if self.code_index else None
        self.loss_function(prob, z_e, emb)
        return sample, prob, current_code_index
        # return sample, prob

    

    
    
    
