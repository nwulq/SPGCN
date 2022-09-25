# -*- coding: utf-8 -*-

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.dynamic_rnn import DynamicLSTM

class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, text, adj):
        hidden = torch.matmul(text.to(torch.float32), self.weight)
        denom = torch.sum(adj, dim=2, keepdim=True) + 1
        output = torch.matmul(adj, hidden) / denom
        if self.bias is not None:
            return output + self.bias
        else:
            return output

class SPGCN(nn.Module):
    def __init__(self, embedding_matrix, opt):
        super(SPGCN, self).__init__()
        self.opt = opt
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        self.text_lstm = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.gc1 = GraphConvolution(2 * opt.hidden_dim, 2 * opt.hidden_dim)
        self.gc2 = GraphConvolution(2 * opt.hidden_dim, 2 * opt.hidden_dim)
        self.gc3 = GraphConvolution(2 * opt.hidden_dim, 2 * opt.hidden_dim)
        self.gc4 = GraphConvolution(2 * opt.hidden_dim, 2 * opt.hidden_dim)
        self.gc5 = GraphConvolution(2 * opt.hidden_dim, 2 * opt.hidden_dim)
        self.gc6 = GraphConvolution(2 * opt.hidden_dim, 2 * opt.hidden_dim)
        self.gc7 = GraphConvolution(2 * opt.hidden_dim, 2 * opt.hidden_dim)
        self.gc8 = GraphConvolution(2 * opt.hidden_dim, 2 * opt.hidden_dim)
        self.gc9 = GraphConvolution(2 * opt.hidden_dim, 2 * opt.hidden_dim)
        self.fc = nn.Linear(2*opt.hidden_dim, opt.polarities_dim)
        #self.dfc = nn.Linear(4*opt.hidden_dim, opt.polarities_dim)
        self.text_embed_dropout = nn.Dropout(0.6)


    def position_weight(self, x, aspect_double_idx, text_len, aspect_len):
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        aspect_double_idx = aspect_double_idx.cpu().numpy()
        text_len = text_len.cpu().numpy()
        aspect_len = aspect_len.cpu().numpy()
        weight = [[] for i in range(batch_size)]
        for i in range(batch_size):
            context_len = text_len[i] - aspect_len[i]
            for j in range(aspect_double_idx[i,0]):
                weight[i].append(1-(aspect_double_idx[i,0]-j)/context_len)
            for j in range(aspect_double_idx[i,0], aspect_double_idx[i,1]+1):
                weight[i].append(0)
            for j in range(aspect_double_idx[i,1]+1, text_len[i]):
                weight[i].append(1-(j-aspect_double_idx[i,1])/context_len)
            for j in range(text_len[i], seq_len):
                weight[i].append(0)
        weight = torch.tensor(weight).unsqueeze(2).to(self.opt.device)
        return weight*x

    def mask(self, x, aspect_double_idx):
        batch_size, seq_len = x.shape[0], x.shape[1]
        aspect_double_idx = aspect_double_idx.cpu().numpy()
        mask = [[] for i in range(batch_size)]
        for i in range(batch_size):
            for j in range(aspect_double_idx[i,0]):
                mask[i].append(0)
            for j in range(aspect_double_idx[i,0], aspect_double_idx[i,1]+1):
                mask[i].append(1)
            for j in range(aspect_double_idx[i,1]+1, seq_len):
                mask[i].append(0)
        mask = torch.tensor(mask).unsqueeze(2).float().to(self.opt.device)
        return mask*x

    def Js_div(self, p_output, q_output, get_softmax = True):
        if get_softmax:
            p_output = F.softmax(p_output)
            q_output = F.softmax(q_output)
        log_mean_output = ((p_output + q_output) / 2).log()
        return (F.kl_div(log_mean_output, p_output) + F.kl_div(log_mean_output, q_output)) / 2

    def forward(self, inputs):
        text_indices, aspect_indices, left_indices, adj, d_adj, e_adj, f_adj = inputs
        text_len = torch.sum(text_indices != 0, dim=-1)
        aspect_len = torch.sum(aspect_indices != 0, dim=-1)
        left_len = torch.sum(left_indices != 0, dim=-1)
        aspect_double_idx = torch.cat([left_len.unsqueeze(1), (left_len+aspect_len-1).unsqueeze(1)], dim=1)
        text = self.embed(text_indices)
        text = self.text_embed_dropout(text)
        text_out, (_, _) = self.text_lstm(text, text_len)
        #regular = F.pairwise_distance(e_adj, f_adj, p=2)
        regular = self.Js_div(e_adj, f_adj)
        regular = 1/(torch.sum(regular) + 1)

        x = F.relu(self.gc1(self.position_weight(text_out, aspect_double_idx, text_len, aspect_len), adj))
        x = F.relu(self.gc2(self.position_weight(x, aspect_double_idx, text_len, aspect_len), adj))
        #x = F.relu(self.gc3(self.position_weight(x, aspect_double_idx, text_len, aspect_len), adj))
        #x = F.relu(self.gc4(self.position_weight(x, aspect_double_idx, text_len, aspect_len), adj))
        #x = F.relu(self.gc5(self.position_weight(x, aspect_double_idx, text_len, aspect_len), adj))
        #x = F.relu(self.gc6(self.position_weight(x, aspect_double_idx, text_len, aspect_len), adj))
        #x = F.relu(self.gc7(self.position_weight(x, aspect_double_idx, text_len, aspect_len), adj))
        #x = F.relu(self.gc8(self.position_weight(x, aspect_double_idx, text_len, aspect_len), adj))
        #x = F.relu(self.gc2(self.position_weight(x, aspect_double_idx, text_len, aspect_len), adj))


        x_d = F.relu(self.gc1(self.position_weight(x, aspect_double_idx, text_len, aspect_len), d_adj))
        x_d = F.relu(self.gc2(self.position_weight(x_d, aspect_double_idx, text_len, aspect_len), d_adj))
        #x_d = F.relu(self.gc3(self.position_weight(x_d, aspect_double_idx, text_len, aspect_len), d_adj))
        #x_d = F.relu(self.gc4(self.position_weight(x_d, aspect_double_idx, text_len, aspect_len), d_adj))
        #x_d = F.relu(self.gc5(self.position_weight(x_d, aspect_double_idx, text_len, aspect_len), d_adj))
        #x_d = F.relu(self.gc6(self.position_weight(x_d, aspect_double_idx, text_len, aspect_len), d_adj))
        #x_d = F.relu(self.gc7(self.position_weight(x_d, aspect_double_idx, text_len, aspect_len), d_adj))
        #x_d = F.relu(self.gc8(self.position_weight(x_d, aspect_double_idx, text_len, aspect_len), d_adj))
        #x_d = F.relu(self.gc9(self.position_weight(x_d, aspect_double_idx, text_len, aspect_len), d_adj))

        x_e = F.relu(self.gc1(self.position_weight(x, aspect_double_idx, text_len, aspect_len), e_adj))
        x_e = F.relu(self.gc2(self.position_weight(x_e, aspect_double_idx, text_len, aspect_len), e_adj))
        #x_e = F.relu(self.gc3(self.position_weight(x_e, aspect_double_idx, text_len, aspect_len), e_adj))
        #x_e = F.relu(self.gc4(self.position_weight(x_e, aspect_double_idx, text_len, aspect_len), e_adj))
        #x_e = F.relu(self.gc5(self.position_weight(x_e, aspect_double_idx, text_len, aspect_len), e_adj))
        #x_e = F.relu(self.gc6(self.position_weight(x_e, aspect_double_idx, text_len, aspect_len), e_adj))
        #x_e = F.relu(self.gc7(self.position_weight(x_e, aspect_double_idx, text_len, aspect_len), e_adj))
        #x_e = F.relu(self.gc8(self.position_weight(x_e, aspect_double_idx, text_len, aspect_len), e_adj))
        #x_e = F.relu(self.gc9(self.position_weight(x_e, aspect_double_idx, text_len, aspect_len), e_adj))

        x_f = F.relu(self.gc1(self.position_weight(x, aspect_double_idx, text_len, aspect_len), f_adj))
        x_f = F.relu(self.gc2(self.position_weight(x_f, aspect_double_idx, text_len, aspect_len), f_adj))
        #x_f = F.relu(self.gc3(self.position_weight(x_f, aspect_double_idx, text_len, aspect_len), f_adj))
        #x_f = F.relu(self.gc4(self.position_weight(x_f, aspect_double_idx, text_len, aspect_len), f_adj))
        #x_f = F.relu(self.gc5(self.position_weight(x_f, aspect_double_idx, text_len, aspect_len), f_adj))
        #x_f = F.relu(self.gc6(self.position_weight(x_f, aspect_double_idx, text_len, aspect_len), f_adj))
        #x_f = F.relu(self.gc7(self.position_weight(x_f, aspect_double_idx, text_len, aspect_len), e_adj))
        #x_f = F.relu(self.gc8(self.position_weight(x_f, aspect_double_idx, text_len, aspect_len), e_adj))
        #x_f = F.relu(self.gc9(self.position_weight(x_f, aspect_double_idx, text_len, aspect_len), e_adj))

        x_dd = F.relu(self.gc3(self.position_weight(x, aspect_double_idx, text_len, aspect_len), d_adj))
        x_dd = F.relu(self.gc4(self.position_weight(x_dd, aspect_double_idx, text_len, aspect_len), d_adj))
        
        x_de = F.relu(self.gc1(self.position_weight(x_dd, aspect_double_idx, text_len, aspect_len), e_adj))
        x_de = F.relu(self.gc2(self.position_weight(x_de, aspect_double_idx, text_len, aspect_len), e_adj))
        
        x_ff = F.relu(self.gc1(self.position_weight(x_de, aspect_double_idx, text_len, aspect_len), f_adj))
        x_ff = F.relu(self.gc2(self.position_weight(x_ff, aspect_double_idx, text_len, aspect_len), f_adj))

    
        x = x + 0.2 * x_d + regular * (x_e + x_f)
        x = self.mask(x, aspect_double_idx)
        alpha_mat = torch.matmul(x, text_out.transpose(1, 2))
        alpha = F.softmax(alpha_mat.sum(1, keepdim=True), dim=2)
        x = torch.matmul(alpha, text_out).squeeze(1)

        output = self.fc(x)
        return output
