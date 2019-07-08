import time, copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.autograd import Variable

def to_var(x):
    # numpy.ndarray から PyTorch の計算に渡す状態にする
    xt = torch.FloatTensor(x)
    if torch.cuda.is_available():
      xt = xt.contiguous().cuda()
    return Variable(xt)


class SetAttention(nn.Module):
    # 集合アテンション畳み込みレイヤー(バッチ計算)
    def __init__(self, in_features, out_features, use_self=True):
        super().__init__()
        self.out_features = out_features
        self.others_weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.self_weight = None if not use_self else Parameter(torch.FloatTensor(in_features, out_features))
        self.att_weight = Parameter(torch.FloatTensor(2 * out_features, 1))
        self.alpha = 0.1
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.reset_parameters()

    def reset_parameters(self):
        for w in self.parameters():
            stdv = 1. / np.sqrt(w.size(1))
            w.data.uniform_(-stdv, stdv)

    def forward(self, x, adj=None):
        if adj is None:
            adj = to_var(1 - np.eye(x.size(1)))
        h = torch.matmul(x, self.others_weight)

        N = x.size(1)
        a_input = torch.cat([
            h.repeat(1, 1, N).view(-1, N * N, self.out_features),
            h.repeat(1, N, 1),
        ], dim=1).view(-1, N, N, 2 * self.out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.att_weight).squeeze(-1))

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=-2)
        h = torch.einsum('ijk,ikl->ijl', attention, h)

        if self.self_weight is not None:
            h += torch.matmul(x, self.self_weight)

        return h

class SetAttention2(nn.Module):
    # 集合アテンション畳み込み
    # https://github.com/akurniawan/pytorch-transformer/blob/master/modules/attention.py
    def __init__(self, in_features, out_features, out_heads, residual=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.out_heads = out_heads
        assert self.out_features % self.out_heads == 0
        self.query_layer = nn.Linear(in_features, out_features, bias=False)
        self.key_layer = nn.Linear(in_features, out_features, bias=False)
        self.value_layer = nn.Linear(in_features, out_features, bias=False)
        self.proj_layer = nn.Linear(out_features, out_features)
        self.ln = nn.LayerNorm(out_features)
        self.residual = residual
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.query_layer.weight, -0.1, 0.1)
        nn.init.uniform_(self.key_layer.weight, -0.1, 0.1)
        nn.init.uniform_(self.value_layer.weight, -0.1, 0.1)
        nn.init.uniform_(self.proj_layer.weight, -0.1, 0.1)

    def forward(self, query, keys):
        """
        Args:
            query (torch.Tensor): [batch, element_size, in_features]
            keys  (torch.Tensor): [batch, element_size, in_features]
        Returns:
            torch.Tensor: [batch, element_size, out_features]
        """
        Q = self.query_layer(query)
        K = self.key_layer(keys)
        V = self.value_layer(keys)

        batch_size = query.size(0)
        element_size = query.size(1)
        chunk_size = self.out_features // self.out_heads
        Q = Q.view(batch_size * self.out_heads, element_size, chunk_size)
        K = K.view(batch_size * self.out_heads, -1, chunk_size)
        V = V.view(batch_size * self.out_heads, -1, chunk_size)

        attention = torch.bmm(Q, K.transpose(1, 2))
        print(attention.size())
        attention *= float(self.in_features) ** -0.5
        attention = F.softmax(attention, dim=-1)

        output = torch.bmm(attention, V).view(batch_size, element_size, -1)

        output = self.proj_layer(output.view(-1, output.size(-1)))
        output = output.view(batch_size, element_size, -1)

        if self.residual:
            output += query

        return self.ln(output)


class MultiSetAttention(nn.Module):
    # 等質集合畳み込みレイヤー(バッチ計算)
    def __init__(self, in_features, out_features, activation=None,
                 use_self=True, use_bias=False, residual=False):
        super().__init__()
        self.out_features = out_features
        self.cross_weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.friend_weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.self_weight = None if not use_self else Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = None if not use_bias else Parameter(torch.FloatTensor(out_features))
        self.att_weight = Parameter(torch.FloatTensor(2 * out_features, 1))
        self.alpha = 0.1
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.activation = activation
        self.residual = residual
        self.reset_parameters()

    def reset_parameters(self):
        for w in self.parameters():
            stdv = 1. / np.sqrt(w.size(1))
            w.data.uniform_(-stdv, stdv)

    def forward(self, x_list, adj_list=None):
        cross_list = []
        for x in x_list:
            # (-1, M, I) x (I, O) -> (-1, 1, M, O)
            cross_list.append(torch.matmul(x, self.cross_weight).unsqueeze(1))

        out_list = []
        for i, x in enumerate(x_list):
            # 自分以外の集合
            cross_others = cross_list[:i] + cross_list[(i+1):]
            # (-1, N, M0 + ..., O)
            cross_concat = torch.cat(cross_others, dim=2).repeat(1, x.size(1), 1, 1)

            # (-1, N, I) x (I, O) -> (-1, N, N, O)
            friend_support = torch.matmul(x, self.friend_weight).unsqueeze(1).repeat(1, x.size(1), 1, 1)
            h = torch.cat([cross_concat, friend_support], dim=2)
            print(h.size())
            N = h.size(2)
            a_input = torch.cat([
                h.repeat(1, 1, 1, N).view(-1, h.size(1), N * N, self.out_features),
                h.repeat(1, 1, N, 1),
            ], dim=1).view(-1, h.size(1), N, N, 2 * self.out_features)
            e = self.leakyrelu(torch.matmul(a_input, self.att_weight).squeeze(-1))

            zero_vec = -9e15 * torch.ones_like(e)
            # 仲間集合と他の集合全体からattentionを取る
            adj = to_var(1 - np.eye(x.size(1))) if adj_list is None else adj_list[i]
            adj_others = torch.ones((cross_concat.size(1), cross_concat.size(2)))
            adj_concat = torch.cat([adj_others, adj], dim=1)

            print(adj_concat.size(), e.size())
            attention = torch.where(adj_concat > 0, e, zero_vec)
            attention = F.softmax(attention, dim=2)
            x_out = torch.einsum('ijk,ikl->ijl', attention, h)

            if self.self_weight is not None:
                x_out += torch.matmul(x, self.self_weight)
            if self.bias is not None:
                x_out += self.bias
            if self.residual:
                x_out += x
            if self.activation is not None:
                x_out = self.activation(x_out)

            out_list.append(x_out)

        return out_list


if __name__ == '__main__':
    x = to_var(np.random.random((3, 2, 1)))
    print(x)
    h = SetAttention(1, 1)(x)
    print(h)

    h = SetAttention2(1, 4, 2)(x, x)
    print(h)

    x = [to_var(np.random.random((3, 2, 1))), to_var(np.random.random((3, 2, 1)))]
    print(x)
    h = MultiSetAttention(1, 1)(x)
    print(h)
