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

class SetConvolution(nn.Module):
  # 集合畳み込みレイヤー(バッチ計算)
  def __init__(self, in_features, out_features, use_self=True, use_bias=False):
    super().__init__()
    self.others_weight = Parameter(torch.FloatTensor(in_features, out_features))
    self.self_weight = None if not use_self else Parameter(torch.FloatTensor(in_features, out_features))
    self.bias = None if not use_bias else Parameter(torch.FloatTensor(out_features))
    self.reset_parameters()

  def reset_parameters(self):
    for w in self.parameters():
      stdv = 1. / np.sqrt(w.size(1))
      w.data.uniform_(-stdv, stdv)

  def forward(self, x, adj=None):
    if adj is None:
      adj = to_var(1 - np.eye(x.size(1)))
    h = torch.matmul(x, self.others_weight)
    h = torch.einsum('jk,ikl->ijl', adj, h)

    if self.self_weight is not None:
      h += torch.matmul(x, self.self_weight)
    if self.bias is not None:
      h += self.bias

    return h

class MultiSetConvolution(nn.Module):
  # 等質集合畳み込みレイヤー(バッチ計算)
  def __init__(self, in_features, out_features, activation=None,
               use_self=True, use_bias=False, residual=False):
    super().__init__()
    self.cross_weight = Parameter(torch.FloatTensor(in_features, out_features))
    self.friend_weight = Parameter(torch.FloatTensor(in_features, out_features))
    self.self_weight = None if not use_self else Parameter(torch.FloatTensor(in_features, out_features))
    self.bias = None if not use_bias else Parameter(torch.FloatTensor(out_features))
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
      # (-1, N, I) x (I, O) -> (-1, 1, N, O)
      cross_list.append(torch.matmul(x, self.cross_weight).unsqueeze(1))

    out_list = []
    for i, x in enumerate(x_list):
      # 自分以外の集合
      cross_others = cross_list[:i] + cross_list[(i+1):]
      cross_sum = torch.sum(torch.cat(cross_others, dim=2), dim=2)
      cross_out = cross_sum.repeat(1, x.size(1), 1)

      adj = to_var(1 - np.eye(x.size(1))) if adj_list is None else adj_list[i]

      # (-1, N, I) x (I, O) -> (-1, N, O)
      friend_support = torch.matmul(x, self.friend_weight)
      friend_out = torch.einsum('jk,ikl->ijl', adj, friend_support)

      x_out = cross_out + friend_out

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
  h = SetConvolution(1, 1)(x)
  print(h)
  h = SetAttention(1, 1)(h)

  x = [to_var(np.random.random((3, 2, 1))), to_var(np.random.random((3, 2, 1)))]
  print(x)
  h = MultiSetConvolution(1, 1)(x)
  print(h)
  h = MultiSetAttention(1, 1)(x)
  print(h)
