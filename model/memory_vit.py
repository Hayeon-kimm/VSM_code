import torch
from torch import nn
import numpy as np
import torch.distributions as td

class Memory(nn.Module):

    def __init__(self, hid_dim, conv_size=4):
        super(Memory, self).__init__()
        self.top_k = 5
        self.memory_class = 60
        self.conv_size = conv_size
        self.double_conv_size = conv_size * conv_size
        self.queue_size = 5
        print(self.memory_class)
        self.hid_dim = hid_dim
        self.device = "cuda"
        self.memory_key = (torch.zeros(self.memory_class, self.double_conv_size, self.hid_dim)).to(self.device).float()
        self.queue_key = (torch.zeros(self.memory_class, self.queue_size, self.double_conv_size, self.hid_dim)).to(self.memory_key.device).float()
        self.memory_labels = torch.ones(self.memory_class).to(self.memory_key.device)
        self.softmax = nn.Softmax(dim=-1)
        self.graph_attention_layer = nn.ModuleList([
            nn.Linear(self.hid_dim * 2, 1),
            nn.LeakyReLU(),
            nn.Softmax(dim=2)])
        self.key = nn.Linear(self.hid_dim, self.hid_dim)
        self.query = nn.Linear(self.hid_dim, self.hid_dim)

        self.key_queue = nn.Linear(self.hid_dim, self.hid_dim)
        self.query_queue = nn.Linear(self.hid_dim, self.hid_dim)

        self.prototype = nn.Linear(self.hid_dim*2, self.hid_dim)
        self.queue_total = torch.zeros(self.memory_class).to(self.memory_key.device).int()
        self.top_k_queue = 0.1

    def forward(self, h, labels, tag):
        bs = h.shape[0]
        h = h.view(bs, -1, self.hid_dim)
        key = self.memory_key.view(-1, self.hid_dim)
        key = self.key(key)
        query = self.query(h)
        weights = torch.matmul(query, key.t())
        top_attention_weights, top_attention_weights_index = weights.topk(self.top_k, dim=-1)
        top_attention_weights = self.softmax(top_attention_weights).unsqueeze(dim=2)
        key = key[top_attention_weights_index]
        memory_z = torch.matmul(top_attention_weights, key).squeeze(dim=2)
        top_attention_weights_index = (top_attention_weights_index/self.double_conv_size).to(torch.int64)
        key_queue = self.queue_key[top_attention_weights_index].view(bs*self.double_conv_size, -1, self.hid_dim)
        key_queue = self.key_queue(key_queue)
        query_queue = self.query_queue(h).view(-1, self.hid_dim)
        weights_queue = torch.matmul(query_queue.unsqueeze(dim=1), key_queue.permute(0, 2, 1)).squeeze(dim=1)
        top_queue, top_queue_index = weights_queue.topk(100, dim=-1)
        top_queue = self.softmax(top_queue).unsqueeze(dim=1)
        index = torch.range(0, bs*self.double_conv_size - 1).unsqueeze(dim=1).repeat(1, 100).long()
        key_queue = key_queue[index, top_queue_index]
        queue = torch.matmul(top_queue, key_queue).squeeze(dim=1).view(bs, self.double_conv_size, self.hid_dim)
        memory_z = torch.cat((queue, memory_z), dim=-1)
        memory_z = self.prototype(memory_z)
        return memory_z.view(bs, -1, self.hid_dim)

    def memory_update(self, a, h, labels, ori_scale):
        bs = h.shape[0]
        a = a.view(bs, -1, self.hid_dim)
        h_old = h.view(bs, -1, self.hid_dim).unsqueeze(dim=1)
        h = h.view(bs, -1, self.hid_dim)
        image_label = labels[:, 0]
        memory_index = self.memory_labels[image_label]
        memory_index = torch.nonzero(memory_index)
        self.memory_labels[image_label] = 0
        memory_keys = self.memory_key[image_label]
        memory_new_keys = self.graph_attention(h, memory_keys, memory_keys)
        memory_new_keys[memory_index] = a[memory_index]
        memory_new_keys[memory_index] = memory_new_keys[memory_index]/(1 - ori_scale)
        for i in range(a.size()[0]):
            index = self.queue_total[image_label[i]] % self.queue_size
            index = np.array(list(range(index, index + h_old.shape[1]))) % self.queue_size
            self.queue_key.data[image_label[i], index] = h_old[i]
            self.queue_total.data[image_label[i]] += h_old.shape[1]
            self.memory_key.data[image_label[i]] = (1 - ori_scale) * (memory_new_keys[i].view(-1, self.hid_dim)) + self.memory_key.data[image_label[i]] * ori_scale
        

    def graph_attention(self, q, k, v):
        k = k.unsqueeze(dim=2)
        k = k.repeat(1, 1, q.size(1), 1)
        v = v.unsqueeze(dim=2)
        v = v.repeat(1, 1, q.size(1), 1)
        q = q.unsqueeze(dim=1)
        q = q.repeat(1, k.size(1), 1, 1)
        q_k = torch.cat((q, k), dim=-1)
        weights = q_k
        for layer in self.graph_attention_layer:
            weights = layer(weights)
        att = torch.matmul(weights.permute(0, 1, 3, 2), v).squeeze(dim=2)
        return att

