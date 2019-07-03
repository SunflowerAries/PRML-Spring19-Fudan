from __init__ import *
import torch
import torch.nn as nn
import torch.nn.functional as F

class MatchingLayer(nn.Module):
    def __init__(self, config):
        super(MatchingLayer, self).__init__()
        self.config = config

    def forward(self, *input):
        sim = torch.matmul(input[0], input[1].t())
        w1 = torch.norm(input[0], 2, 1).view(-1, 1)
        w2= torch.norm(input[1], 2, 1).view(-1, 1)
        tmp = sim / torch.matmul(w1, w2.t())
        output = F.softmax(tmp, dim=1)
        return output

class Matching(nn.Module):
    def __init__(self, config, d_in=100, d_out=10, normal=False):
        super(Matching, self).__init__()
        self.config = config
        self.d_in = d_in
        self.d_out = d_out
        self.normal = normal
        embed = nn.Embedding(num_embeddings=config.n_embed, embedding_dim=config.d_embed)
        embed.weight.data.normal_(mean=0)
        self.embedding = embed
        self.encoder = nn.Sequential(
            Transpose(),
            nn.Conv1d(in_channels=self.d_in, out_channels=self.d_out, kernel_size=5, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4)
        )

        self.classifier = MatchingLayer(self.config)

    def forward(self, *input):
        embed = self.embedding(input[0])
        embed = embed.permute(2, 0, 1)
        size = embed.size(1)
        hidden = self.encoder(embed).view(size, -1)
        size = input[1].size(0)
        support = self.embedding(input[1]).permute(2, 0, 1)
        s_hidden = self.encoder(support).view(size, -1)
        output = self.classifier(hidden, s_hidden)
        return output

class ProtoLayer(nn.Module):
    def __init__(self, config):
        super(ProtoLayer, self).__init__()
        self.config = config

    def forward(self, *input):
        sum0 = torch.zeros_like(input[1][0])
        sum1 = torch.zeros_like(input[2][0])
        size0 = input[1].size(0)
        size1 = input[2].size(0)
        for i in range(size0):
            sum0 += input[1][i]
        sum0 /= size0
        for i in range(size1):
            sum1 += input[2][i]
        sum1 /= size1
        tmp0 = input[0] - sum0
        tmp1 = input[0] - sum1

        output0 = -torch.sqrt(torch.sum(tmp0.mul(tmp0), 1))
        output1 = -torch.sqrt(torch.sum(tmp1.mul(tmp1), 1))
        output = torch.cat([output0.view(-1, 1), output1.view(-1, 1)], dim=1)
        return output

class Prototypical(nn.Module):
    def __init__(self, config, d_in=100, d_out=10, normal=False):
        super(Prototypical, self).__init__()
        self.config = config
        self.d_in = d_in
        self.d_out = d_out
        self.normal = normal
        embed = nn.Embedding(num_embeddings=config.n_embed, embedding_dim=config.d_embed)
        embed.weight.data.normal_(mean=0)
        self.embedding = embed
        self.encoder = nn.Sequential(
            Transpose(),
            nn.Conv1d(in_channels=self.d_in, out_channels=self.d_out, kernel_size=5, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4)
        )
        self.classifier = ProtoLayer(self.config)

    def forward(self, *input):
        embed = self.embedding(input[0]).permute(2, 0, 1)
        size = embed.size(1)
        hidden = self.encoder(embed).view(size, -1)
        size0 = input[1].size(0)
        size1 = input[2].size(0)
        support0 = self.embedding(input[1]).permute(2, 0, 1)
        s0_hidden = self.encoder(support0).view(size0, -1)
        support1 = self.embedding(input[2]).permute(2, 0, 1)
        s1_hidden = self.encoder(support1).view(size1, -1)
        output = self.classifier(hidden, s0_hidden, s1_hidden)
        # print(output)
        output = F.softmax(output, dim=1)
        # print(output)
        return output
