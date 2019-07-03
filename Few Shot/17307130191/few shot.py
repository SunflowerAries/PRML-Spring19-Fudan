from __init__ import *
from model import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import random

Dev_loss = []
best = 0

class Config(object):
    maxepoch = 200
    batchsize = 16
    length = 60
    d_embed = 0
    n_embed = 0
    eps = 1e-8

def testing(model, test, config, plot, kind=1):
    if plot:
        sum = 0
        hit = 0
        for batch in test.batch:
            size = batch[1]['onehot'].size(0)
            sum += size
            tar = torch.max(batch[1]['onehot'], 1)[1]
            if kind == 1:
                output = model(torch.LongTensor(batch[0]['words']), torch.LongTensor(test.support['words']))
                tmp = torch.matmul(output, torch.FloatTensor(test.support['onehot']))
                for i in range(size):
                    if tmp[i][int(tar[i])] > tmp[i][1 - int(tar[i])]:
                        hit += 1
            elif kind == 2:
                output = model(torch.LongTensor(batch[0]['words']), torch.LongTensor(test.support0['words']),
                               torch.LongTensor(test.support1['words']))
                for i in range(size):
                    if output[i][int(tar[i])] > output[i][1 - int(tar[i])]:
                        hit += 1

        Dev_loss.append(hit / sum)
        tmp = hit / sum
        if tmp > best:
            # torch.save(model.state_dict(), 'bestmatching.path')
            torch.save(model.state_dict(), 'bestproto.path')
        print(hit / sum)
    else:
        sum = 0
        hit = 0
        mymodel = Matching(config)
        # mymodel.load_state_dict(torch.load('bestmatching.path'))
        mymodel.load_state_dict(torch.load('bestproto.path'))
        for batch in test.batch:
            size = batch[1]['onehot'].size(0)
            sum += size
            tar = torch.max(batch[1]['onehot'], 1)[1]
            if kind == 1:
                output = model(torch.LongTensor(batch[0]['words']), torch.LongTensor(test.support['words']))
                tmp = torch.matmul(output, torch.FloatTensor(test.support['onehot']))
                for i in range(size):
                    if tmp[i][int(tar[i])] > tmp[i][1 - int(tar[i])]:
                        hit += 1
            elif kind == 2:
                output = model(torch.LongTensor(batch[0]['words']), torch.LongTensor(test.support0['words']),
                               torch.LongTensor(test.support1['words']))
                for i in range(size):
                    if output[i][int(tar[i])] > output[i][1 - int(tar[i])]:
                        hit += 1
        print(hit / sum)

def train():
    config = Config()
    training, dev, test, vocabsize = getdata_matching('books.', 't2', batch=config.batchsize)
    config.d_embed = 100
    config.n_embed = vocabsize
    model = Matching(config)
    optim = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    for i in range(config.maxepoch):
        epochloss = torch.tensor(0.)
        index = random.randint(0, 24)
        j = 0
        for batch in training[index].batch:
            length = len(batch[1]['onehot'].numpy())
            j += 1
            optim.zero_grad()
            output = model(torch.LongTensor(batch[0]['words']), torch.LongTensor(training[index].support['words']))
            tmp = torch.matmul(output, torch.FloatTensor(training[index].support['onehot']))
            tar = torch.max(batch[1]['onehot'], 1)[1]
            batchloss = criterion(tmp, tar) / length
            batchloss.backward()
            optim.step()
            epochloss += batchloss
        print("epoch %d:" %(i), epochloss / j)
        if i % 10 == 0:
            testing(model, dev, config, 1, kind=1)
    plt.plot(Dev_loss, label='Loss on Dev')
    plt.show()
    testing(model, test, config, 0, kind=1)

def trainer():
    config = Config()
    training, dev, test, vocabsize = getdata_proto('books.', 't2', batch=config.batchsize)
    config.d_embed = 100
    config.n_embed = vocabsize
    model = Prototypical(config)
    optim = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    for i in range(config.maxepoch):
        epochloss = torch.tensor(0.)
        index = random.randint(0, 24)
        j = 0
        for batch in training[index].batch:
            length = len(batch[1]['onehot'].numpy())
            j += 1
            optim.zero_grad()
            output = model(torch.LongTensor(batch[0]['words']), torch.LongTensor(training[index].support0['words']), torch.LongTensor(training[index].support1['words']))
            tar = torch.max(batch[1]['onehot'], 1)[1]
            # print(output, batch[1]['onehot'])
            batchloss = criterion(output, tar) / length
            batchloss.backward()
            optim.step()
            epochloss += batchloss
        print("epoch %d:" % (i), epochloss / j)
        if i % 10 == 0:
            testing(model, dev, config, 1, kind=2)
    plt.plot(Dev_loss, label='Loss on Dev')
    plt.show()
    testing(model, test, config, 0, kind=2)

#train() matching 0.9064039408866995
trainer() # proto 0.8468033775633294