import os
from fastNLP import DataSet
from fastNLP import Instance
from fastNLP import Vocabulary
from fastNLP import BucketSampler
from fastNLP import Batch
import torch
import torch.nn as nn
import numpy as np
import re
import string
import random

os.chdir(os.getcwd() + '/Amazon_few_shot')
Filelist = os.listdir(os.getcwd())
test = ['books.', 'dvd.', 'electronics.', 'kitchen_housewares.']

raw2 = []
raw3 = []

def pre(x, limit):
    data = re.sub(r'['+string.whitespace+']+', ' ', x)
    data = re.split(r' +', data.strip())
    length = len(data)
    if(length < limit):
        tmp = [' '] * limit
        tmp[:length] = data
    else:
        tmp = data[:limit]
    return tmp

class Pair():
    def __init__(self, x, y):
        self.batch = x
        self.support = y

class Triple():
    def __init__(self, x, y, z):
        self.batch = x
        self.support0 = y
        self.support1 = z

def preprocess(task, type, length=60):
    Task = []
    i = 0
    index = 0
    ind = []
    raw = []
    for File in filter(lambda s: True if s.find('.' + type + '.train') != -1 else False, Filelist):
        raw1 = []
        j = 0
        with open(File, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                j += 1
                if line[-3] == '-':
                    label = -1
                    onehot = [1, 0]
                else:
                    label = 1
                    onehot = [0, 1]
                raw1.append(Instance(sentence = line[:-3], label = label, onehot=onehot))
                raw.append(Instance(sentence = line[:-3], label = label, onehot=onehot))
        raw1 = DataSet(raw1)
        raw1.apply(lambda x: pre(x['sentence'], length), new_field_name='words')
        if File in [task + type + '.train' for task in test]:
            index = i
        if j <= 30:
            ind.append(i)
        i += 1
        Task.append(raw1)

    raw3 = []
    for File in filter(lambda s: True if s == task + type + '.dev' else False, Filelist):
        with open(File, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                if line[-3] == '-':
                    label = -1
                    onehot = [1, 0]
                else:
                    label = 1
                    onehot = [0, 1]
                raw3.append(Instance(sentence=line[:-3], label=label, onehot=onehot))
    raw3 = DataSet(raw3)
    raw3.apply(lambda x: pre(x['sentence'], length), new_field_name='words')

    raw2 = []
    for File in filter(lambda s: True if s == task + type + '.test' else False, Filelist):
        raw2 = []
        with open(File, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                if line[-3] == '-':
                    label = -1
                    onehot = [1, 0]
                else:
                    label = 1
                    onehot = [0, 1]
                raw2.append(Instance(sentence=line[:-3], label=label, onehot=onehot))
    raw2 = DataSet(raw2)
    raw2.apply(lambda x: pre(x['sentence'], length), new_field_name='words')

    raw = DataSet(raw)
    raw.apply(lambda x: pre(x['sentence'], length), new_field_name='words')
    vocab = Vocabulary(min_freq=2).from_dataset(raw, raw3, raw2, field_name='words')
    vocab.index_dataset(raw2, raw3, field_name='words', new_field_name='words')
    return Task, vocab, ind, index, raw3, raw2

def getdata_matching(task, type, batch=4):
    Task, vocab, ind, index, Devset, Testset = preprocess(task, type)
    Train = []
    global Test
    j = 0
    for i in range(len(Task)):
        vocab.index_dataset(Task[i], field_name='words', new_field_name='words')
        if i in ind:
            list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            slice = random.sample(list, 4)
            another = [x for x in list if x not in slice]
            train = Task[i][another]
            support = Task[i][slice]
        else:
            length = len(Task[i])
            list = [x - 1 for x in range(length)]
            slice = random.sample(list, 20)
            another = [x for x in list if x not in slice]
            train, support = Task[i][another], Task[i][slice]
        if i == index:
            Test = Pair(Testset, support)
            Dev = Pair(Devset, support)
        Train.append(Pair(train, support))

    for i in range(len(Train)):
        Train[i].batch.set_input('words')
        Train[i].support.set_input('words')
        Train[i].batch.set_target('onehot')
        Train[i].support.set_target('onehot')
        Train[i].batch.apply(lambda x: len(x['words']), new_field_name='seq_len')
        Train[i].support.apply(lambda x: len(x['words']), new_field_name='seq_len')

    Test.batch.set_input('words')
    Test.support.set_input('words')
    Test.batch.set_target('onehot')
    Test.support.set_target('onehot')
    Test.batch.apply(lambda x: len(x['words']), new_field_name='seq_len')
    Test.support.apply(lambda x: len(x['words']), new_field_name='seq_len')

    Dev.batch.set_input('words')
    Dev.support.set_input('words')
    Dev.batch.set_target('onehot')
    Dev.support.set_target('onehot')
    Dev.batch.apply(lambda x: len(x['words']), new_field_name='seq_len')
    Dev.support.apply(lambda x: len(x['words']), new_field_name='seq_len')

    Train_batch = []
    for i in range(len(Train)):
        if i in ind:
            sampler = BucketSampler(num_buckets=1, batch_size=batch, seq_len_field_name='seq_len')
            Train_batch.append(Pair(Batch(batch_size=batch, dataset=Train[i].batch, sampler=sampler), Train[i].support))
        else:
            sampler = BucketSampler(batch_size=batch, seq_len_field_name='seq_len')
            Train_batch.append(Pair(Batch(batch_size=batch, dataset=Train[i].batch, sampler=sampler), Train[i].support))

    sampler = BucketSampler(batch_size=batch, seq_len_field_name='seq_len')
    Test_batch = Pair(Batch(batch_size=batch, dataset=Test.batch, sampler=sampler), Test.support)
    Dev_batch = Pair(Batch(batch_size=batch, dataset=Dev.batch, sampler=sampler), Dev.support)
    return Train_batch, Dev_batch, Test_batch, len(vocab)

def getdata_proto(task, type, batch=4):
    Task, vocab, ind, index, testset, devset = preprocess(task, type)
    Train = []
    global Test
    for i in range(len(Task)):
        vocab.index_dataset(Task[i], field_name='words', new_field_name='words')
        if i in ind:
            list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            rawsupport0 = []
            rawsupport1 = []
            while (len(rawsupport0) == 0 or len(rawsupport1) == 0):
                slice = random.sample(list, 4)
                another = [x for x in list if x not in slice]
                train = Task[i][another]

                for inn in slice:
                    if Task[i][inn]['label'] == -1:
                        rawsupport0.append(inn)
                    else:
                        rawsupport1.append(inn)
            support0 = Task[i][rawsupport0]
            support1 = Task[i][rawsupport1]
        else:
            length = len(Task[i])
            list = [x - 1 for x in range(length)]
            rawsupport0 = []
            rawsupport1 = []
            while (len(rawsupport0) == 0 or len(rawsupport1) == 0):
                slice = random.sample(list, 20)
                another = [x for x in list if x not in slice]
                train = Task[i][another]

                for inn in slice:
                    if Task[i][inn]['label'] == -1:
                        rawsupport0.append(inn)
                    else:
                        rawsupport1.append(inn)

            support0 = Task[i][rawsupport0]
            support1 = Task[i][rawsupport1]
        if i == index:
            Test = Triple(testset, support0, support1)
            Dev = Triple(devset, support0, support1)
        Train.append(Triple(train, support0, support1))

    for i in range(len(Train)):
        Train[i].batch.set_input('words')
        Train[i].support0.set_input('words')
        Train[i].support1.set_input('words')
        Train[i].batch.set_target('onehot')
        Train[i].support0.set_target('onehot')
        Train[i].support1.set_target('onehot')
        Train[i].batch.apply(lambda x: len(x['words']), new_field_name='seq_len')
        Train[i].support0.apply(lambda x: len(x['words']), new_field_name='seq_len')
        Train[i].support1.apply(lambda x: len(x['words']), new_field_name='seq_len')

    Test.batch.set_input('words')
    Test.support0.set_input('words')
    Test.support1.set_input('words')
    Test.batch.set_target('onehot')
    Test.support0.set_target('onehot')
    Test.support1.set_target('onehot')
    Test.batch.apply(lambda x: len(x['words']), new_field_name='seq_len')
    Test.support0.apply(lambda x: len(x['words']), new_field_name='seq_len')
    Test.support1.apply(lambda x: len(x['words']), new_field_name='seq_len')

    Dev.batch.set_input('words')
    Dev.support0.set_input('words')
    Dev.support1.set_input('words')
    Dev.batch.set_target('onehot')
    Dev.support0.set_target('onehot')
    Dev.support1.set_target('onehot')
    Dev.batch.apply(lambda x: len(x['words']), new_field_name='seq_len')
    Dev.support0.apply(lambda x: len(x['words']), new_field_name='seq_len')
    Dev.support1.apply(lambda x: len(x['words']), new_field_name='seq_len')

    Train_batch = []
    for i in range(len(Train)):
        if i in ind:
            sampler = BucketSampler(num_buckets=1, batch_size=batch, seq_len_field_name='seq_len')
            Train_batch.append(Triple(Batch(batch_size=batch, dataset=Train[i].batch, sampler=sampler), Train[i].support0, Train[i].support1))
        else:
            sampler = BucketSampler(batch_size=batch, seq_len_field_name='seq_len')
            Train_batch.append(Triple(Batch(batch_size=batch, dataset=Train[i].batch, sampler=sampler), Train[i].support0, Train[i].support1))

    sampler = BucketSampler(batch_size=batch, seq_len_field_name='seq_len')
    Test_batch = Triple(Batch(batch_size=batch, dataset=Test.batch, sampler=sampler), Test.support0, Test.support1)
    Dev_batch = Triple(Batch(batch_size=batch, dataset=Dev.batch, sampler=sampler), Dev.support0, Dev.support1)
    return Train_batch, Dev_batch, Test_batch, len(vocab)

class Transpose(nn.Module):
    def __init__(self, dim1=0, dim2=1):
        super(Transpose, self).__init__()
        self.dim1 = dim1
        self.dim2 = dim2

    def forward(self, input):
        return input.transpose(self.dim1, self.dim2).contiguous()

class View(nn.Module):
    def __init__(self, *size):
        super(View, self).__init__()
        self.size = size

    def forward(self, input):
        return input.view(*self.size)

class CnnEncoder(nn.Module):
    def __init__(self, config, d_in, d_out, padding=1, normal=False):
        super(CnnEncoder, self).__init__()
        self.config = config
        self.d_in = d_in
        self.d_out = d_out
        self.padding = padding
        self.normal = normal

        conv = nn.Conv1d(self.d_in, self.d_out, kernel_size=5,padding=self.padding)
        if self.normal:
            conv.weight.data.normal_(mean=0, std=0.1)

        self.cnn = nn.Sequential(
            ('cnn', conv),
            ('relu', nn.ReLU()),
            ('max', nn.MaxPool2d(kernel_size=5)) #TODO
        )

    def forward(self, input):
        output = self.cnn.forward(input)
        return output