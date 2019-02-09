# -*- coding: utf-8 -*-
from __future__ import division
from Networks import Generator_net, Discriminator_net
from utils import *

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

# if gpu is to be used
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor

def pretrain_model(pos_data, neg_data, inputs, embeddings, args, acc_threshold, model_name='Generator', weights=[1.0, 1.0], epochs=10):

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    model = Generator_net(embeddings, args) if model_name == 'Generator' else Discriminator_net(embeddings, args)
    model = model.cuda() if use_cuda else model

    x = pos_data+neg_data
    y = [1]*len(pos_data)+[0]*len(neg_data)
    batch_size = int(len(y)/50)+1
    trainloader = generate_trainloader(inputs, x, y, batch_size, shuf=True)

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(parameters)

    scheduler = optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.95, last_epoch=-1)

    fix_emb = 0
    for epoch in range(epochs):
        model.train()
        for i, (x, y) in enumerate(trainloader, 0):
            logits = calculate_logits(model, x, args.max_sent)
            loss = calculate_loss(logits, y, weights)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

        accuracy, accuracy_non_NA = calculate_accuracy(trainloader, model, args, epoch)

        if accuracy_non_NA > 80.0 and fix_emb == 0:
            fix_emb = 1
            model = fix_params(model)

        if accuracy_non_NA > acc_threshold and epoch >= 4:
            break

    torch.save(model.state_dict(), './models/' + model_name + '_pretrain.pkl')

    return model, model.state_dict()

def fix_params(model):
    model.embed.weight.requires_grad = False
    model.embed_pf1.weight.requires_grad = False
    model.embed_pf2.weight.requires_grad = False
    return model
