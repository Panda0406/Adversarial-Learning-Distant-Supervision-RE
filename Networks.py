# -*- coding: utf-8 -*-
#!/usr/bin/env 
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Rank_CNN(nn.Module):
    def __init__(self, embeddings, args, emb_change=True):
        super(Rank_CNN,self).__init__()

        self.embed = nn.Embedding(embeddings.shape[0], embeddings.shape[1])
        self.embed.weight.data.copy_(embeddings)
        if emb_change==False:
            self.embed.weight.requires_grad = False
        self.embed_pf1 = nn.Embedding(args.pos_num, args.pos_dim)
        self.embed_pf2 = nn.Embedding(args.pos_num, args.pos_dim)
        self.convs1 = nn.ModuleList([nn.Conv2d(1, args.hidden_size, (K, args.word_dim + args.pos_dim * 2)) for K in args.window_size])
        #self.dropout = nn.Dropout(p=args.dropout_rate)
        self.fc2 = nn.Linear(len(args.window_size)*args.hidden_size, args.label_num)

    def forward(self, x, pf1, pf2):
        word = self.embed(x)
        pf1 = self.embed_pf1(pf1)
        pf2 = self.embed_pf2(pf2)

        x_all = torch.cat((word,pf1,pf2), 2)
        x_all = x_all.unsqueeze(1)
        out = [F.relu(conv(x_all)).squeeze(3) for conv in self.convs1] 
        out = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in out]
        out = torch.cat(out, 1)
        #out = self.dropout(out)
        logit = self.fc2(out)
        return logit

class Generator_net(nn.Module):
    def __init__(self, embeddings, args, emb_change=True, conv_change=True):
        super(Generator_net, self).__init__()

        self.embed = nn.Embedding(embeddings.shape[0], embeddings.shape[1])
        self.embed.weight.data.copy_(embeddings)
        self.embed_pf1 = nn.Embedding(args.pos_num, args.pos_dim)
        self.embed_pf2 = nn.Embedding(args.pos_num, args.pos_dim)
        self.convs1 = nn.ModuleList([nn.Conv2d(1, args.hidden_size, (K, args.word_dim + args.pos_dim * 2)) for K in args.window_size])
        #self.dropout = nn.Dropout(p=args.dropout_rate)
        self.fc1 = nn.Linear(len(args.window_size)*args.hidden_size, args.label_num)

        if not emb_change:
            self.embed.weight.requires_grad = False
            self.embed_pf1.weight.requires_grad = False
            self.embed_pf2.weight.requires_grad = False
        if not conv_change:
            for conv in self.convs1:
                conv.weight.requires_grad = False
        self.dropout = nn.Dropout(p=args.dropout_rate)

    def forward(self, x, pf1, pf2):
        word = self.embed(x)
        pf1 = self.embed_pf1(pf1)
        pf2 = self.embed_pf2(pf2)

        x_all = torch.cat((word,pf1,pf2), 2)
        x_all = x_all.unsqueeze(1)
        out = [F.relu(conv(x_all)).squeeze(3) for conv in self.convs1] 
        out = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in out]
        out = torch.cat(out, 1)
        #out = self.dropout(out)
        logit = self.fc1(out)
        return logit


class Discriminator_net(nn.Module):
    def __init__(self, embeddings, args, emb_change=True, conv_change=True):
        super(Discriminator_net, self).__init__()
        self.embed = nn.Embedding(embeddings.shape[0], embeddings.shape[1])
        self.embed.weight.data.copy_(embeddings)
        self.embed_pf1 = nn.Embedding(args.pos_num, args.pos_dim)
        self.embed_pf2 = nn.Embedding(args.pos_num, args.pos_dim)
        self.convs1 = nn.ModuleList([nn.Conv2d(1, args.hidden_size, (K, args.word_dim + args.pos_dim * 2)) for K in args.window_size])
        self.dropout = nn.Dropout(p=args.dropout_rate)
        self.fc1 = nn.Linear(len(args.window_size)*args.hidden_size, args.label_num)

        if not emb_change:
            self.embed.weight.requires_grad = False
            self.embed_pf1.weight.requires_grad = False
            self.embed_pf2.weight.requires_grad = False
        if not conv_change:
            for conv in self.convs1:
                conv.weight.requires_grad = False

    def forward(self, x, pf1, pf2):
        word = self.embed(x)
        pf1 = self.embed_pf1(pf1)
        pf2 = self.embed_pf2(pf2)

        x_all = torch.cat((word,pf1,pf2), 2)
        x_all = x_all.unsqueeze(1)
        out = [F.relu(conv(x_all)).squeeze(3) for conv in self.convs1] 
        out = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in out]
        out = torch.cat(out, 1)
        #out = self.dropout(out)
        logit = self.fc1(out)
        return logit
