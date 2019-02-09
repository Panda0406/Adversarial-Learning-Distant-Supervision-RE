# -*- coding: utf-8 -*-
from __future__ import division
import sys
import math
import random
import numpy as np
from copy import deepcopy
from pretrain import *
from Networks import Generator_net, Discriminator_net
from utils import *
import datetime
import cPickle as pickle

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as T
import torch.utils.data as Data

# if gpu is to be used
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor


class Denoiser(object):

    def __init__(self, args, embeddings, inputs, pos_data, neg_data):
        super(Denoiser, self).__init__()
        for k, v in vars(args).items(): setattr(self, k, v)

        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.backends.cudnn.deterministic=True

        self.args = args
        self.embeddings = embeddings
        self.inputs = inputs
        self.pos_data = pos_data
        self.neg_data = neg_data

        self.sample_negative_set()

        self.Generator = Generator_net(self.embeddings, self.args)
        self.Generator = self.Generator.cuda() if use_cuda else self.Generator

        self.Discriminator = Discriminator_net(self.embeddings, self.args)
        self.Discriminator = self.Discriminator.cuda() if use_cuda else self.Discriminator

        self.G_init_params, self.D_init_params = self.pretrain()

        for conv in self.Discriminator.convs1:
            conv.weight.requires_grad = False
        for conv in self.Generator.convs1:
            conv.weight.requires_grad = False

        self.G_parameters = filter(lambda p: p.requires_grad, self.Generator.parameters())
        self.G_optimizer = optim.Adam(self.G_parameters, lr=self.G_lr)

        if len(self.pos_data) > 10000:
            #self.ad_pos_data = self.sub_sampling(self.pos_data)
            self.ad_pos_data = random.sample(self.pos_data, 7800)
            print "NEW_pos_data has %d sentences." % (len(self.ad_pos_data))

        else:
            self.ad_pos_data = deepcopy(self.pos_data)

        self.D_pos_prob_dict, self.D_neg_prob_dict = dict(), dict()

        self.neg_criterion = self.neg_validation(self.Discriminator, self.D_neg_data)
        print 'The size of neg_criterion:', len(self.neg_criterion)

    def sub_sampling(self, dataset):

        # For the relation with too many sentence, we select part of them to do adversarial training
        data_ranked = rank_by_prob(self.Generator, dataset, self.inputs, self.max_sent, label=1)
        print "data_ranked:", len(data_ranked)
        den = int(len(dataset)/10000) + 1
        new_dataset = sample_data(data_ranked, den)

        return new_dataset

    def sample_negative_set(self):

        if len(self.pos_data) < 10000:
            nl = 50000
            neg_sample = self.neg_data[:100000]
            G_neg_data = [neg_sample[2*i] for i in range(nl)]
            G_neg_data = random.sample(G_neg_data, max(5*len(self.pos_data), 20000))
            D_neg_data = [neg_sample[2*i+1] for i in range(nl)]
            D_neg_data = random.sample(D_neg_data, max(5*len(self.pos_data), 20000))
        else:
            nl = min(5*len(self.pos_data), 150000)
            neg_sample = self.neg_data[:2*nl]
            G_neg_data = [neg_sample[2*i] for i in range(nl)]
            D_neg_data = [neg_sample[2*i+1] for i in range(nl)]

        print "Positive_total: %d, G_negative: %d, D_negative: %d" %(len(self.pos_data), len(G_neg_data), len(D_neg_data))

        self.G_neg_data = G_neg_data
        self.D_neg_data = D_neg_data

    def pretrain(self):

        # pretraining
        print '\n###### Generator Pretraining ......'
        self.Generator, G_init_params = pretrain_model(self.pos_data, self.G_neg_data, self.inputs, self.embeddings, \
                                    self.args, 92.0, model_name='Generator', weights=[1.0, 1.5], epochs=10)

        print '\n###### Discriminator Pretraining ......'
        self.Discriminator, D_init_params = pretrain_model(self.pos_data, self.D_neg_data, self.inputs, self.embeddings, \
                                    self.args, 90.0, model_name='Discriminator', epochs=8)

        return G_init_params, D_init_params

    def neg_validation(self, model, dataset, select_num=5000):

        data_ranked = rank_by_prob(model, dataset, self.inputs, self.max_sent, label=1)
        data_ranked = [item[0] for item in data_ranked]

        return data_ranked[:5000]

    def select_action(self, model, x, random=True):
        probs = calculate_prob(model, x, self.max_sent, self.inputs, detach=True)
        actions_probs = probs[:,1]
        if random:
            actions = list()
            for prob in probs:
                if abs(prob[0]-prob[1])<0.3:
                    actions.append(int(np.random.choice(np.arange(2), p=prob)))
                else:
                    actions.append(int(np.argmax(prob)))
        else:
            actions = list(np.argmax(probs, axis=1))
        return actions

    def generate_samples(self, actions, sents):
        high_conf_sents, low_conf_sents = None, None

        actions, sents = np.array(actions), np.array(sents)
        high_conf_idx = list(np.where(actions == 1)[0])
        low_conf_idx = list(np.where(actions == 0)[0])
        high_conf_sents = list(sents[high_conf_idx])
        low_conf_sents = list(sents[low_conf_idx])
        return high_conf_sents, low_conf_sents

    def select_sentences(self, model, x):
        actions = self.select_action(model, x, random=False)
        x, actions = np.array(x), np.array(actions)
        assert x.shape == actions.shape
        sents_retain = list(x[np.where(actions == 1)[0]])
        sents_remove = list(x[np.where(actions == 0)[0]])
        return sents_remove, sents_retain


    def train_Generator(self, epoch, bag_id, sent_bag, is_Training=True):
        actions = self.select_action(self.Generator, sent_bag, random=is_Training)
        high_conf_sents, low_conf_sents = self.generate_samples(actions, sent_bag)

        #the outputs of Discriminator are the rewards
        self.train_Discriminator(high_conf_sents, low_conf_sents)
        
        D_pos_probs = calculate_prob(self.Discriminator, high_conf_sents, self.max_sent, self.inputs)
        D_pos_prob = mean_probs(D_pos_probs)
        D_neg_probs = calculate_prob(self.Discriminator, self.neg_criterion, self.max_sent, self.inputs)
        D_neg_prob = mean_probs(D_neg_probs)

        if is_Training:
            D_pos_prob_list = [self.D_pos_prob_dict[item][bag_id] for item in range(epoch)]
            D_neg_prob_list = [self.D_neg_prob_dict[item][bag_id] for item in range(epoch)]
            self.D_pos_prob_dict[epoch].append(D_pos_prob)
            self.D_neg_prob_dict[epoch].append(D_neg_prob)

            if epoch > 0:
                reward_PD = self.D_pos_prob_dict[epoch][bag_id]-max(D_pos_prob_list)
                reward_ND = self.D_neg_prob_dict[epoch][bag_id]-max(D_neg_prob_list)
            else:
                reward_PD = -0.01
                reward_ND = -0.01

            reward = 100*(reward_PD+reward_ND-0.05)

            # high_confidence = hc, low_confidence = lc
            hs = torch.LongTensor(self.inputs[high_conf_sents])
            hc_logits = calculate_logits(self.Generator, hs, self.max_sent)
            y = Variable(torch.cuda.LongTensor([1]*hc_logits.size()[0]))
            loss_G = F.cross_entropy(hc_logits, y)
            loss_G = reward * loss_G

            self.G_optimizer.zero_grad()
            loss_G.backward()
            self.G_optimizer.step()

            screen = [loss_G.data[0], reward, D_pos_prob, D_neg_prob]
            return high_conf_sents, low_conf_sents, screen
        else:
            return high_conf_sents, low_conf_sents, D_neg_prob

    def train_Discriminator(self, high_conf_sents, low_conf_sents):
        bag_size = len(high_conf_sents+low_conf_sents)
        pp = float(len(low_conf_sents) / bag_size)

        self.D_optimizer.param_groups[0]['lr'] = self.D_lr * (1 - pp)
        trainloader = generate_trainloader(self.inputs, low_conf_sents, [1]*len(low_conf_sents), len(low_conf_sents), shuf=False)
        for epoch in range(self.D_epoch):
            self.Discriminator.train()
            for i, (x, y) in enumerate(trainloader, 0):
                logits = calculate_logits(self.Discriminator, x, self.max_sent)
                loss = calculate_loss(logits, y)
                self.D_optimizer.zero_grad()
                loss.backward()
                self.D_optimizer.step()

        self.D_optimizer.param_groups[0]['lr'] = self.D_lr * pp
        trainloader = generate_trainloader(self.inputs, high_conf_sents, [0]*len(high_conf_sents), len(high_conf_sents), shuf=False)
        for epoch in range(self.D_epoch):
            self.Discriminator.train()
            for i, (x, y) in enumerate(trainloader, 0):
                logits = calculate_logits(self.Discriminator, x, self.max_sent)
                loss = calculate_loss(logits, y)
                self.D_optimizer.zero_grad()
                loss.backward()
                self.D_optimizer.step()

    def adversarial_training(self):

        random.shuffle(self.ad_pos_data)
        chunk_size = int(len(self.ad_pos_data)/5) + 1
        print "Chunk Size: %d" %(chunk_size)

        Criterion_best, epoch_selected = 0.0, 0
        sents_remove_best, sents_retain_best = list(), list()
        G_best_params = None
        for epoch in range(self.max_epoch):
            self.D_pos_prob_dict[epoch], self.D_neg_prob_dict[epoch] = list(), list()
            pos_data_bags = [self.ad_pos_data[i:i+chunk_size] for i in range(0, len(self.ad_pos_data), chunk_size)]

            # TRAINING
            retain_train, remove_train = 0, 0
            self.Discriminator.load_state_dict(self.D_init_params)
            self.D_parameters = filter(lambda p: p.requires_grad, self.Discriminator.parameters())
            self.D_optimizer = optim.Adam(self.D_parameters, lr=self.D_lr)

            for bag_id in range(len(pos_data_bags)):
                retain_sents, remove_sents, screen = self.train_Generator(epoch, bag_id, pos_data_bags[bag_id])
                retain_train += len(retain_sents)
                remove_train += len(remove_sents)
            print ' '

            # VALIDATION
            retain_val, remove_val = 0, 0
            remove_total_val, retain_total_val = list(), list()
            self.Discriminator.load_state_dict(self.D_init_params)
            self.D_parameters = filter(lambda p: p.requires_grad, self.Discriminator.parameters())
            self.D_optimizer = optim.Adam(self.D_parameters, lr=self.D_lr)

            for bag_id in range(len(pos_data_bags)):
                retain_sents, remove_sents, Criterion = self.train_Generator(epoch, bag_id, pos_data_bags[bag_id], is_Training=False)
                retain_val += len(retain_sents)
                remove_val += len(remove_sents)
                retain_total_val += retain_sents
                remove_total_val += remove_sents

            print '###Epoch %d， Retain_num: %d, Remove_num: %d, Criterion: %.4f ###' % (epoch, retain_val, remove_val, Criterion)

            if Criterion > Criterion_best:
                sents_remove_best = deepcopy(remove_total_val)
                sents_retain_best = deepcopy(retain_total_val)
                epoch_selected = epoch
                Criterion_best = Criterion
                G_best = deepcopy(self.Generator)
                print 'MAX: epoch: %d, Criterion: %.4f' %(epoch_selected, Criterion_best)

            if retain_val < remove_val or epoch - epoch_selected > 30:
                print 'END'
                break

        if len(self.pos_data) > 10000:
            sents_remove_best, sents_retain_best = self.select_sentences(G_best, self.pos_data)
        print "***ACTUAL FINISH: Sent_remove: %d, Sent_retain: %d" %(len(sents_remove_best), len(sents_retain_best))

        return sents_remove_best, sents_retain_best
