#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 10:28:56 2023

@author: adan
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data 
from torch.autograd import Variable

#%%
# Import dataset

movies = pd.read_csv('./ml-1m/movies.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
users = pd.read_csv('./ml-1m/users.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
ratings = pd.read_csv('./ml-1m/ratings.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')


#%%
# Preparing training and test dataset

train_set = pd.read_csv('./ml-100k/u1.base', delimiter = '\t')
train_set = np.array(train_set, dtype = 'int')

test_set = pd.read_csv('./ml-100k/u1.test', delimiter = '\t')
test_set = np.array(test_set, dtype = 'int')

#%%
# Getting number of users and movies

nb_users = int(max((max(train_set[:,0]),max(test_set[:,0]))))
nb_movies = int(max((max(train_set[:,1]),max(test_set[:,1]))))

#%%
# Convert into user - movies matrix

def convert(data):
    new_data = []
    for id_users in range(1,nb_users + 1): #range doesn't take last bound
        id_movies = data[:,1][data[:,0] == id_users]
        id_ratings = data[:,2][data[:,0] == id_users]
        ratings = np.zeros(nb_movies)
        ratings[id_movies - 1] = id_ratings # adjust to account for first index eq. 0
        new_data.append(list(ratings))
    return new_data
    
train_set = convert(train_set)
test_set = convert(test_set)


#%%
# Convert into pytorch tensors

train_set = torch.FloatTensor(train_set)
test_set = torch.FloatTensor(test_set)

#%%
# transform ratings into binary 1-liked / 0-didn't liked

train_set[train_set == 0] = -1 # non rated movies transform into -1
train_set[train_set == 1] = 0 # and-or don't work for pythorch
train_set[train_set == 2] = 0
train_set[train_set >= 3] = 1

test_set[test_set == 0] = -1 # non rated movies transform into -1
test_set[test_set == 1] = 0 # and-or don't work for pythorch
test_set[test_set == 2] = 0
test_set[test_set >= 3] = 1


#%%
# create architecturee of neural network

# using a class allows set architechture and reuse it
class RBM():
    
    def __init__(self, nv, nh):
        self.W = torch.randn(nh, nv) # initializes weight for visible nodes (nv) and hidden nodes (nh)
        self.a = torch.randn(1, nh) # initialize biases for hidden nodes with the added dimension for batch
        self.b = torch.randn(1, nv) # initialize biases for visible nodes with the added dimension for batch
        
    def sample_h(self, x): # sampling for Gibbs algorithm in hidden layer x will correspond to observations for visible nodes
        wx = torch.mm(x, self.W.t()) # product of two torch tensors x and weights (need to be transposed)
        activation = wx + self.a.expand_as(wx) # biases are expanded for the batches with expand_as
        p_h_given_v = torch.sigmoid(activation)
        return p_h_given_v, torch.bernoulli(p_h_given_v) # we simulate a layer for the hidden nodes with 0 if inactive and 1 for activated neurons
    
    def sample_v(self, y): # sampling for Gibbs algorithm in hidden layer y will correspond to observations for hidden nodes
        wy = torch.mm(y, self.W) # product of two torch tensors y and weights (no tansposing this time)
        activation = wy + self.b.expand_as(wy) # biases are expanded for the batches with expand_as
        p_v_given_h = torch.sigmoid(activation)
        return p_v_given_h, torch.bernoulli(p_v_given_h) # we simulate a layer for the visible nodes with 0 if inactive and 1 for activated neurons
    
    def train(self, v0 , vk, ph0, phk): # class with CONTRASTIVE DIVERGENCE, since it's an energy based (probabilistic graphical) model we try to minimize the energy (maximize log-likelihood) 
        # we execute the Gibbs algorithm as the MCMC ([v0 < 0]Markov Chain Monte Carlo) type aproximation to the gradient
        # v0 is the initial visible nodes and ph0 the activation probability given v0
        # vk is the initial visible nodes and ph0 the activation probability given vk
        self.W += (torch.mm(v0.t(), ph0) - torch.mm(vk.t(), phk)).t()
        self.b += torch.sum((v0 - vk), 0) # keeps format for batches dimension in the tensor
        self.a += torch.sum((ph0 - phk), 0) # keeps format for batches dimension in the tensor
        
# create RBM object
nv = len(train_set[0])
nh = 100 # features the model learns in the hidden nodes
batch_size = 100
rbm = RBM(nv, nh)


#%%
# Train the network

nb_epoch = 10

for epoch in range(1, nb_epoch +1):
    train_loss = 0
    s = 0. # counter after each epoch, float to divide normalizing loss
    
    for id_user in range(0, nb_users - batch_size, batch_size):
        vk = train_set[id_user:id_user + batch_size] # these will be updated with the simulation
        v0 = train_set[id_user:id_user + batch_size] 
        ph0,_ = rbm.sample_h(v0) # ,_ to take only first arg returned
        
        for k in range(10): # random walk that updates v0 
            _,hk = rbm.sample_h(vk) # _, takes only second argument
            _,vk = rbm.sample_v(hk) # updates vk of the random walk
            vk[v0 < 0] = v0[v0 < 0] # does not changes the nodes without observations
            
        phk,_ = rbm.sample_h(vk) # probability of the final state for the visible simulation
        rbm.train(v0, vk, ph0, phk)
        train_loss += torch.mean(torch.abs(v0[v0 >= 0] - vk[v0 >= 0])) # uses abs distance to measure loss
        # train_loss += np.sqrt(torch.mean((v0[v0>=0] - vk[v0>=0])**2)) # RMSE here
        
        s += 1. # float to use to divide
    print('epoch: ' + str(epoch) + ' loss: ' + str(train_loss/s))


#%%
# test RBM

test_loss = 0 
s = 0.

for id_user in range(nb_users):
    v = train_set[id_user:id_user + 1] # not needed to change to test because we'll use its hidden nodes to update vt
    vt = test_set[id_user:id_user + 1] 
    
    # we need only 1 step of the blind walk (random walk with updating probabilities)
    if len(vt[vt >= 0]) > 0:
        _,h = rbm.sample_h(v) 
        _,v = rbm.sample_v(h) # gets target of blind walk

        test_loss += torch.mean(torch.abs(vt[vt >= 0] - v[vt >= 0])) # uses abs distance to measure loss
        # test_loss += np.sqrt(torch.mean((vt[vt>=0] - v[vt>=0])**2)) # RMSE here
        s += 1. # float to use to divide
print('Test loss: ' + str(test_loss/s))



