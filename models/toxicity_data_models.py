import numpy as np
from tqdm.notebook import tqdm
import random
from sklearn.neural_network import MLPClassifier
from collections import Counter
import copy

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim

import pandas as pd


# Network architecture for allocation model per expert

def getDeferrer_expert(input_size):
    model = nn.Sequential(
        nn.Linear(input_size, 1),
        nn.Sigmoid(),
    )
    return model  

## Train initial allocation model using dSim for Strict-Matching algorithm
def getInitialDeferrer(all_data, postsToFeatures, postsToIds, train):
    
    deferrer_lgbtq = copy.deepcopy(getDeferrer_expert(26))
    deferrer_aa = copy.deepcopy(getDeferrer_expert(26))
    deferrer_con = copy.deepcopy(getDeferrer_expert(26))

    opt_lgbtq = optim.SGD(list(deferrer_lgbtq.parameters()), lr=0.05)
    opt_aa = optim.SGD(list(deferrer_aa.parameters()), lr=0.05)
    opt_con = optim.SGD(list(deferrer_con.parameters()), lr=0.05)

    X = list(postsToFeatures.values())
    ps = list(postsToFeatures.keys())
    typs = []
    for i in (train[:1000]):
        feat = X[i]
        post = ps[i]

        typ =  "Control"
        typs.append(3)
        if all_data[postsToIds[post]]['black_bin'] == 1:
            typ = "AA"
            typs[-1] = 2
        if all_data[postsToIds[post]]['lgbtq_bin'] == 1:
            typ = "LGBTQ" 
            typs[-1] = 1


        feat = np.append(feat, typs[-1])

        criterion = nn.BCELoss()
        output = deferrer_lgbtq(torch.Tensor(feat))
        label = 1 if typ == "LGBTQ" else 0
        opt_lgbtq.zero_grad()
        loss = criterion(output, torch.Tensor([label]))
        loss.backward()
        opt_lgbtq.step()

        criterion = nn.BCELoss()
        output = deferrer_aa(torch.Tensor(feat))
        label = 1 if typ == "AA" else 0
        opt_aa.zero_grad()
        loss = criterion(output, torch.Tensor([label]))
    #     print (label)
        opt_aa.step()

        criterion = nn.BCELoss()
        output = deferrer_con(torch.Tensor(feat))
        label = 1 if typ == "Control" else 0
        opt_con.zero_grad()
        loss = criterion(output, torch.Tensor([label]))
        loss.backward()
        opt_con.step()
        
    return deferrer_lgbtq, deferrer_aa, deferrer_con


def getDeferrer(algorithm, experts, expert_pools, all_data, postsToFeatures, postsToIds, train):
    deferrer = {e:copy.deepcopy(getDeferrer_expert(26)) for e in experts}
    for e, model in deferrer.items():
        for param in model.parameters():
            param.data = nn.parameter.Parameter(torch.ones_like(param))    

    if algorithm == "Strict-Matching":
        deferrer_lgbtq, deferrer_aa, deferrer_con = getInitialDeferrer(all_data, postsToFeatures, postsToIds, train)
        deferrer = {}
        for e in experts:
            if expert_pools[e] == "LGBTQ":
                deferrer[e] = copy.deepcopy(deferrer_lgbtq)
            elif expert_pools[e] == "AA":
                deferrer[e] = copy.deepcopy(deferrer_lgbtq)
            elif expert_pools[e] == "Control":
                deferrer[e] = copy.deepcopy(deferrer_lgbtq)
            else:
                print ("error")
    
    return deferrer