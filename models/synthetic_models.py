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
import utils

### Synthetic experts design
class UniformExpert:
    def __init__(self, groups):
        self.groups = groups
    
    def prediction(self, label, group):
        pred = label if group in self.groups else utils.flip(np.abs(label-0.8))
        return pred
    
    def predictionMod(self, label, group, sU, sL):
        pred = label if group in self.groups else utils.flip(np.abs(label-0.8))
        score = sU if group in self.groups else sL
        pred = pred * score + (1-pred) * (1-score)
        return pred
    
    def dSim(self, group, sU, sL):
        score = sU if group in self.groups else sL
        return score
    
def getUniformExperts(n_groups=2):
    experts = [UniformExpert([i]) for i in range(n_groups)]    
    return experts


def getExpertCosts(num, costs=[]):
    if len(costs) == 0:
        c = 1
        costs = [c for _ in range(num)]
        costs.append(0)    
    else:
        costs = [2,1,0]
    return torch.Tensor(costs)

def getExpertPredictions(experts, label, group, sU, sL, dropout=0, modified=False):
    NUM_EXPERTS = len(experts)
    if not modified:
        expertPreds = [experts[j].prediction(label, group) for j in range(NUM_EXPERTS)]
    else:
        expertPreds = [experts[j].predictionMod(label, group, sU, sL) for j in range(NUM_EXPERTS)]
        
    expertPreds = [expertPreds[j] if utils.flip(1-dropout) else 0 for j in range(NUM_EXPERTS)]    
    return expertPreds


## Task allocation model per expert for the synthetic experiments
def getDeferrer_expert(input_size):
    model = nn.Sequential(
        nn.Linear(input_size, 5),
        nn.ReLU(),
        nn.Linear(5, 2),        
        nn.ReLU(),
        nn.Linear(2, 1),        
        nn.Sigmoid(),
    )
    
    return model

## Task allocation model for Strict-Matching
def getDeferrerWithPrior(input_size, output_size, X, groups, train, experts, sU, sL):
    NUM_EXPERTS = len(experts)
    random.shuffle(train)

    deferrer = [copy.deepcopy(getDeferrer_expert(input_size)) for _ in range(NUM_EXPERTS)]
    while True:
        deferrer = [copy.deepcopy(getDeferrer_expert(input_size)) for _ in range(NUM_EXPERTS)]
        opts = [optim.SGD(list(deferrer[e].parameters()), lr=0.5) for e in range(NUM_EXPERTS)]
        criterion = nn.MSELoss()
        batch_size = 2
        for _ in (range(500)):

            random.shuffle(train)
            train_features = [X[i] for i in train[:batch_size]]
            train_groups = [groups[i] for i in train[:batch_size]]
            prob = [list([e.dSim(g, sU, sL) for e in experts]) for g in train_groups]

            loss = [0 for _ in range(NUM_EXPERTS)]
            for feat, p in zip(train_features, prob):                
                for e in range(NUM_EXPERTS):
                    output = deferrer[e](torch.Tensor(feat))
                    loss[e] += criterion(output, torch.Tensor([p[e]]))

            for e in range(NUM_EXPERTS):
                opts[e].zero_grad()
                loss[e] = loss[e]/batch_size
                if loss[e] == 0:
                    continue
                loss[e].backward()
                opts[e].step()


        o = [deferrer[e](torch.Tensor(train_features[0])).item() for e in range(NUM_EXPERTS)]
        c1 = prob[0][train_groups[0]] - o[train_groups[0]]

        o = [deferrer[e](torch.Tensor(train_features[1])).item() for e in range(NUM_EXPERTS)]
        c2 = prob[1][train_groups[1]] - o[train_groups[1]]
        if c1 < 0.1 and c2  < 0.1:
            break

    return deferrer

