import numpy as np
from tqdm.notebook import tqdm
import random
from sklearn.neural_network import MLPClassifier
from collections import Counter

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
# import nltk

from utils import *
import copy
from sklearn import tree
import sklearn

from synthetic_models import *

def train_allocation(X, y, groups, deferrer, algorithm, train, experts, sU, sL, dropout=0):
    NUM_EXPERTS = len(experts)

    df_losses, dfc_losses, clf_losses = [], [], []
    batch_size = 10
    xs = list(range(1, len(train), batch_size))

    crowdIndices, crowdFeatures, crowdLabels = [], {}, {}
    opts = [optim.SGD(list(deferrer[e].parameters()), lr=0.001) for e in range(NUM_EXPERTS)]

    for j in (range(1, len(xs))): 
        t = xs[j-1]
        train_hist = list(train[xs[j-1]:xs[j]])

        random.shuffle(train_hist)

        trainFeatures = [X[i] for i in train_hist]
        trainGroups = [int(groups[i]) for i in train_hist]
        trainLabels_true = [int(y[i]) for i in train_hist]

        outputs, inputs, expPreds = [], [], []
        loss_clf, loss_all, cost = 0, 0, 0
        loss_all = [0 for _ in range(NUM_EXPERTS)]
        cost = [0 for _ in range(NUM_EXPERTS)]

        if algorithm == "Strict-Matching":
            mu = 1
        else:
            mu = t/(t + 10000)

        for index, feat, label_true, group in zip(train_hist, trainFeatures, trainLabels_true, trainGroups):

            x = torch.Tensor(feat)
            output = [d(x) for d in deferrer]
            outputItem = [o.item() for o in output]

            outputItem = [np.clip(o, 0, 1) for o in outputItem]
            outputItem = np.array(outputItem)/sum(outputItem)

            ## Note that label_true is not used for training - just used to get synthetic annotator predictions here
            expertPreds = getExpertPredictions(experts, label_true, group, sU, sL, dropout, modified=False)

            dSim = np.array([e.dSim(group, sU, sL) for e in experts])
            outputItem = mu * outputItem + (1-mu) * dSim

            ## aggregated decision of selected annotators
            label_crowd = np.dot(outputItem, expertPreds)
            label_crowd = np.exp(label_crowd)/(np.exp(1 - label_crowd) + np.exp(label_crowd))
            label_final = int(label_crowd > 0.5)

            crowdLabels[index] = label_crowd
            crowdIndices.append(index)
            old = False

            exps = list(expertPreds)
            for i in range(len(exps)):
                correct = int(label_final==exps[i])
                loss_all[i] -= torch.log(output[i]) if correct else torch.log(1-output[i])

        total_loss = 0
        for i in range(NUM_EXPERTS):
            opts[i].zero_grad()
            loss = loss_all[i]/batch_size

            loss.backward()
            opts[i].step()        
            total_loss += loss.item()
                
    return deferrer
    
    
def test_allocation(X, y, groups, deferrer, test, experts, sU, sL, n_groups, dropout=0):
    NUM_EXPERTS = len(experts)
    
    testFeatures = [X[i] for i in test]
    testLabels = [int(y[i]) for i in test]
    testGroups = [int(groups[i]) for i in test]

    a = 0
    random_pred, random_fair_pred, joint_pred, joint_pred_fair, joint_pred_sp, joint_pred_fair_sp = [], [], [], [], [], []
    clf_pred = []
    wts, xs = [], []
    for feat, label, group in zip(testFeatures, testLabels, testGroups):
        x = torch.Tensor(feat)
        output = [d(x) for d in deferrer]
        outputItem = [o.item() for o in output]
        outputItem = np.array(outputItem)/sum(outputItem)

        expertPreds = getExpertPredictions(experts, label, group, sU, sL, dropout, modified=False)            
        joint_pred.append(getFinalPrediction3(outputItem, expertPreds))    

        wts.append([1 if j == np.max(outputItem) else 0 for j in outputItem])


    acc = (getAccuracy(joint_pred, testLabels))
    print ("\n s=", sU, "\n Overall accuracy", acc)
    print ("Group-specific accuracy")
    for n in range(n_groups):
        acc_g = (getAccuracyForGroup(joint_pred, testLabels, testGroups, n))
        print (acc_g)

    return acc, wts