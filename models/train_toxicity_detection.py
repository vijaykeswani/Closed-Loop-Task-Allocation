import numpy as np
from tqdm.notebook import tqdm
import random
from sklearn.neural_network import MLPClassifier
from collections import Counter

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
import sklearn
import math

from utils import *
import copy
from sklearn import tree
from sklearn.metrics import fbeta_score
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from scipy.spatial.distance import cosine
from sklearn.linear_model import LogisticRegression
import copy
import pandas as pd

from toxicity_data_models import *

def train_allocation(algorithm, train, deferrer, experts, expert_pools, postsToLabels, postsToIds, X, ps, all_data, spec_data):
    batch_size = 1000
    xs = list(range(1, len(train), batch_size))
    train_acc, train_roc = [], []
    final_labels = []

    for j in (range(1, len(xs))): 

        opts = {e: optim.SGD(list(deferrer[e].parameters()), lr=0.3) for e in experts}
        train_datasets = {e:([], []) for e in experts}

        t = xs[j-1]
        train_hist = list(train[xs[j-1]:xs[j]])
        random.shuffle(train_hist)

        B = 32
        trainFeatures = [X[i] for i in train_hist[:B]]
        trainPosts = [ps[i] for i in train_hist[:B]]
        trainLabels_true = np.array([int(postsToLabels[ps[i]]) for i in train_hist[:B]])

        loss_all, cost = {e:0 for e in experts}, 0
        num_all = {e:0 for e in experts}
        acc = []
        
        for index, post, feat, label_true in (zip(train_hist, trainPosts, trainFeatures, trainLabels_true)):
            if algorithm == "Strict-Matching":
                mu = 1
            else:
                mu = t/(t + 100000) 

            exps, dems = {}, {}
            rows = spec_data[spec_data['id'] == postsToIds[post]]
            
            preds = [int(p < 0) if not np.isnan(p) else 0 for p in list(rows["identity_attack"])]
            ann_ids = list(rows["unique_contributor_id"])
            exps = dict(zip(ann_ids, preds))
            
            expertPreds, dSim = [], []
            for e in exps.keys():
                expertPreds.append(exps[e])


            typ =  "Control"
            typ_num = 3
            if all_data[postsToIds[post]]['lgbtq_bin'] == 1:
                typ = "LGBTQ" 
                typ_num = 1
            elif all_data[postsToIds[post]]['black_bin'] == 1:
                typ = "AA"
                typ_num = 2
                
            output = []
            feat = np.append(feat, typ_num)
            x = torch.Tensor(feat)
            for e in exps.keys():
                output.append(deferrer[e](x))
                
                group = expert_pools[e]
                if typ == group:
                    dSim.append(1)
                else:
                    dSim.append(0)

            outputItem = [o.item() for o in output]
            outputItem = np.array(outputItem)/sum(outputItem)
            dSim = np.array(dSim)/sum(dSim) if sum(dSim)!=0 else np.array([1/len(exps.keys()) for _ in exps.keys()])

            outputItem = mu * outputItem + (1-mu) * dSim
            if sum(outputItem) == 0:
                continue

            committee = np.random.choice(list(range(len(exps))), p=outputItem, size=7)
            label_final = np.mean([expertPreds[i] for i in committee]) > 0.5

            for i in committee:
                correct = int(label_final==expertPreds[i])
                loss_all[list(exps.keys())[i]] -= torch.log(output[i]) if correct else torch.log(1-output[i])
                num_all[list(exps.keys())[i]] += 1

            final_labels.append(label_final)
            acc.append(label_final == label_true)

        train_acc.append(np.mean(acc))
        
        total_loss = 0
        for e in experts:
            opts[e].zero_grad()
            if num_all[e] == 0:
                continue

            loss = loss_all[e]/num_all[e]

            loss.backward()
            opts[e].step()        
            total_loss += loss.item()

    return deferrer
    
    
def test_allocation(test, deferrer, postsToLabels, postsToIds, X, ps, all_data, spec_data):
    testLabels_true = np.array([int(postsToLabels[ps[i]]) for i in test])

    random_pred, joint_pred, types = [], [], []
    joint_pred_by_k = {k:[] for k in range(1,8)}
    clf_preds = []
    for i in (test):
        feat = X[i]
        post = ps[i]
        typ =  "Control"
        if all_data[postsToIds[post]]['lgbtq_bin'] == 1:
            typ = "LGBTQ" 
            types.append(1)
        elif all_data[postsToIds[post]]['black_bin'] == 1:
            typ = "AA"
            types.append(2)
        else:
            types.append(3)
            

        feat = np.append(feat, types[-1])
        x = torch.Tensor(feat)

        rows = spec_data[spec_data['id'] == postsToIds[post]]

        preds = [int(p < 0) if not np.isnan(p) else 0 for p in list(rows["identity_attack"])]
        ann_ids = list(rows["unique_contributor_id"])
        exps = dict(zip(ann_ids, preds))

        expertPreds, dSim = [], []
        for e in exps.keys():
            expertPreds.append(exps[e])

        output = []
        for e in exps.keys():
            output.append(deferrer[e](x).item())

        output = np.array(output)/sum(output)

        label_final = expertPreds[np.argmax(output)]
        joint_pred.append(label_final)


    acc = (getAccuracy(joint_pred, testLabels_true))
    f1 = (recall_score(testLabels_true, joint_pred))
    roc = (roc_auc_score(testLabels_true, joint_pred))

    print ("Overall", acc, roc, f1)
    for i in range(1,4):
        roc_i = roc_auc_score([testLabels_true[j] for j in range(len(testLabels_true)) if types[j] == i], [joint_pred[j] for j in range(len(testLabels_true)) if types[j] == i])
#         rocs_group[i-1].append(roc_i)
        print ("Accuracy - Type", i, np.mean([joint_pred[j] == testLabels_true[j] for j in range(len(testLabels_true)) if types[j] == i]))
        print ("ROC - Type", i, roc_i)
        print ("F1 - Type", i, f1_score([testLabels_true[j] for j in range(len(testLabels_true)) if types[j] == i], [joint_pred[j] for j in range(len(testLabels_true)) if types[j] == i]))
    
    return acc, f1, roc