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


def flip(p):
    return 1 if random.random() < p else 0

def getAccuracy(predictions, labels):
    return np.mean(np.array(predictions) == labels)

def getAccuracyForGroup(predictions, labels, groups, g):
    predictions = [predictions[i]  for i in range(len(predictions)) if groups[i] == g]
    labels = [labels[i]  for i in range(len(labels)) if groups[i] == g]
    
    return np.mean(np.array(predictions) == labels)

## Generate synthetic dataset 
def getSyntheticDataset(total=1000, n_groups=3):
    X, y, groups = [], [], []
    N = int(total/n_groups)
    d = 2
    mu = np.array([random.random() for _ in range(d)])
    sig = [[0 for _ in range(d)] for _ in range(d)]
    
    for i in (range(d)):
        sig[i][i] = random.random()
            
    for g in range(n_groups):
        X = X + list(np.random.multivariate_normal(mu+2.5*(g), sig, N))
        for _ in range(N):
            label = flip(0.5)
            y.append(label)
            groups.append(g)     
    
    return X, y, groups, mu, sig

def getPartition(frac=0.7, total=1000):
    indices = list(range(total))
    N = int(total * frac)
    
    random.shuffle(indices)
    train = indices[:N]
    test = indices[N:]
    
    return train, test

def getCvPartition(postsToFeatures, fold):
    indices = list(range(len(postsToFeatures)))
    N = int(len(indices)/5)
    
    part = []
    part.append(indices[:N])
    part.append(indices[N:N*2])
    part.append(indices[N*2:N*3])
    part.append(indices[N*3:N*4])
    part.append(indices[N*4:])
    
    train = []
    for j in range(5):
        if j == fold:
            continue
        train = train + part[j]
        
    return train, part[fold]

def getDictPartition(postsToFeatures):
    indices = list(range(len(postsToFeatures)))
    N = int(len(indices)/5)
    random.shuffle(indices)    
    test = indices[:N]
    train = indices[N:]
            
    return train, test


## Different ways of aggregating annotator decisions 
def getFinalPrediction(output, expertPreds):
    output = output.tolist()
    return np.dot(output, expertPreds) > 0.5

def getFinalPrediction2(output, expertPreds):
    output = output.tolist()
    output2 = np.dot(output, expertPreds)
    output2 = np.exp(output2)/(np.exp(1 - output2) + np.exp(output2))
    return output2 > 0.5

def getFinalPrediction3(output, expertPreds):
    output = output.tolist()
    return expertPreds[np.argmax(output)]

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

# Loading the pretrained Glove model
def loadGloveModel(File):
    print("Loading Glove Model")
    f = open(File,'r')
    gloveModel = {}
    for line in f:
        splitLines = line.split()
        word = splitLines[0]
        wordEmbedding = np.array([float(value) for value in splitLines[1:]])
        gloveModel[word] = wordEmbedding
    return gloveModel


### Preprocess and organize data
def getJigsawData(jigsaw_loc, spec_raters_loc):

    all_data_pd = pd.read_csv(jigsaw_loc)

    spec_data = pd.read_csv(spec_raters_loc)
    spec_data = spec_data.replace("African American", "AA")

    all_data = {}
    ids = spec_data['id'].unique()
    for id in (ids):
        row = all_data_pd[all_data_pd['id'] == id]
        text = str(row['comment_text'].item())
        score = float(row['identity_attack'])
        trans = float(row['transgender'])
        lg = float(row['homosexual_gay_or_lesbian'])
        bi = float(row['bisexual'])
        other_or = float(row['other_sexual_orientation'])
        other_gender = float(row['other_gender'])
    #     print (row['black'])
        black = float(row['black'])
        data = {"text": text, "toxicity": score, "trans": trans, "homosexual_gay_or_lesbian": lg,
               "other_sexual_orientation": other_or, "black": black,
               "other_gender": other_gender, "bisexual": bi}

        all_data[id] = dict(data)

    for id, item in (all_data.items()):
        item['toxicity_binary'] = int(item['toxicity'] >= 0.5)
        item['lgbtq'] = np.nanmax([item['trans'], item['other_gender'], item['homosexual_gay_or_lesbian'], 
                                   item['bisexual'], item['other_sexual_orientation']])

        
    return all_data, spec_data


def getFeatures(all_data, spec_data, glove_model, analysis_type="objective"):
    postsToFeatures, postsToLabels, postsToIds = {}, {}, {}
    vocab = glove_model.keys()

    for id, item in (all_data.items()):
        feat = []
        for w in item['text']:
            if w in vocab:
                feat.append(glove_model[w])

        if len(feat) == 0:
            continue

        if np.isnan(item['lgbtq']) or np.isnan(item['black']):
            continue

        feat = np.mean(feat, axis=0)
        postsToFeatures[item['text']] = feat
        postsToLabels[item['text']] = item['toxicity_binary']
        postsToIds[item['text']] = id

    for id, item in (all_data.items()):
        item['lgbtq_bin'] = 0
        item['black_bin'] = 0
        if item['lgbtq'] > 0:
            item['lgbtq_bin'] = 1
        if item['black'] > 0:
            item['black_bin'] = 1
        
    if analysis_type == "subjective":
        postsToLabels = {}
        for id, item in (all_data.items()):
            rows = spec_data[spec_data['id'] == id]    

            if item['lgbtq_bin'] == 1:
                rows_p = rows[rows["rater_group"] == "LGBTQ"]
            elif item['black_bin'] == 1:
                rows_p = rows[rows["rater_group"] == "AA"]
            else:
                rows_p = rows[rows["rater_group"] == "Control"]

            preds = [int(p) < 0 if not np.isnan(p) else 0 for p in list(rows_p["identity_attack"])]
            pred = int(np.mean(preds) > 0.5)

            postsToLabels[item['text']] = pred
            
    return postsToFeatures, postsToLabels, postsToIds

def getExperts(all_data, spec_data):
    experts = []
    for id, item in (all_data.items()):
        rows = spec_data[spec_data['id'] == id]
        ann_ids = list(rows["unique_contributor_id"])
        for ann in ann_ids:
            if ann not in experts:
                experts.append(ann)

    print (len(experts), "annotators")
    return experts


def getPrediction(deferrer, expertPreds, dSim, mu, k=3, useDSim = False):
    outputDef = np.array([deferrer[e] for e in expertPreds.keys()])
    
    outputDSim = np.array([dSim[e] for e in expertPreds.keys()])
    if useDSim:
        output = outputDef * (1-mu) + mu * outputDSim
    else:
        output = outputDef
    
    output = output/sum(output)
#     print (deferrer, expertPreds)
    
    exps = list(expertPreds.keys())
    committee = np.random.choice(exps, p=output, size=k)
    
#     print (committee, expertPreds)
    pred = np.mean([expertPreds[e] for e in committee]) > 0.5
    return int(pred)

def getPrediction_mab(deferrer, expertPreds, dSim, mu, k=3, useDSim = False):
    outputDef = np.array([deferrer[e] for e in expertPreds.keys()])
    
    outputDSim = np.array([dSim[e] for e in expertPreds.keys()])
    if useDSim:
        output = outputDef * (1-mu) + mu * outputDSim
    else:
        output = outputDef
    
    output = output/sum(output) * 0.5 + 1/len(output) * 0.5
#     print (deferrer, expertPreds)
    
    exps = list(expertPreds.keys())
    committee = np.random.choice(exps, p=output, size=k)
    
#     print (committee, expertPreds)
    pred = np.mean([expertPreds[e] for e in committee]) > 0.5
    return int(pred)

