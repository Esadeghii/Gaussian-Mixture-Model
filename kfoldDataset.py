"""
Created on Wed Dec 3 2021

@author: vyeruva@albany.edu
"""
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset ,WeightedRandomSampler
from sklearn.preprocessing import OneHotEncoder
from probabilityBin import AttributeProbabilityBin
import sys
os.chdir(sys.path[0]) #compatability hack

def dataSampler(labels):
    

    class_sample_count  = np.unique(labels, return_counts=True)[1]
    weights = 1. / torch.tensor(class_sample_count, dtype=torch.float)
    samples_weights = weights[labels]
    samples_weights = torch.reshape(samples_weights, (-1,))
    return WeightedRandomSampler(weights=samples_weights, num_samples=len(samples_weights), replacement=True)


class KFoldDataset:
    
    def __init__(self,datafile = './data-and-cleaning/supercleanGMMFilteredClusterd.xlsx', seqlen=10, noofbuckets = 7, kfolds = 10, curfold=0):
        if datafile[-4:] == 'xlsx':
            self.dataset = pd.read_excel(datafile)
        else:
            self.dataset = pd.read_csv(datafile)
        if "Unnamed: 0" in self.dataset.columns:
            self.dataset.drop(columns=["Unnamed: 0"],inplace=True)
        self.ALPHABET = ['A','C','G','T']
        self.seqlen = seqlen
        self.kfolds = kfolds
        self.curfold = curfold
        self.seqs = self.transform_sequences(
            self.dataset['Sequence'].apply(lambda x: pd.Series([c for c in x])).to_numpy())
        self.perm = torch.randperm(self.seqs.shape[0])
        self.noofbuckets = noofbuckets

    def updatefold(self,curfold):
        self.curfold = curfold

    def transform_sequences(self,seqs):
        enc = OneHotEncoder()
        enc.fit(np.array(self.ALPHABET).reshape(-1,1))
        return enc.transform(seqs.reshape(-1,1)).toarray().reshape(
            -1, self.seqlen, len(self.ALPHABET))
        
    # @staticmethod
    # def cust_collate(batch):
    #     #print(list(batch))
    #     #print(len(batch))
    #     return [x for x in batch]
        
    def data_loaders(self, batch_size, split=(0.85, 0.15)):
        
        self.Dlables = self.dataset['Label'].to_numpy(dtype="float").reshape(-1,1)
        
        nval = self.seqs.shape[0]
        foldsize = nval//self.kfolds
        splitstart = foldsize * self.curfold
        splitend = foldsize + splitstart
        self.train_seq = self.seqs[torch.cat((self.perm[:splitstart], self.perm[splitend:])),:,:]
        self.train_label = self.Dlables[torch.cat((self.perm[:splitstart], self.perm[splitend:])),:]


        train_ds = TensorDataset(
            torch.from_numpy(self.train_seq),
            torch.from_numpy(self.train_label),
            #torch.from_numpy(localII[perm[:split1],:])
        )

        test_ds = TensorDataset(
            torch.from_numpy(self.seqs[self.perm[splitstart:splitend],:,:]),
            torch.from_numpy(self.Dlables[self.perm[splitstart:splitend],:]),
            #torch.from_numpy(localII[perm[split1:split1+split2],:])
        )
        self.val_seq = self.seqs[self.perm[splitstart:splitend:],:,:]
        self.val_label = self.Dlables[self.perm[splitstart:splitend:],:]
        val_ds = TensorDataset(
            torch.from_numpy(self.val_seq),
            torch.from_numpy(self.val_label),
            #torch.from_numpy(localII[perm[split1+split2:],:])
        )

        print(len(train_ds),len(val_ds))
        
        train_dl = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True
            )
        test_dl = DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False
            )
        val_dl = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False
            )
        
        print(len(train_dl),len(test_dl),len(val_dl))
        
        return train_dl, val_dl, None
