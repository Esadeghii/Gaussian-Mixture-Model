import os
import time
import numpy as np
import torch
import pandas as pd
from sklearn.model_selection import ParameterGrid
from sequenceModel import SequenceModel
from sequenceTrainer import SequenceTrainer
from sequenceDataset import SequenceDataset
from plotRun import genPlotForRun

# CPU or GPU?
os.environ["CUDA_VISIBLE_DEVICES"]=""

batchSize = 32
numEpochs =20
betas = [0.007]
gammas = [0.02]
deltas = [0.1]
dropout = [0]
latentDims = [19]
lstmLayers = [1]
hiddenSize = [13] # features inside lstm
trainTestSplit = (0.85, 0.15)
lambdas = [0.1]
weighted = False

# Post-processing of log data, averages metrics for each epoch
def averageCols(logMat):
    print(logMat.shape)
    rv = np.zeros((numEpochs, 6))
    for epoch in range(numEpochs):
        for col in range(1, 6):
            num = 0
            for row in range(logMat.shape[0]):
                if logMat[row, 0] == epoch:
                    rv[epoch, col] += logMat[row, col]
                    num += 1
            rv[epoch, col] /= num
        rv[epoch, 0] = epoch
    return rv

paramDict = {"beta": betas, "gamma": gammas, "delta": deltas, 
             "latentDims": latentDims, "lstmLayers": lstmLayers, "dropout": dropout, "hiddenSize": hiddenSize, "lambda": lambdas}

for params in list(ParameterGrid(paramDict)):   # gridsearch
    # Set up the model and trainer
    model = SequenceModel(hidden_layers=params["lstmLayers"], emb_dim=params["latentDims"], dropout=params["dropout"], hidden_size=params["hiddenSize"])
    data = SequenceDataset(split=trainTestSplit)
    filename = f"a{params['latentDims']}lds{params['latentDims']}b{params['beta']}g{params['gamma']}d{params['delta']}h{params['hiddenSize']}l{params['lambda']}"
    if torch.cuda.is_available(): 
        print('cuda available')
        model.cuda()
    trainer = SequenceTrainer(data, model, beta=params["beta"], gamma=params["gamma"], delta=params["delta"], logTerms=True, IICorVsEpoch=True, lambda_=params["lambda"])
    if torch.cuda.is_available(): 
        trainer.cuda()

    # Train model, use internal logging
    print("Training Model")
    distence, distence_m, correlation, correlation_valid = trainer.train_model(batchSize, numEpochs, filename, params, log=False, weightedLoss=weighted)

    # Save the model
    if weighted:
        torch.save(model, f"./models/weighted/{filename}.pt")
    else:
        torch.save(model, f"./models/{filename}.pt")

    # Adding vmim_loss to the saved data
    vmim_loss = trainer.vmim_loss_list  
    # Split vmim_loss into training and validation parts
    loss_vmimT = [vmim_loss[i] for i in range(0, len(vmim_loss), 2)]
    loss_vmimV = [vmim_loss[i+1] for i in range(0, len(vmim_loss), 2)]

    print("Averaging stats for batches")
    tl = averageCols(trainer.trainList)
    vl = averageCols(trainer.validList)

    # Save the training logs
    par = np.array([params["beta"], params["gamma"], params["delta"], params["latentDims"], params["lstmLayers"], params["dropout"], params["hiddenSize"], params["lambda"]])
    if weighted:
        np.savez(f"./runs/weighted/{filename}.npz", par=par, tl=tl, vl=vl, distence=distence, distence_m=distence_m, correlation=correlation, correlation_valid=correlation_valid, vmim_loss=vmim_loss, loss_vmimT=loss_vmimT, loss_vmimV=loss_vmimV, trainLabel=data.train_label, validLabel=data.val_label)
    else:
        np.savez(f"./runs/{filename}.npz", par=par, tl=tl, vl=vl, distence=distence, distence_m=distence_m, correlation=correlation, correlation_valid=correlation_valid, vmim_loss=vmim_loss, loss_vmimT=loss_vmimT, loss_vmimV=loss_vmimV, trainLabel=data.train_label, validLabel=data.val_label)
