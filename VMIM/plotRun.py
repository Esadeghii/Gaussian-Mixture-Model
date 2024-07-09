import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

def genFigure(filePath):
    data = np.load(filePath)
    
    parameters = data["par"]
    print(f"Number of parameters: {len(parameters)}")
    
    if len(parameters) == 8:
        beta, gamma, delta, latentDims, lstmLayers, drop, lstmInfo, lambda_ = parameters
    else:
        raise ValueError(f"Expected 8 parameters, but got {len(parameters)}")
    
    trainMat = data["tl"]
    epochsT = trainMat[:, 0]
    rLossT = trainMat[:, 1]
    kdLossT = trainMat[:, 2]
    regLossT = trainMat[:, 3]
    lossT = trainMat[:, 4]
    accT = trainMat[:, 5]

    validMat = data["vl"]
    epochsV = validMat[:, 0]
    rLossV = validMat[:, 1]
    kdLossV = validMat[:, 2]
    regLossV = validMat[:, 3]
    lossV = validMat[:, 4]
    accV = validMat[:, 5]

    distence = data["distence"]
    distence_m = data["distence_m"]
    correlation = data["correlation"]
    correlation_valid = data["correlation_valid"]
    
    loss_vmimT = data["loss_vmimT"]
    loss_vmimV = data["loss_vmimV"]

    fig, ax = plt.subplots(7, figsize=(8, 17))  
    fig.suptitle(r" $\beta$=" + str(beta) + r" $\gamma$=" + str(gamma) + r" $\delta$=" + str(delta) + 
                 " latentDims=" + str(latentDims) + " lstmLayers=" + str(lstmLayers) + 
                 " Dropout=" + str(drop) + " lstmInfo=" + str(lstmInfo) + " $\lambda$=" + str(lambda_), fontsize=12)

    ax[0].plot(epochsT, rLossT, label="r Loss (Train)")
    ax[0].plot(epochsV, rLossV, label="r Loss (Valid)", ls="--")
    ax[0].set_ylabel("r Loss")
    ax[0].legend(loc="upper right")

    ax[1].plot(epochsT, kdLossT, label="kld Loss (Train)")
    ax[1].plot(epochsV, kdLossV, label="kld Loss (Valid)", ls="--")
    ax[1].set_ylabel("kld Loss")
    ax[1].legend(loc="upper right")

    ax[2].plot(epochsT, regLossT, label="reg Loss (Train)")
    ax[2].plot(epochsV, regLossV, label="reg Loss (Valid)", ls="--")
    ax[2].set_ylabel("reg Loss")
    ax[2].legend(loc="upper right")

    ax[3].plot(epochsT, lossT, label="Training Loss")
    ax[3].plot(epochsV, lossV, label="Validation Loss", ls="--")
    ax[3].set_ylabel("Total Loss")
    ax[3].legend(loc="upper right")

    ax[4].plot(epochsT, accT, label="Training Accuracy")
    ax[4].plot(epochsV, accV, label="Validation Accuracy", ls="--")
    ax[4].set_ylabel("Accuracy")
    ax[4].legend(loc="upper right")

    ax[5].plot(list(range(0, len(correlation)*100, 100)), correlation, label="Training Correlation")
    ax[5].plot(list(range(0, len(correlation_valid)*100, 100)), correlation_valid, label="Validation Correlation", ls="--")
    ax[5].set_ylabel("Correlation")
    ax[5].set_xlabel("Epoch")
    ax[5].legend(loc="upper right")
    
    # Ensure that the lengths of epochs and loss_vmim arrays match
    min_len_vmimT = min(len(epochsT), len(loss_vmimT))
    min_len_vmimV = min(len(epochsV), len(loss_vmimV))
    
    ax[6].plot(epochsT[:min_len_vmimT], loss_vmimT[:min_len_vmimT], label="Training VMIM Loss")
    ax[6].plot(epochsV[:min_len_vmimV], loss_vmimV[:min_len_vmimV], label="Validation VMIM Loss", ls="--")
    ax[6].set_ylabel("VMIM Loss")
    ax[6].set_xlabel("Epoch")
    ax[6].legend(loc="upper right")

    plt.tight_layout()
    plt.subplots_adjust(top=0.95)  
    return fig

def genPlotForRun(runsPath, run, graphsPath, graph):
    f = genFigure(runsPath + "/" + run)
    f.savefig(graphsPath + "/" + graph)
    plt.show()  

runsPath = "./runs"
graphsPath = "./graphs"
os.makedirs(graphsPath, exist_ok=True)

run_files = [f for f in os.listdir(runsPath) if f.endswith('.npz')]

for run_file in run_files:
    run = run_file
    graph = os.path.splitext(run_file)[0] + ".png"
    
    print(f"Generating plot for {run_file}...")
    genPlotForRun(runsPath, run_file, graphsPath, graph)
    print(f"Saved plot to {graphsPath}/{graph}")
