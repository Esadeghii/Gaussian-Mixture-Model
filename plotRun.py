
#James Oswald 1/12/21

import numpy as np
import matplotlib.pyplot as plt
import argparse
import pandas as pd
import matplotlib.ticker as mtick


def genFigure(filePath):
    data = np.load(filePath)
    beta, gamma, delta, latentDims, lstmLayers, drop, lstmInfo = data["par"]

    trainMat = data["tl"]
    epochsT = trainMat[:,0]
    rLossT = trainMat[:,1]
    kdLossT = trainMat[:,2]
    regLossT = trainMat[:,3]
    lossT = trainMat[:,4]
    accT = trainMat[:,5]

    validMat = data["vl"]
    epochsV = validMat[:,0]
    rLossV = validMat[:,1]
    kdLossV = validMat[:,2]
    regLossV = validMat[:,3]
    lossV = validMat[:,4]
    accV = validMat[:,5]

    distence = data["distence"]
    distence_m = data["distence_m"]
    correlation = data["correlation"]
    correlation_valid = data["correlation_valid"]

    fig, ax = plt.subplots(6, figsize=(8, 15))
    fig.suptitle(r" $\beta$=" + str(beta) + r" $\gamma$=" + str(gamma) + r" $\delta$=" + str(delta) + " latentDims=" + str(latentDims) + " lstmLayers=" + str(lstmLayers) + " Dropout=" + str(drop) + " lstmInfo=" + str(lstmInfo), fontsize=12)

    # Separate subplots for each loss
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

    plt.tight_layout()
    plt.subplots_adjust(top=0.95)  # Adjust to not overlap with title
    return fig
 



# Example usage with argument parsing setup
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True, help="Input file path")
    parser.add_argument("-o", "--output", required=True, help="Output file path")
    args = parser.parse_args()

""" if args.i != "None":
    f = genFigure(args.i)
    name = ""
    if args.o == "None":
        raise Exception("No output name provided")
        #name = "./graphs/b"+args.i+".png"
    else:
        name = args.o
    f.savefig(name)
else:
    fileName = "./runs/1623311492.npz"
    f = genFigure(fileName)
    f.savefig("./graphs/latest.png") """

def genPlotForRun(runsPath, run, graphsPath, graph):
    f = genFigure(runsPath+"/"+run)
    f.savefig(graphsPath+"/"+graph)
    # f2.savefig(graphsPath+"/2"+graph)
    # f3.savefig(graphsPath+"/3"+graph)


#genFigure("all-results/1-18-22-res/runs/a19lds19b0.007g1.0d1.0h13.npz")