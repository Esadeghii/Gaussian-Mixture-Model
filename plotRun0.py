
#James Oswald 1/12/21

import numpy as np
import matplotlib.pyplot as plt
import argparse

#takes a file path and returns a matplot figure object
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

    #validation losses vs epoch
    validMat = data["vl"]
    epochsV = validMat[:,0]
    rLossV = validMat[:,1]
    kdLossV = validMat[:,2]
    regLossV = validMat[:,3]
    lossV = validMat[:,4]
    accV = validMat[:,5]

    #corelations with each dimention vs epoch
    distence = data["distence"]
    distence_m = data["distence_m"]

    #fig, ax = plt.subplots(3)
    fig, ax = plt.subplots(3)
    fig.set_size_inches(8, 7)

    fig2, ax2 = plt.subplots(2,3)
    fig2.set_size_inches(8, 7)
    fig3, ax3 = plt.subplots(2,3)
    fig3.set_size_inches(8, 7)
    fig.suptitle( r" $\beta$=" + str(beta) + r" $\gamma$=" + str(gamma) + r" $\delta$=" + str(delta) + " latentDims=" + str(latentDims) + " lstmLayers=" + str(lstmLayers) + " Dropout=" + str(drop) + " lstmInfo=" + str(lstmInfo) + " linearWidth" , fontsize=12)
    fig2.suptitle( r" $\beta$=" + str(beta) + r" $\gamma$=" + str(gamma) + r" $\delta$=" + str(delta) + " latentDims=" + str(latentDims) + " lstmLayers=" + str(lstmLayers) + " Dropout=" + str(drop) + " lstmInfo=" + str(lstmInfo) + " linearWidth" , fontsize=12)
    fig3.suptitle( r" $\beta$=" + str(beta) + r" $\gamma$=" + str(gamma) + r" $\delta$=" + str(delta) + " latentDims=" + str(latentDims) + " lstmLayers=" + str(lstmLayers) + " Dropout=" + str(drop) + " lstmInfo=" + str(lstmInfo) + " linearWidth" , fontsize=12)


    ax[0].plot(epochsT, rLossT, label="r Loss")
    ax[0].plot(epochsT, kdLossT, label="kld Loss")
    ax[0].plot(epochsT, regLossT, label="reg Loss")
    ax[0].plot(epochsT, lossT, label="Training Loss")
    ax[0].plot(epochsV, lossV, label="Valid Loss", ls="--")
    ax[0].legend(loc="center left", bbox_to_anchor=(1, 0.5))
    ax[0].set(xlabel="epoch", ylabel="Training Loss")
    ax[0].label_outer()

    ax[1].plot(epochsT, rLossV, label="r Loss")
    ax[1].plot(epochsT, kdLossV, label="kld Loss")
    ax[1].plot(epochsT, regLossV, label="reg Loss")
    ax[1].plot(epochsV, lossV, label="Valid Loss")
    ax[1].legend(loc="center left", bbox_to_anchor=(1, 0.5))
    ax[1].set(xlabel="epoch", ylabel="Validation Loss")
    ax[1].label_outer()

    ax[2].plot(epochsT, accT, label="Train Accuracy")
    ax[2].plot(epochsV, accV, label="Valid Accuracy")
    ax[2].legend(loc="center left", bbox_to_anchor=(1, 0.5))
    ax[2].set(xlabel="epoch",ylabel="Accuracy")
    ax[2].text(x=1300, y=0.62, s=f"Training Accuracy: {'{:.3f}'.format(accT[len(accT) - 1])}")
    ax[2].text(x=1300, y=0.50, s=f"Testing Accuracy: {'{:.3f}'.format(accV[len(accV) - 1])}")
    ax[2].label_outer()
    plt.xticks(range(len(trainMat))) 

    help1= { 0:'cluster 0',
             1:'cluster 1',
             2:'cluster 2',
             3:'cluster 3',
             4:'cluster 4',
             5:'cluster 5'}
    wavg = np.array(distence).astype('float')
    y1 = np.array(distence[:,0])
    y2 = np.array(distence[:,1])
    y3 = np.array(distence[:,2])
    y4 = np.array(distence[:,3])
    y5 = np.array(distence[:,4])
    y6 = np.array(distence[:,5])
  
    ax2[0][0].plot(y1 ,'b', label=help1[0])
    ax2[0][1].plot(y2 ,'r', label=help1[1])
    ax2[0][2].plot(y3 ,'g', label=help1[2])
    ax2[1][0].plot(y4 ,'darkgreen', label=help1[3])
    ax2[1][1].plot(y5 ,'coral', label=help1[4])
    ax2[1][2].plot(y6 ,'tomato', label=help1[5])


    ax2[0][0].legend()
    ax2[0][1].legend()
    ax2[0][2].legend()
    ax2[1][0].legend()
    ax2[1][1].legend()
    ax2[1][2].legend()

    y1_ = np.array(distence_m[:,0])
    y2_ = np.array(distence_m[:,1])
    y3_ = np.array(distence_m[:,2])
    y4_ = np.array(distence_m[:,3])
    y5_ = np.array(distence_m[:,4])
    y6_ = np.array(distence_m[:,5])
  
    ax3[0][0].plot(y1_ ,'b', label=help1[0])
    ax3[0][1].plot(y2_ ,'r', label=help1[1])
    ax3[0][2].plot(y3_ ,'g', label=help1[2])
    ax3[1][0].plot(y4_ ,'darkgreen', label=help1[3])
    ax3[1][1].plot(y5_ ,'coral', label=help1[4])
    ax3[1][2].plot(y6_ ,'tomato', label=help1[5])


    ax3[0][0].legend()
    ax3[0][1].legend()
    ax3[0][2].legend()
    ax3[1][0].legend()
    ax3[1][1].legend()
    ax3[1][2].legend()

    # ax[8].set(xlabel="epoch",ylabel='distence: '+str(round(np.mean(wavg),4)))
    # ax[8].legend(loc="center left", bbox_to_anchor=(1, 0.5), ncol=2)
    # ax[3].label_outer()
   

    fig.subplots_adjust(right=0.75)
    fig2.subplots_adjust(right=0.75)
    fig3.subplots_adjust(right=0.75)
    plt.setp(ax, xticks=list(range(0,len(trainMat),50)), xticklabels=list(range(0,len(trainMat),50)))    
    plt.setp(ax2, xticks=list(range(0,len(distence_m))), xticklabels=list(range(0,len(trainMat),50)))    
    plt.setp(ax3, xticks=list(range(0,len(distence_m))), xticklabels=list(range(0,len(trainMat),50)))     
    return fig ,fig2,fig3

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", dest="i", default="None")
parser.add_argument("-o", "--output", dest="o", default="None")
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
    f2.savefig(graphsPath+"/2"+graph)
    f3.savefig(graphsPath+"/3"+graph)


#genFigure("all-results/1-18-22-res/runs/a19lds19b0.007g1.0d1.0h13.npz")