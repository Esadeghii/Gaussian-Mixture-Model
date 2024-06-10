
#James Oswald 1/12/21
import numpy as np
import matplotlib.pyplot as plt
import os.path as path

#takes a file path and returns a matplot figure object
class Plotter:

    def __init__(self, metricsfolder=path.join(".","runs","kfold"), graphsfolder=path.join(".","graphs","kfold")):
        self.inputfolder = metricsfolder
        self.outputfolder = graphsfolder
        self.avg_corrT = []
        self.avg_corrV = []
     
    def genAvgFigure(self, inputFiles, outputFigure):
        data =[]
        for file in inputFiles:
            data.append(np.load(path.join(self.inputfolder, file)))

        beta, gamma, delta, latentDims, lstmLayers, drop, lstmInfo = data[0]["par"]
        
        rLossM = []
        kdLossM = []
        regLossM = []
        lossM = []
        accM = []
        for datasub in data:
            trainMat = datasub["tl"]
            rLossM = np.append(rLossM, trainMat[:,1])
            kdLossM = np.append(kdLossM, trainMat[:,2])
            regLossM = np.append(regLossM, trainMat[:,3])
            lossM = np.append(lossM, trainMat[:,4])
            accM = np.append(accM, trainMat[:,5])


        epochsT = data[0]["tl"][:,0]
        rLossT = np.mean(np.reshape(rLossM, (len(data), len(epochsT))), axis=0)
        kdLossT = np.mean(np.reshape(kdLossM, (len(data), len(epochsT))), axis=0)
        regLossT =  np.mean(np.reshape(regLossM, (len(data), len(epochsT))), axis=0)
        lossT =  np.mean(np.reshape(lossM, (len(data), len(epochsT))), axis=0)
        accT =  np.mean(np.reshape(accM, (len(data),len(epochsT))), axis=0)

        rLossM = []
        kdLossM = []
        regLossM = []
        lossM = []
        accM = []

        for datasub in data:
            validMat = datasub["vl"]
            rLossM = np.append(rLossM, validMat[:,1])
            kdLossM = np.append(kdLossM, validMat[:,2])
            regLossM = np.append(regLossM, validMat[:,3])
            lossM = np.append(lossM, validMat[:,4])
            accM = np.append(accM, validMat[:,5])
        
        #validation losses vs epoch
        epochsV = data[0]["vl"][:,0]
        rLossV = np.mean(np.reshape(rLossM, (len(data), len(epochsV))), axis=0)
        kdLossV = np.mean(np.reshape(kdLossM, (len(data), len(epochsV))), axis=0)
        regLossV =  np.mean(np.reshape(regLossM, (len(data), len(epochsV))), axis=0)
        lossV =  np.mean(np.reshape(lossM, (len(data), len(epochsV))), axis=0)
        accV =  np.mean(np.reshape(accM, (len(data), len(epochsV))), axis=0)
        fvaccuray = accV[len(epochsV)-1]


         

        #fig, ax = plt.subplots(3)
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

        ax[5].plot(list(range(0, np.mean(np.array(self.avg_corrT),axis=0).shape[0]*100, 100)), np.mean(self.avg_corrT,axis=0), label="Training Correlation")
        ax[5].plot(list(range(0, np.mean(np.array(self.avg_corrV),axis=0).shape[0]*100, 100)), np.mean(self.avg_corrV,axis=0), label="Validation Correlation", ls="--")
        ax[5].set_ylabel("Correlation")
        ax[5].set_xlabel("Epoch")
        ax[5].legend(loc="upper right")



        fig.subplots_adjust(right=0.75)
        fig.savefig(path.join(self.outputfolder, outputFigure))
        return fig

    def genFigure(self, inputFile , outputFigure):
        data = np.load(path.join(self.inputfolder, inputFile))
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
        correlation = data["correlation"]
        correlation_valid = data["correlation_valid"]
        self.avg_corrT.append(correlation)
        self.avg_corrV.append(correlation_valid)



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
        fig.subplots_adjust(right=0.75)
        fig.savefig(path.join(self.outputfolder, outputFigure))
        return fig
