
#James Oswald 1/12/21
import numpy as np
import matplotlib.pyplot as plt
import os.path as path

#takes a file path and returns a matplot figure object
class Plotter:

    def __init__(self, metricsfolder=path.join(".","runs","kfold"), graphsfolder=path.join(".","graphs","kfold")):
        self.inputfolder = metricsfolder
        self.outputfolder = graphsfolder
     
    def genAvgFigure(self, inputFiles, outputFigure):
        data =[]
        for file in inputFiles:
            data.append(np.load(path.join(self.inputfolder, file)))

        alpha, beta, gamma, delta, latentDims, lstmLayers, drop, lstmInfo = data[0]["par"]
        
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
            validMat = data[0]["vl"]
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

        #corelations with each dimention vs epoch
        wldims = data[0]["wldims"]
        lidims = data[0]["liidims"]
        wldimsM = np.reshape(wldims, np.shape(wldims)+(1,))
        lidimsM = np.reshape(wldims, np.shape(lidims)+(1,))

        for datasub in data:
            wldims = datasub["wldims"]
            lidims = datasub["liidims"]
            wldimsM = np.concatenate((wldimsM, np.reshape(wldims, np.shape(wldims)+(1,))), axis=2)
            lidimsM = np.concatenate((lidimsM, np.reshape(lidims, np.shape(lidims)+(1,))), axis=2)

        wldimCors = np.mean(wldimsM, axis=2) 
        lidimCors = np.mean(lidimsM, axis=2) 
        dimRange = [j * 50 for j in range(wldimCors.shape[0])]
        finalWIIcor = wldimCors[:,0][len(wldimCors[:,0])-1]
        finalLIIcor = wldimCors[:,1][len(wldimCors[:,1])-1]  

        #fig, ax = plt.subplots(3)
        fig, ax = plt.subplots(5)
        fig.set_size_inches(8, 7)
        fig.suptitle(r"$\alpha$=" + str(alpha) + r"$\beta$=" + str(beta) + r" $\gamma$=" + str(gamma) + r" $\delta$=" + str(delta) + " latentDims=" + str(latentDims) + " lstmLayers=" + str(lstmLayers) + " Dropout=" + str(drop) + " lstmInfo=" + str(lstmInfo), fontsize=12)

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
        ax[2].label_outer()
        ax[2].text(400, 0.5, "V accuracy="+ str(fvaccuray))

        for i in range(int(latentDims)):
            ax[3].plot(dimRange, wldimCors[:,i], label=str(i), ls="-" if i == 0 else "--")
            ax[4].plot(dimRange, lidimCors[:,i], label=str(i), ls="-" if i == 1 else "--")
        ax[3].legend(loc="center left", bbox_to_anchor=(1, 0.5), ncol=2)
        ax[3].set(xlabel="epoch",ylabel="WL Cor")
        ax[3].label_outer()
        ax[3].text(400, 0.2, "WL z0 corr="+ str(finalWIIcor))

        ax[4].legend(loc="center left", bbox_to_anchor=(1, 0.3), ncol=2)
        ax[4].set(xlabel="epoch",ylabel="LII Cor")
        ax[4].label_outer()
        ax[4].text(400, 0.1, "LII z1 corr="+ str(finalLIIcor))

        fig.subplots_adjust(right=0.75)
        fig.savefig(path.join(self.outputfolder, outputFigure))
        return fig

    def genFigure(self, inputFile , outputFigure):
        data = np.load(path.join(self.inputfolder, inputFile))
        alpha, beta, gamma, delta, latentDims, lstmLayers, drop, lstmInfo = data["par"]

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
        wldimCors = data["wldims"]
        lidimCors = data["liidims"]
        dimRange = [j * 50 for j in range(wldimCors.shape[0])]

        #fig, ax = plt.subplots(3)
        fig, ax = plt.subplots(5)
        fig.set_size_inches(8, 7)
        fig.suptitle(r"$\alpha$=" + str(alpha) + r"$\beta$=" + str(beta) + r" $\gamma$=" + str(gamma) + r" $\delta$=" + str(delta) + " latentDims=" + str(latentDims) + " lstmLayers=" + str(lstmLayers) + " Dropout=" + str(drop) + " lstmInfo=" + str(lstmInfo), fontsize=12)

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
        ax[2].label_outer()

        for i in range(int(latentDims)):
            ax[3].plot(dimRange, wldimCors[:,i], label=str(i), ls="-" if i == 0 else "--")
            ax[4].plot(dimRange, lidimCors[:,i], label=str(i), ls="-" if i == 1 else "--")
        ax[3].legend(loc="center left", bbox_to_anchor=(1, 0.5), ncol=2)
        ax[3].set(xlabel="epoch",ylabel="WL Cor")
        ax[3].label_outer()
        ax[4].legend(loc="center left", bbox_to_anchor=(1, 0.3), ncol=2)
        ax[4].set(xlabel="epoch",ylabel="LII Cor")
        ax[4].label_outer()

        fig.subplots_adjust(right=0.75)
        fig.savefig(path.join(self.outputfolder, outputFigure))
        return fig
