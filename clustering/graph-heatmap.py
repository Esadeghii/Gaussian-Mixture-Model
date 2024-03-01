import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering
from sklearn.model_selection import ParameterGrid
import os 
from itertools import chain
from WDistanceCS import Distence
import warnings
import seaborn as sns
warnings.filterwarnings('ignore')
if 'clustersGraph' not in os.listdir():
    os.mkdir('clustersGraph')

data = pd.read_excel('./clustering/gaussian_all_data.xlsx' )
data = data.fillna('0')
data = data.dropna()
ddata = pd.read_excel('./clustering/CSdistances_gaussian_all_data.xlsx' ).fillna(0).to_numpy(float)
flattenddata = list(chain.from_iterable(ddata)) ###
#delta = max(flattenddata)-min(flattenddata) ###
sequenceNames = list(pd.read_excel('./clustering/gaussian_all_data.xlsx')['Sequence'])
index = 15
mini = 410 #np.min(np.array(datas, dtype="float")) 
maxi = 1100#np.max(np.array(datas, dtype="float"))
tresholdv = (maxi - mini)//index
ranges = [(-np.inf,mini)]+[(i,i+tresholdv)  for i in np.arange(mini,maxi,tresholdv) ]+[(maxi,np.inf)]

def colerize(number , datas,tr):
        # if number > 800: return 'Nir'
        # if 660 < number <= 800: return 'VeryRed'
        # if 590 < number <= 660: return 'Red'
        # if number <= 590: return 'Green'


        return str([idx for idx ,rng in enumerate(ranges) if rng[0]<=float(number)<rng[1]][0])


numClasters = list(range(4,7,1)) # type int
#numClasters = [7,8]
eigen_solvers = [None] # {‘arpack’, ‘lobpcg’, ‘amg’}, default=None
n_components = [None] # type int, default=None
random_states = [100] # type int, default=None
n_inits = [30] # type int, default=10
gammas = [1.0] # type float, default=1.0
affinitys =  ['precomputed']# str or callable, default=’rbf’
n_neighbors = [50] # int, default=10
eigen_tols = ['auto'] # float, default=”auto”
assign_labels = ['kmeans'] # {‘kmeans’, ‘discretize’, ‘cluster_qr’}, default=’kmeans’
degrees = [3] # type float
coef0s = [1] # type float
kernel_params = [None] # dict of str to any, default=None
n_jobs = [None] #int, default=None 
deltas = [0.8]
# affinity='precomputed', you can ignore the n_jobs parameter, and it won't impact the clustering process. 

paramsDict = {
        'numClaster' : numClasters,
        'eigen_solver' : eigen_solvers,
        'n_component' : n_components,
        'random_state' : random_states,
        'n_init' : n_inits,
        'gamma' : gammas,
        'affinity' : affinitys,
        'n_neighbor' : n_neighbors,
        'eigen_tol' : eigen_tols,
        'assign_label' : assign_labels,
        'degree' : degrees,
        'coef0' : coef0s,
        'kernel_param' : kernel_params,
        'n_job' : n_jobs,
        'delta' :deltas
} 
Inter_Mean = []
for idx , params in enumerate(list(ParameterGrid(paramsDict))):
        if str('-'.join(map(str, params.values()))) not in os.listdir('clustersGraph'):os.mkdir('clustersGraph/'+str('-'.join(map(str, params.values()))))
        # try:
        smdata = [list(map(lambda X:np.exp(- X ** 2 / (2. * params["delta"] ** 2)), row)) for row in ddata] ###
        clustering = SpectralClustering(n_clusters=params["numClaster"],
                                        eigen_solver=params["eigen_solver"],
                                        n_components=params["n_component"],
                                        random_state=params["random_state"],
                                        n_init=params["n_init"],
                                        gamma=params["gamma"],
                                        affinity=params["affinity"],
                                        n_neighbors=params["n_neighbor"],
                                        eigen_tol=params["eigen_tol"],
                                        assign_labels=params["assign_label"],
                                        degree=params["degree"],
                                        coef0=params["coef0"],
                                        kernel_params=params["kernel_param"],
                                        n_jobs=params["n_job"],
                                        verbose=False).fit(smdata)
        data['labels'] = clustering.labels_
        data.to_excel('clustersGraph/'+str('-'.join(map(str, params.values())))+'/data.xlsx',index=False)
        
     # Calculate histograms for each cluster based on class labels from df1
        cldata = []
        cluster_sums = []
        for cluster_label in set(data['labels']):
                if cluster_label == None:break
                cluster_data = data[data['labels'] == cluster_label]
                cluster_data.reset_index(drop=True, inplace=True)
                results = []
                for idx in range(len(cluster_data)):
                        colers = dict()
                        _ = [colers.update({str(idxs):[]}) for idxs in range(index+2)]
                        #dict({'Green' : [],'Red' : [],'VeryRed' : [],'Nir' : []})
                        try:colers[colerize(cluster_data['Peak 1 b WAV'  ][idx] ,cluster_data['Peak 1 b WAV'  ],index)] = [*colers[colerize(cluster_data['Peak 1 b WAV'  ][idx],cluster_data['Peak 1 b WAV'  ],index )],float(cluster_data['Peak 1 a Log WAV'  ][idx])]
                        except:pass
                        try:colers[colerize(cluster_data['Peak 2 b WAV'  ][idx] ,cluster_data['Peak 2 b WAV'  ],index)] = [*colers[colerize(cluster_data['Peak 2 b WAV'  ][idx],cluster_data['Peak 2 b WAV'  ],index )],float(cluster_data['Peak 2 a Log WAV'  ][idx])]
                        except:pass
                        try:colers[colerize(cluster_data['Peak 3 b WAV'  ][idx] ,cluster_data['Peak 3 b WAV'  ],index)] = [*colers[colerize(cluster_data['Peak 3 b WAV'  ][idx],cluster_data['Peak 3 b WAV'  ],index )],float(cluster_data['Peak 3 a Log WAV'  ][idx])]
                        except:pass
                        try:colers[colerize(cluster_data['Peak NIR WAV'  ][idx] ,cluster_data['Peak NIR WAV'  ],index)] = [*colers[colerize(cluster_data['Peak NIR WAV'  ][idx],cluster_data['Peak NIR WAV'  ],index )],float(cluster_data['Peak NIR a Log WAV'  ][idx])]
                        except:pass
                        results.append(pd.DataFrame([np.mean(idxx) for idxx in colers.values() ]).fillna(0)[0].to_numpy())
                mappinglist = list(data['Sequence'])
                tmplist = list(cluster_data['Sequence'])
                listOfDistances = []
                for _ in range(len(cluster_data['Sequence'][:])):
                        tmpinstanse = tmplist.pop()
                        for jx in tmplist:
                                listOfDistances.append(ddata[mappinglist.index(jx)][mappinglist.index(tmpinstanse)])

                
                plt.figure(figsize=(7, 7))
                plt.bar(list(colers.keys()),np.mean(results,axis=0))#,yerr=np.std(results,axis=0))
                plt.errorbar(list(colers.keys()),np.mean(results,axis=0),[np.zeros_like(np.std(results, axis=0)), np.std(results, axis=0)], fmt='.', color='Black', elinewidth=2,capthick=2,errorevery=1, alpha=0.8, ms=0, capsize = 8)
                plt.xlabel('Class')
                plt.xticks(list(colers.keys()), [str(ids) for ids in ranges], rotation='vertical') 
                plt.ylabel('weight')
                cluster_sums.append(round(np.sum(listOfDistances), 3))
                if len(listOfDistances) != 0: plt.title(f'Cluster {cluster_label} - Number of Sequences: {len(cluster_data)}\n Intra-cluster distance: (Avg: {round(np.mean(listOfDistances), 3)}) ; (Max: {round(np.max(listOfDistances), 3)})\n (sum: {round(np.sum(listOfDistances), 3)})')
                #plt.show()
                plt.savefig(f'clustersGraph/{str("-".join(map(str, params.values())))}/cluster_{cluster_label}_barchart.png')
                plt.close()
                cldata.append(np.mean(np.array(cluster_data[['Peak 1 a Log','Peak 1 b','Peak 1 c','Peak 2 a Log','Peak 2 b','Peak 2 c','Peak 3 a Log','Peak 3 b','Peak 3 c', 'Peak NIR a Log', 'Peak NIR b', 'Peak NIR c']].to_numpy(),dtype=float),axis=0))
        dis = Distence(pd.DataFrame(cldata,columns=['Peak 1 a Log','Peak 1 b','Peak 1 c','Peak 2 a Log','Peak 2 b','Peak 2 c','Peak 3 a Log','Peak 3 b','Peak 3 c', 'Peak NIR a Log', 'Peak NIR b', 'Peak NIR c']))
        dis.to_excel(f'clustersGraph/{str("-".join(map(str, params.values())))}/cluster_{cluster_label}_distences.xlsx',index=False)    
        
        dis_Mean = np.mean(dis.to_numpy())
        Inter_Mean.append((params["numClaster"], dis_Mean))


        # Mask the lower triangle of the matrix (including the diagonal)
        mask = np.triu(np.ones_like(dis, dtype=bool))

        # Create a larger heatmap using Seaborn, with only the upper triangle
        sns.set(font_scale=1.5)  # Adjust font size if needed
        plt.figure(figsize=(12, 10))  # Set the size of the larger heatmap

        # Round the distance values to 2 decimal places
        rounded_distance_matrix = dis.round(2)

        # Customize the appearance of the heatmap with the "viridis" colormap
        sns.heatmap(rounded_distance_matrix, annot=True, cmap='viridis', square=True, mask=mask, fmt=".2f", cbar=True, cbar_kws={"shrink": 0.75}, annot_kws={"size": 10})
        # Add labels and title
        plt.xlabel('Cluster')
        plt.ylabel('Cluster')
        plt.title('Inter-cluster CS-Distance')
        # Generate the filename based on parameter values
        plt.savefig(f'clustersGraph/{str("-".join(map(str, params.values())))}/_graphchart.png')
        # Close the current figure to release resources
        plt.close()

        
        print(str(idx)+' done' , end='\r')
        #for errors
        # except Exception as e:
        #         with open('clustersGraph/'+str('-'.join(map(str, params.values())))+'/error.txt','w') as f:
        #               f.write(str(e))
        #         print(str(idx)+' done' , end='\r')
        plt.figure(figsize=(10,9))
        cluster_sums_Df = pd.DataFrame({'cluster':[str(j) for j in set(data['labels'])],'sum':cluster_sums})
        cluster_sums_Df.sort_values('sum', ascending=True, inplace=True)
        #plt.scatter(list(set(data['labels'])),cluster_sums)
        plt.scatter('cluster', 'sum', data=cluster_sums_Df)
        plt.xlabel('Clusters')
        plt.ylabel('sums')
        plt.savefig(f'clustersGraph/{str("-".join(map(str, params.values())))}/clastersHeat.png')
        plt.close()


plt.scatter(*zip(*Inter_Mean))
plt.savefig('clustersGraph/Inter_Mean.png')
plt.close()
