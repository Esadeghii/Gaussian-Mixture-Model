import numpy as np
import matplotlib.pyplot as plt
import networkx as nx  
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


GmmData = pd.read_excel('data-and-cleaning/supercleanGMMFilteredClusterd.xlsx')
GmmDData = pd.read_excel('data-and-cleaning/supercleanGMMFiltered-distances.xlsx')
GmmDataS = GmmData['Sequence'].to_list()


def Distencecs(data_per_cluster0, data_per_cluster1):
    filterdDdata = np.zeros((len(data_per_cluster0),len(data_per_cluster1)))
    for idx,clmns in enumerate(data_per_cluster0):
        for jdx,rows in enumerate(data_per_cluster1):
            if np.isnan(GmmDData[GmmDataS.index(clmns)][GmmDataS.index(rows)]):
                raise ValueError('GmmDData is nan')
            filterdDdata[idx][jdx] =  GmmDData[GmmDataS.index(clmns)][GmmDataS.index(rows)]

    return np.mean(filterdDdata)

data_per_cluster = []
for clusternum in set(GmmData['Label']):
    data_per_cluster.append(GmmData[GmmData['Label'] == clusternum]['Sequence'].tolist())

distance_clusters = np.zeros((len(data_per_cluster),len(data_per_cluster)))  
for idx,clmns in enumerate(data_per_cluster):
    for jdx,rows in enumerate(data_per_cluster):
        distance_clusters[idx][jdx] = Distencecs(clmns,rows) 
pd.DataFrame(distance_clusters).to_excel('clustering/distance_clusters.xlsx')


G = nx.Graph()
for i in range(len(data_per_cluster)):
    for j in range(i+1,len(data_per_cluster)):
        G.add_edge(i,j,weight=distance_clusters[i][j])

pos = nx.spring_layout(G)
edge_labels=dict([((u,v,),d['weight'])
             for u,v,d in G.edges(data=True)])

nx.draw_networkx_edge_labels(G,pos,edge_labels=edge_labels)
nx.draw(G, pos, with_labels=True, node_size=700, node_color='skyblue', font_size=20, font_weight='bold', edge_color='black', width=2.0, edge_cmap=plt.cm.Blues)
plt.show()
    