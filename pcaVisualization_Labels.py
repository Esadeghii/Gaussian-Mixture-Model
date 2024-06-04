import math
import time
import sequenceDataset as sd
import pandas as pd
import numpy as np
import torch
from sklearn.decomposition import KernelPCA
import matplotlib.pyplot as plt
import plotly.express as px


def process_data_file(path_to_dataset: str, sequence_length=10):
    """Takes in a filepath to desired dataset and the sequence length of the sequences for that dataset,
    saves .npz file with arrays for one hot encoded sequences, array of wavelengths and array of local
    integrated intensities"""
    data = sd.SequenceDataset(path_to_dataset, sequence_length)
    ohe_sequences = data.transform_sequences(data.dataset['Sequence'].apply(lambda x: pd.Series([c for c in x])).
                                             to_numpy())
    Wavelen = np.array(data.dataset['Wavelen'])
    LII = np.array(data.dataset['LII'])
    np.savez(f"data-for-sampling/processed-data-files/pca-data-{time.time()}", Wavelen=Wavelen, LII=LII,
             ohe=ohe_sequences)






# 3D plot of pc1, pc2, pc3
# def pca_visualize_3D(principalComponents, principalDf, Labels, z):
#     """A 3D version of pca_visualize_wavelength(), where z provides a third dimension in the visualization of
#     principal component analysis"""

#     clusterscolor={0:'R-FR',1:'G', 2:'N', 3:'G-R', 4:'G-N', 5:'G-FR'}
#     Labels_with_color = [clusterscolor[i] for i in Labels]
#     principalDf['Labels'] = Labels_with_color
#     fig = px.scatter_3d(
#         principalDf,
#         x = 'PC1',
#         y = 'PC2',
#         z = z[:,0],
#         color = [clusterscolor[i] for i in sorted(Labels)],
#         color_discrete_sequence=["#93220a","green", "yellow", "red", "blue", "black"],
#     )
#     #Color to html Link : https://htmlcolorcodes.com/

#     fig.show()
#     #fig.write_image("pca.png")


def pca_visualize_3D(principalComponents, principalDf, Labels):
    clusterscolor = {0: 'R-FR', 1: 'G', 2: 'N', 3: 'G-R', 4: 'G-N', 5: 'G-FR'}
    Labels_with_color = [clusterscolor[i] for i in Labels]
    principalDf['Labels'] = Labels_with_color
    fig = px.scatter_3d(
        principalDf,
        x='PC1',
        y='PC2',
        z='PC3',  # Use the third principal component for the z-axis
        color=Labels_with_color,
        color_discrete_sequence=["#93220a", "green", "yellow", "red", "blue", "black"],
    )
    fig.show()
    # fig.write_image("pca.png")    


def process_data(path_to_dataset):
    """Basic wrapper for loading the given dataset using numpy"""
    data = np.load(path_to_dataset)
    return data



def get_label_arr(data):
    """Basic wrapper for accessing local integrated intensity array from data array"""
    label = data["Labels"]
    return label


def get_ohe_data(data):
    """Wrapper function that accesses one hot encoded array from data array and returns it as a pytorch tensor"""
    ohe_data = torch.from_numpy(data['ohe'])
    return ohe_data


def process_model(path_to_model):
    """Basic wrapper for loading the archived .pt pytorch model"""
    model = torch.load(path_to_model)
    return model


def get_z_from_latent_distribution(model, ohe_data):
    """Returns the array of data points in latent space z from the parametrized latent distribution latent_dist"""
    latent_dist = model.encode(ohe_data)
    z = latent_dist.loc.detach().numpy()
    return z


def preprocess_pca(z):
    """Preprocessing before pca visualization can occur (returns needed objects such as principal components, principal
    degress of freedom and labels"""

    # PCA
    pca = KernelPCA(n_components=4,kernel='rbf')
    principalComponents = pca.fit_transform(z)
    principalDf = pd.DataFrame(data=principalComponents, columns=['PC1', 'PC2', 'PC3', 'PC4'])
    # plt.scatter(principalComponents[:,0], principalComponents[:,1])
    labels = {
        str(i): f"PC {i + 1} ({var:.1f}%)"
        for i, var in enumerate(pca.explained_variance_ratio_ * 100)
    }
    return principalComponents, principalDf, labels



def conduct_visualizations(path_to_dataset: str, path_to_model):
    """Function for conducting visualizations of pca. Needs the path to the processed dataset you want to use (must
    use a dataset that is returned by process_data_file()), the path to the model you wish to use (has .pt extension)
    and boolean tuple representing whether you want to visualize lii, wavelength 2d or wavelength 3d. Type True if
    you want to produce the respective visualization, False otherwise."""
    data = process_data(path_to_dataset)
    model = process_model(path_to_model)

    ohe_data = get_ohe_data(data)
    z = get_z_from_latent_distribution(model, ohe_data)

    Labels = get_label_arr(data)

    principalComponents, principalDf, labels = preprocess_pca(z)


    
    pca_visualize_3D(principalComponents, principalDf, Labels)#, z)


# conduct_visualizations('data-for-sampling/past-samples-with-info/samples-1642783730.685655/pca-merged-1642783735.882887.npz',
#               'all-results/1-18-22-res/models/a19lds19b0.007g1.0d1.0h13.pt', (True, True, True))

conduct_visualizations('data-for-sampling/processed-data-files/processed-1710555155.767748.npz', 
'models/a20lds20b0.007g1d1h10.pt')


              


