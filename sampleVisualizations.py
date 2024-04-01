
# """
# Created on Thu Jan 20 20:03:10 2022

# @author: farihamoomtaheen
# """

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
from matplotlib import path
from sampleSequences import unpack_and_load_data, encode_data

df = pd.read_csv("data-for-sampling/past-samples-with-info/samples-1711985076.2931993/detailed-sequences", index_col = 0)


def Linechart(x, y1, y2, dimName):

# compare z0,z1 before and after re-encoding. Linechart
    plt.figure()
    plt.plot(x, y1, label= dimName + ' Generated')  
    plt.plot(x, y2, label= dimName + ' Encoded')
    
    plt.legend(loc="upper left")
    plt.xlabel("No of Samples")
    plt.ylabel(dimName)
    plt.title(dimName + " Comparison-Linechart")
    
    plt.show()
    
def Scatter(x, y1, y2, dimName):
    
# compare z0,z1 before and after re-encoding. Scatter
    plt.figure()    
    plt.scatter(x,y1,c='blue', label= dimName + ' Generated')
    plt.scatter(x,y2,c='red', label= dimName + ' Encoded')
    
    plt.legend(loc="upper left")
    plt.xlabel("No of Samples")
    plt.ylabel(dimName)
    plt.title(dimName + " Comparison-Scatterplot")
    
    plt.show()
    
# Wavelength plots
x = np.linspace(1, len(df), len(df))
y1 = df['Value Generated']
y2 = df['Value Encoded']
dimension = 'Value'

Linechart(x,y1,y2, dimension)
Scatter(x, y1, y2, dimension)




# fig = px.scatter(x, y1, color_discrete_sequence=['red'])
# fig = px.scatter(x, y2, color_discrete_sequence=['blue'])
# fig.show()

#compare difference in z0 before and after re-encoding. 
y = np.subtract(y1,y2)
plt.plot(x,y)
# plt.show()

plt.scatter(x,y, c='red')
plt.show()


def process_sequences_value(base_data_path: str):
    """Function used for writing original data to file"""
    df = pd.read_excel(base_data_path)
    label_arr = df['Label'].to_numpy()
    sequence_arr = df['Sequence'].to_numpy()

    return label_arr, sequence_arr


def write_clean_data(base_data_csv: str, processed_data_file: str, model: str, generated_file: str):
    """This function is used to write a csv file that has sequence, wavelength, lii, z-wavelength, z-lii information
    for a given model and dataset"""
    processed_data_file, model = unpack_and_load_data(processed_data_file, model)

    label_array = processed_data_file['label']
    ohe_sequences_tensor = processed_data_file['ohe']

    latent_dist = encode_data(ohe_sequences_tensor, model)

    mean = latent_dist.mean.detach().numpy()
    mean = np.array(mean)

    label_data, sequence_data = process_sequences_value(base_data_csv)

    with open(generated_file, 'w+') as f:
        f.write("Sequence,label,Z-Value\n")
        for i, matrix in enumerate(mean):
            f.write(f"{sequence_data[i]},{label_data[i]},{matrix[0]}\n") #dimension 0 is the z-value

#write_clean_data('data-for-sampling/past-samples-with-info/samples-1702884742.4237525-GREEN-DEC2023/generated-sequences.csv', 'data-for-sampling/past-samples-with-info/samples-1702884742.4237525-GREEN-DEC2023/generated-sequences--1702884750.56106.npz', 'models/weighted/a0.007lds15b0.007g1d1h13.pt', 'reencoded-data-info-model-4.csv')
write_clean_data('data-and-cleaning/supercleanGMMFilteredClusterd.xlsx', 'data-for-sampling/processed-data-files/clean-data-base-1711983115.9798691.npz', 'models/a20lds20b0.007g0.01d1h13.pt', 'data-info-model-4.csv')




