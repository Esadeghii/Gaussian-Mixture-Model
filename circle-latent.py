from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import KernelPCA
import pandas as pd
import numpy as np
import torch
import plotly.express as px
import plotly.graph_objects as go
from plotly.offline import plot

def preprocess_pca(z):
    """Preprocessing before PCA visualization can occur (returns needed objects such as principal components, principal
    components dataframe, and explained variance ratio)"""

    # Standardize the data
    scaler = StandardScaler()
    z_scaled = scaler.fit_transform(z)

    # Kernel PCA
    pca = KernelPCA(n_components=3, kernel='rbf')
    principalComponents = pca.fit_transform(z_scaled)
    explained_variance = np.var(principalComponents, axis=0)
    explained_variance_ratio = explained_variance / np.sum(explained_variance)
    principalDf = pd.DataFrame(data=principalComponents, columns=['PC1', 'PC2', 'PC3'])
    return principalComponents, principalDf, explained_variance_ratio

def calculate_centroids_and_std(principalDf, Labels):
    """Calculate centroids and standard deviations for each cluster."""
    principalDf['Labels'] = Labels
    centroids = principalDf.groupby('Labels').mean().reset_index()
    std_devs = principalDf.groupby('Labels').std().reset_index()
    return centroids, std_devs

def generate_circle(center, radius, color, opacity):
    """Generates a circle in 3D with a given center, radius, color, and opacity."""
    theta = np.linspace(0, 2 * np.pi, 100)
    x = radius * np.cos(theta) + center[0]
    y = radius * np.sin(theta) + center[1]
    z = np.full_like(theta, center[2])

    return go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode='lines',
        line=dict(color=color, width=2),
        showlegend=False
    )

def pca_visualize_3D_centroids_and_std(centroids, std_devs, explained_variance_ratio, circle_opacity=0.1):
    """Visualizes the centroids and their standard deviations as circles in 3D."""
    clusterscolor = {
        'R-FR': 'purple',
        'G': '#90EE90',
        'N': 'black',
        'G-R': 'orange',
        'G-N': '#006400',
        'G-FR': 'blue'
    }
    
    fig = go.Figure()

    # Add centroids and legends with explained variance ratios
    for i, row in centroids.iterrows():
        label = row['Labels']
        color = clusterscolor[label]
        fig.add_trace(go.Scatter3d(
            x=[row['PC1']],
            y=[row['PC2']],
            z=[row['PC3']],
            mode='markers',
            marker=dict(size=5, color=color, opacity=1),  # Adjusted size to make centroids smaller
            name=f"{label} (Var: x={explained_variance_ratio[0]:.2f}, y={explained_variance_ratio[1]:.2f}, z={explained_variance_ratio[2]:.2f})"
        ))

    # Add circles representing maximum standard deviations
    for i, row in centroids.iterrows():
        center = row[['PC1', 'PC2', 'PC3']].values
        radius = std_devs.loc[i, ['PC1', 'PC2', 'PC3']].max()
        color = clusterscolor[row['Labels']]
        fig.add_trace(generate_circle(center, radius, color, circle_opacity))

    plot(fig)  # Use plotly.offline.plot to display the plot in the browser

def process_data(path_to_dataset):
    """Loads the dataset from a .npz file."""
    data = np.load(path_to_dataset)
    return data

def get_label_arr(data):
    """Retrieves the label array from the dataset."""
    label = data["Labels"]
    return label

def get_ohe_data(data):
    """Retrieves the one-hot encoded data from the dataset and returns it as a PyTorch tensor."""
    ohe_data = torch.from_numpy(data['ohe'])
    return ohe_data

def process_model(path_to_model):
    """Loads a PyTorch model from a .pt file."""
    model = torch.load(path_to_model)
    return model

def get_z_from_latent_distribution(model, ohe_data):
    """Extracts the latent space representation from the model."""
    latent_dist = model.encode(ohe_data)
    z = latent_dist.loc.detach().numpy()
    return z

def conduct_visualizations(path_to_dataset: str, path_to_model, circle_opacity=0.2):
    """Conducts PCA visualizations using the given dataset and model."""
    data = process_data(path_to_dataset)
    model = process_model(path_to_model)
    ohe_data = get_ohe_data(data)
    z = get_z_from_latent_distribution(model, ohe_data)
    Labels = get_label_arr(data)
    principalComponents, principalDf, explained_variance_ratio = preprocess_pca(z)
    
    centroids, std_devs = calculate_centroids_and_std(principalDf, Labels)

    # Map numeric labels to string labels
    label_mapping = {0: 'R-FR', 1: 'G', 2: 'N', 3: 'G-R', 4: 'G-N', 5: 'G-FR'}
    centroids['Labels'] = centroids['Labels'].map(label_mapping)
    std_devs['Labels'] = std_devs['Labels'].map(label_mapping)

    pca_visualize_3D_centroids_and_std(centroids, std_devs, explained_variance_ratio, circle_opacity)

# Example of how to use the functions (make sure the paths are correct)
conduct_visualizations('data-for-sampling/processed-data-files/processed-1710555155.767748.npz', 'models/a20lds20b0.007g1d1h10.pt', circle_opacity=0.2)
