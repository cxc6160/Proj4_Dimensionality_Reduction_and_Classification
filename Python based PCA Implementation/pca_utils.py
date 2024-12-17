import numpy as np

def compute_pca(data):
    mean_face = np.mean(data, axis=0)
    centered_data = data - mean_face
    covariance_matrix = np.dot(centered_data.T, centered_data) / len(data)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]
    return mean_face, eigenvalues, eigenvectors

def reconstruct_images(data, mean_face, eigenvectors, num_components):
    centered_data = data - mean_face
    components = eigenvectors[:, :num_components]
    projections = np.dot(centered_data, components)
    reconstructed = np.dot(projections, components.T) + mean_face
    return reconstructed

def reconstruction_loss(original, reconstructed):
    return np.mean((original - reconstructed) ** 2)
