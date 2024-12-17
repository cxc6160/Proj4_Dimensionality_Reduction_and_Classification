import argparse
from sklearn.model_selection import train_test_split
from data_loader import load_faces
from pca_utils import compute_pca, reconstruct_images, reconstruction_loss
from visualization import visualize_reconstruction
from config import DATA_DIR

def main(num_components):
    # Load the face dataset
    images, labels = load_faces(DATA_DIR)
    print(f"Loaded {len(images)} images with shape {images[0].shape}.")

    # Split into training and testing sets
    train_data, test_data, _, _ = train_test_split(images, labels, test_size=0.2, random_state=42)

    # Perform PCA
    mean_face, eigenvalues, eigenvectors = compute_pca(train_data)
    print("PCA computation completed.")

    # Reconstruct test images
    reconstructed_test = reconstruct_images(test_data, mean_face, eigenvectors, num_components)
    loss = reconstruction_loss(test_data, reconstructed_test)
    print(f"Reconstruction loss with {num_components} components: {loss:.4f}")

    # Visualize original vs reconstructed images
    visualize_reconstruction(test_data, reconstructed_test)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Face recognition with PCA.")
    parser.add_argument("num_components", type=int, help="Number of principal components to use.")
    args = parser.parse_args()

    main(args.num_components)
