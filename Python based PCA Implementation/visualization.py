import matplotlib.pyplot as plt

def visualize_reconstruction(test_data, reconstructed_data, image_shape=(112, 92)):
    plt.figure(figsize=(10, 5))
    for i in range(5):
        plt.subplot(2, 5, i + 1)
        plt.imshow(test_data[i].reshape(image_shape), cmap='gray')
        plt.title("Original")
        plt.axis("off")

        plt.subplot(2, 5, i + 6)
        plt.imshow(reconstructed_data[i].reshape(image_shape), cmap='gray')
        plt.title("Reconstructed")
        plt.axis("off")
    plt.tight_layout()
    plt.show()
