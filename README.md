# Dimensionality Reduction and Classification Project 4

This project was done to fulfill the project 4 requirements for the course CISC 820 - Quantitative Foundations.

In this project, we experiment with the application of dimensionality reduction and classification methods in face recognition. Given a dataset of face images, which consists of 10 different images for each of the 40 different subjects, we use PCA to extract the eigenfaces and
use them in combination with linear classification methods for image reconstruction, face recognition, and face identification.


## Reproducing the Project

To reproduce this project, follow these steps:

### Clone the repository:
   ```bash
   git clone https://github.com/cxc6160/Proj4_Dimensionality_Reduction_and_Classification.git
   ```

### Make sure this is the working directory in matlab.
 
### Running the script:
Run `main.m` to produce results.
```bash
    main;
```

### Ensure the following files and folders are in the working directory:

1. `plot_eigenfaces_and_reconstructions.m`
2. `basis_expansion.m`
3. `classification.m`
4. `pca_knn_classification.m`
5. `pca_kmeans_clustering.m`
6. `Principal_Component_Analysis.m`
7. `load_datasets.m`
8. `main.m`
9. `att_faces` (image datasets folder)
10. `load_dataset_pca.m`

## Some of the plots and results can be found in the [Artifacts](./Artifacts/) folder.

## Authors
1. Image Adhikari
2. Ester Chen