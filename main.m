function main()
    % Executes tasks for eigenface visualization, face recognition, and identification
    % using PCA, KNN, and K-means clustering.

    % Step 1: Plot eigenfaces and reconstructions
    disp("Step 1: Plotting eigenfaces and reconstructions for different numbers of eigenfaces...");
    plot_eigenfaces_and_reconstructions(); 

    % Step 2: Face Recognition and Identification using Classification
    disp("Step 2: Performing Face Recognition using PCA-based Classification...");
    [recognition_labels] = classification(1);
    disp("Face Recognition Completed.");

    disp("Step 3: Performing Face Identification using PCA-based Classification...");
    [identification_labels] = classification(2);
    disp("Face Identification Completed.");

    % Step 3: Face Recognition and Identification using KNN
    disp("Step 4: Performing Face Recognition using KNN...");
    [knn_recognition_labels] = pca_knn_classification(1);
    disp("KNN Face Recognition Completed.");

    disp("Step 5: Performing Face Identification using KNN...");
    [knn_identification_labels] = pca_knn_classification(2);
    disp("KNN Face Identification Completed.");

    % Step 4: Face Recognition and Identification using K-means
    disp("Step 6: Performing Face Recognition using K-means Clustering...");
    [kmeans_recognition_indices] = pca_kmeans_clustering(1);
    disp("K-means Face Recognition Completed.");

    disp("Step 7: Performing Face Identification using K-means Clustering...");
    [kmeans_identification_indices] = pca_kmeans_clustering(2);
    disp("K-means Face Identification Completed.");

    % Completion message
    disp("All tasks completed successfully!");
end





