function main()
    % Executes tasks for eigenface visualization, face recognition, and identification
    % using PCA, KNN, and K-means clustering.

    % Step 1: Eigenfaces and Reconstructions
    execute_task(@plot_eigenfaces_and_reconstructions, ...
        "Step 1: Plotting eigenfaces and reconstructions for different numbers of eigenfaces...", ...
        "Eigenface visualization completed.");

    % Step 2: Face Recognition using PCA-based Classification
    recognition_labels = execute_classification(1, ...
        "Step 2: Performing Face Recognition using PCA-based Classification...", ...
        "Face Recognition Completed.");

    % Step 3: Face Identification using PCA-based Classification
    identification_labels = execute_classification(2, ...
        "Step 3: Performing Face Identification using PCA-based Classification...", ...
        "Face Identification Completed.");

    % Step 4: Face Recognition using KNN
    knn_recognition_labels = execute_knn_classification(1, ...
        "Step 4: Performing Face Recognition using KNN...", ...
        "KNN Face Recognition Completed.");

    % Step 5: Face Identification using KNN
    knn_identification_labels = execute_knn_classification(2, ...
        "Step 5: Performing Face Identification using KNN...", ...
        "KNN Face Identification Completed.");

    % Step 6: Face Recognition using K-means
    kmeans_recognition_indices = execute_kmeans_clustering(1, ...
        "Step 6: Performing Face Recognition using K-means Clustering...", ...
        "K-means Face Recognition Completed.");

    % Step 7: Face Identification using K-means
    kmeans_identification_indices = execute_kmeans_clustering(2, ...
        "Step 7: Performing Face Identification using K-means Clustering...", ...
        "K-means Face Identification Completed.");

    % Final Message
    disp("All tasks completed successfully!");
end

%% Helper Functions

function execute_task(task_function, start_message, end_message)
    % Generic executor for a single task with display messages.
    disp(start_message);
    task_function();
    disp(end_message);
end

function labels = execute_classification(task_number, start_message, end_message)
    % Executes classification tasks for face recognition or identification.
    disp(start_message);
    labels = classification_functions(task_number, 'classification'); 
    disp(end_message);
end

function labels = execute_knn_classification(task_number, start_message, end_message)
    % Executes KNN-based classification tasks.
    disp(start_message);
    labels = classification_functions(task_number, 'knn_classification'); 
    disp(end_message);
end

function indices = execute_kmeans_clustering(task_number, start_message, end_message)
    % Executes K-means clustering tasks.
    disp(start_message);
    indices = classification_functions(task_number, 'kmeans_clustering');
    disp(end_message);
end
