function varargout = classification_functions(task, method)
    % Perform classification or clustering based on the selected method and task.
    %
    % Args:
    %   task: Task 1 (Face vs. Non-Face) or Task 2 (Face Subject Identification/Clustering)
    %   method: 'classification', 'kmeans_clustering', or 'knn_classification'
    %
    % Returns:
    %   Outputs depend on the method:
    %     - For classification: Predicted labels
    %     - For K-means: Cluster indices
    %     - For KNN: Predicted labels and distances

    switch method
        case 'classification'
            varargout{1} = perform_classification(task);
        case 'kmeans_clustering'
            varargout{1} = perform_kmeans_clustering(task);
        case 'knn_classification'
            [varargout{1}, varargout{2}] = perform_knn_classification(task);
        otherwise
            error('Invalid method. Choose "classification", "kmeans_clustering", or "knn_classification".');
    end
end

%% ---------------- Classification Function ----------------
function [predicted_labels] = perform_classification(task)
    % Perform classification (Face vs Non-Face or Subject Identification)

    % Load training and testing data
    [~, training_data] = load_datasets('att_faces/', true);
    [~, testing_data] = load_datasets('att_faces/', false);

    % Perform PCA and preprocess
    [projected_training_data, projected_testing_data] = preprocess_pca(training_data, testing_data);

    if task == 1
        predicted_labels = task1_face_classification(projected_training_data, projected_testing_data);
    else
        predicted_labels = task2_subject_identification(projected_training_data, projected_testing_data);
    end
end

%% ---------------- K-means Clustering Function ----------------
function [cluster_indices] = perform_kmeans_clustering(task)
    % Perform K-means clustering on PCA-reduced data.

    % Load datasets
    [~, training_data] = load_datasets('att_faces/', true);
    [~, testing_data] = load_datasets('att_faces/', false);

    % Apply PCA
    [projected_training_data, principal_components, mean_vector] = Principal_Component_Analysis(2, 200, training_data);

    % Preprocess testing data
    [num_test_samples, ~] = size(testing_data);
    centered_testing_data = testing_data' - repmat(mean_vector, 1, num_test_samples);
    projected_testing_data = principal_components' * centered_testing_data;

    if task == 1
        cluster_indices = kmeans(projected_testing_data', 2); % Face vs Non-Face
    else
        projected_testing_data(:, 121:150) = []; % Remove non-face
        cluster_indices = kmeans(projected_testing_data', 40); % Subject Clustering
    end
end

%% ---------------- KNN Classification Function ----------------
function [predicted_labels, distances] = perform_knn_classification(task)
    % Perform KNN classification for Face vs Non-Face or Subject Identification.

    % Load datasets
    [~, training_data] = load_datasets('att_faces/', true);
    [~, testing_data] = load_datasets('att_faces/', false);

    % Perform PCA
    [projected_training_data, principal_components, mean_vector] = Principal_Component_Analysis(2, 200, training_data);

    % Preprocess testing data
    [num_test_samples, ~] = size(testing_data);
    centered_testing_data = testing_data' - repmat(mean_vector, 1, num_test_samples);
    projected_testing_data = principal_components' * centered_testing_data;

    if task == 1
        threshold = 4270.8;
        [neighbor_indices, distances] = knnsearch(projected_training_data', projected_testing_data');
        predicted_labels = ones(num_test_samples, 1);
        predicted_labels(distances > threshold) = 2; % Label non-face
    else
        threshold = 3578.0;
        projected_testing_data(:, 121:150) = [];
        subject_train_labels = repelem((1:35)', 8);
        [neighbor_indices, distances] = knnsearch(projected_training_data', projected_testing_data');
        predicted_labels = subject_train_labels(neighbor_indices);
        predicted_labels(distances > threshold) = 36; % Label non-face
    end
end

%% ---------------- Helper Functions ----------------
function [train_data_pca, test_data_pca] = preprocess_pca(train_data, test_data)
    % Apply PCA for dimensionality reduction.
    [train_data_pca, principal_components, mean_vector] = Principal_Component_Analysis(2, 200, train_data);

    % Center and project test data
    [num_test_images, ~] = size(test_data);
    centered_test_data = test_data' - repmat(mean_vector, 1, num_test_images);
    test_data_pca = principal_components' * centered_test_data;
end

function predicted_labels = task1_face_classification(train_data, test_data)
    % Task 1: Face vs Non-Face Classification
    degree = 3;
    binary_training_labels = ones(280, 1);
    one_hot_labels = bsxfun(@eq, binary_training_labels(:), 1:2)';
    expanded_train_data = basis_expansion(train_data, degree, 1);
    expanded_test_data = basis_expansion(test_data, degree, 1);
    weights = train_model_pseudoinverse(expanded_train_data, one_hot_labels);
    binary_test_labels = [ones(120, 1); ones(30, 1) * 2];
    model_outputs = weights * expanded_test_data;
    [~, predicted_labels] = max(model_outputs);
    disp(['Task 1 Accuracy: ', num2str(calculate_accuracy(predicted_labels', binary_test_labels))]);
end

function predicted_labels = task2_subject_identification(train_data, test_data)
    % Task 2: Subject Identification
    degree = 1;
    test_data(:, 121:150) = [];
    training_subject_labels = repelem((1:35)', 8);
    one_hot_labels = bsxfun(@eq, training_subject_labels(:), 1:36)';
    expanded_train_data = basis_expansion(train_data, degree, 1);
    expanded_test_data = basis_expansion(test_data, degree, 1);
    weights = train_model_pseudoinverse(expanded_train_data, one_hot_labels);
    model_outputs = weights * expanded_test_data;
    [~, predicted_labels] = max(model_outputs);
    disp(['Task 2 Accuracy: ', num2str(calculate_accuracy(predicted_labels', [repelem((1:35)', 2); ones(50, 1) * 36]))]);
end

function weights = train_model_pseudoinverse(data, labels)
    % Train model using pseudoinverse
    weights = (labels * data') * pinv(data * data');
end

function accuracy = calculate_accuracy(predictions, ground_truth)
    % Calculate accuracy
    accuracy = sum(predictions == ground_truth) / length(ground_truth);
end
