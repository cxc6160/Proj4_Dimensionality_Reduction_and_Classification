function [predicted_labels, distances] = pca_knn_classification(task)
    % Perform KNN-based classification on the dataset.
    % Args:
    %   task: Task 1 (Face vs. Non-Face) or Task 2 (Face Subject Identification)
    %
    % Returns:
    %   predicted_labels: Predicted class labels for the test data.
    %   distances: Distances from the nearest neighbors in the KNN search.

    % Load datasets
    [training_labels, training_data] = load_datasets('att_faces/', true);
    [testing_labels, testing_data] = load_datasets('att_faces/', false);

    % Apply PCA for dimensionality reduction
    [projected_training_data, principal_components, mean_vector] = Principal_Component_Analysis(2, 200, training_data);

    % Preprocess testing data
    [num_test_samples, ~] = size(testing_data);
    testing_data = testing_data';
    centered_testing_data = testing_data - repmat(mean_vector, 1, num_test_samples);
    projected_testing_data = principal_components' * centered_testing_data;

    if task == 1
        % Task 1: Face vs. Non-Face Classification

        % Set the threshold for face vs. non-face classification
        threshold = 4270.8; % Pre-calculated threshold (based on validation)

        % Perform KNN search
        [neighbor_indices, distances] = knnsearch(projected_training_data', projected_testing_data');

        % Assign initial class labels (all face by default)
        predicted_labels = ones(num_test_samples, 1);

        % Assign "non-face" label (2) for distances exceeding the threshold
        predicted_labels(distances > threshold) = 2;

        % Define ground truth test labels (1 for face, 2 for non-face)
        ground_truth_labels = [ones(120, 1); ones(30, 1) * 2];

        % Calculate accuracy
        accuracy = sum(predicted_labels == ground_truth_labels) / length(ground_truth_labels);
        disp(['Task 1 Accuracy: ', num2str(accuracy)]);

    else
        % Task 2: Subject Identification

        % Remove non-face images from the testing data
        projected_testing_data(:, 121:150) = [];

        % Define thresholds for subject classification
        threshold = 3578.0; % Pre-calculated threshold (based on validation)

        % Define ground truth test labels
        subject_test_labels = repelem((1:35)', 2);
        non_face_test_labels = ones(50, 1) * 36; % Label for non-face images
        ground_truth_labels = [subject_test_labels; non_face_test_labels];

        % Define training labels
        subject_train_labels = repelem((1:35)', 8);

        % Perform KNN search
        [neighbor_indices, distances] = knnsearch(projected_training_data', projected_testing_data');

        % Assign labels based on the nearest neighbor in the training set
        predicted_labels = subject_train_labels(neighbor_indices);

        % Assign "non-face" label (36) for distances exceeding the threshold
        predicted_labels(distances > threshold) = 36;

        % Calculate accuracy
        accuracy = sum(predicted_labels == ground_truth_labels) / length(ground_truth_labels);
        disp(['Task 2 Accuracy: ', num2str(accuracy)]);
    end
end




