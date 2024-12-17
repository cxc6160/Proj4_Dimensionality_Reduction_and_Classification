function [predicted_labels] = classification(task)
    % Perform classification on the dataset.
    % Args:
    %   task: Task 1 (Face vs. Non-Face) or Task 2 (Face Subject Identification)
    %
    % Returns:
    %   predicted_labels: Predicted class labels for the test data.

    % Load training and testing data
    [training_labels, training_data] = load_datasets('att_faces/', true);
    [testing_labels, testing_data] = load_datasets('att_faces/', false);

    % Perform PCA and preprocess data
    [projected_training_data, projected_testing_data] = preprocess_pca(training_data, testing_data);

    % Perform classification based on task
    if task == 1
        predicted_labels = task1_face_classification(projected_training_data, projected_testing_data);
    else
        predicted_labels = task2_subject_identification(projected_training_data, projected_testing_data);
    end
end

%% Helper Functions

function [train_data_pca, test_data_pca] = preprocess_pca(train_data, test_data)
    % Perform PCA to reduce dimensionality and preprocess testing data.
    [train_data_pca, principal_components, mean_vector] = Principal_Component_Analysis(2, 200, train_data);

    % Center and project testing data using the same PCA components
    [num_test_images, ~] = size(test_data);
    centered_test_data = test_data' - repmat(mean_vector, 1, num_test_images);
    test_data_pca = principal_components' * centered_test_data;
end

function predicted_labels = task1_face_classification(train_data, test_data)
    % Task 1: Face vs. Non-Face Classification

    % Polynomial basis expansion settings
    degree = 3;

    % Binary training labels: 1 for face, 2 for non-face
    binary_training_labels = ones(280, 1);
    one_hot_labels = bsxfun(@eq, binary_training_labels(:), 1:2)';

    % Expand training and testing data
    expanded_train_data = basis_expansion(train_data, degree, 1);
    expanded_test_data = basis_expansion(test_data, degree, 1);

    % Train the model
    weights = train_model_pseudoinverse(expanded_train_data, one_hot_labels);

    % Binary test labels: 1 for face, 2 for non-face
    binary_test_labels = [ones(120, 1); ones(30, 1) * 2];

    % Predict test data
    model_outputs = weights * expanded_test_data;
    [~, predicted_labels] = max(model_outputs);

    % Display accuracy
    accuracy = calculate_accuracy(predicted_labels', binary_test_labels);
    disp(['Task 1 Accuracy: ', num2str(accuracy)]);
end

function predicted_labels = task2_subject_identification(train_data, test_data)
    % Task 2: Subject Identification

    % Basis expansion (linear)
    degree = 1;

    % Filter out non-face images from the test set
    test_data(:, 121:150) = [];

    % Assign labels for training and testing data
    training_subject_labels = repelem((1:35)', 8);
    test_subject_labels = repelem((1:35)', 2);
    test_other_labels = ones(50, 1) * 36;
    testing_subject_labels = [test_subject_labels; test_other_labels];

    % One-hot encode training labels
    one_hot_labels = bsxfun(@eq, training_subject_labels(:), 1:36)';

    % Expand training and testing data
    expanded_train_data = basis_expansion(train_data, degree, 1);
    expanded_test_data = basis_expansion(test_data, degree, 1);

    % Train the model
    weights = train_model_pseudoinverse(expanded_train_data, one_hot_labels);

    % Predict test data
    model_outputs = weights * expanded_test_data;
    [~, predicted_labels] = max(model_outputs);

    % Display accuracy
    accuracy = calculate_accuracy(predicted_labels', testing_subject_labels);
    disp(['Task 2 Accuracy: ', num2str(accuracy)]);
end

function weights = train_model_pseudoinverse(data, labels)
    % Train the model using pseudoinverse
    weights = (labels * data') * pinv(data * data');
end

function accuracy = calculate_accuracy(predictions, ground_truth)
    % Calculate accuracy
    accuracy = sum(predictions == ground_truth) / length(ground_truth);
end



