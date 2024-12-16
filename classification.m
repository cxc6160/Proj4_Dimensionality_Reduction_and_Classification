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

    % Perform PCA to reduce dimensionality
    [projected_training_data, principal_components, mean_vector] = Principal_Component_Analysis(2, 200, training_data);

    % Preprocess testing data for PCA transformation
    [num_test_images, ~] = size(testing_data);
    testing_data = testing_data';
    centered_testing_data = testing_data - repmat(mean_vector, 1, num_test_images);
    projected_testing_data = principal_components' * centered_testing_data;

    if task == 1
        % Task 1: Face vs. Non-Face Classification
        degree = 3; % Degree for polynomial basis expansion

        % Assign binary labels: 1 for face, 2 for non-face
        binary_training_labels = ones(280, 1);

        % One-hot encode training labels
        one_hot_labels = bsxfun(@eq, binary_training_labels(:), 1:2)';
        
        % Apply basis expansion to training data
        expanded_training_data = basis_expansion(projected_training_data, degree, 1);

        % Train the model using pseudoinverse
        weights = (one_hot_labels * expanded_training_data') * pinv(expanded_training_data * expanded_training_data');

        % Preprocess and expand test data
        expanded_testing_data = basis_expansion(projected_testing_data, degree, 1);

        % Create test labels: 1 for face, 2 for non-face
        binary_test_labels = [ones(120, 1); ones(30, 1) * 2];

        % Predict using the trained model
        model_outputs = weights * expanded_testing_data;
        [~, predicted_labels] = max(model_outputs);

        % Calculate and display accuracy
        accuracy = sum(predicted_labels' == binary_test_labels) / length(binary_test_labels);
        disp(['Task 1 Accuracy: ', num2str(accuracy)]);

    else
        % Task 2: Subject Identification
        degree = 1; % Linear basis expansion

        % Filter out non-face images from the testing data
        projected_testing_data(:, 121:150) = [];

        % Assign subject labels for testing
        test_subject_labels = repelem((1:35)', 2);
        test_other_labels = ones(50, 1) * 36;
        testing_subject_labels = [test_subject_labels; test_other_labels];

        % Assign subject labels for training
        training_subject_labels = repelem((1:35)', 8);

        % One-hot encode training labels
        one_hot_labels = bsxfun(@eq, training_subject_labels(:), 1:36)';

        % Apply basis expansion to training data
        expanded_training_data = basis_expansion(projected_training_data, degree, 1);

        % Train the model using pseudoinverse
        weights = (one_hot_labels * expanded_training_data') * pinv(expanded_training_data * expanded_training_data');

        % Expand testing data
        expanded_testing_data = basis_expansion(projected_testing_data, degree, 1);

        % Predict using the trained model
        model_outputs = weights * expanded_testing_data;
        [~, predicted_labels] = max(model_outputs);

        % Calculate and display accuracy
        accuracy = sum(predicted_labels' == testing_subject_labels) / length(testing_subject_labels);
        disp(['Task 2 Accuracy: ', num2str(accuracy)]);
    end
end





