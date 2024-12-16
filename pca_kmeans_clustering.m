function [cluster_indices] = pca_kmeans_clustering(task)
    % Perform K-means clustering on the dataset.
    % Args:
    %   task: Task 1 (Face vs. Non-Face) or Task 2 (Face Subject Clustering)
    %
    % Returns:
    %   cluster_indices: Cluster indices assigned by K-means for the test data.

    % Load datasets
    [training_labels, training_data] = load_datasets('att_faces/', true);
    [testing_labels, testing_data] = load_datasets('att_faces/', false);

    % Perform PCA to reduce dimensionality
    [projected_training_data, principal_components, mean_vector] = Principal_Component_Analysis(2, 200, training_data);

    % Preprocess testing data for PCA transformation
    [num_test_samples, ~] = size(testing_data);
    testing_data = testing_data';
    centered_testing_data = testing_data - repmat(mean_vector, 1, num_test_samples);
    projected_testing_data = principal_components' * centered_testing_data;

    if task == 1
        % Task 1: Face vs. Non-Face Clustering
        num_clusters = 2; % 2 clusters: Face and Non-Face
        cluster_indices = kmeans(projected_testing_data', num_clusters);

    else
        % Task 2: Subject Clustering

        % Remove non-face data (images 121 to 150)
        projected_testing_data(:, 121:150) = [];

        % Perform K-means clustering for subjects
        num_clusters = 40; % 40 clusters: One for each subject
        cluster_indices = kmeans(projected_testing_data', num_clusters);
    end
end




