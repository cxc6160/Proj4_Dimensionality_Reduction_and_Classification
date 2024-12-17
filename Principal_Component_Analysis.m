function [transformed_data, principal_components, mean_vector, cumulative_variance] = Principal_Component_Analysis(method, num_components, input_data)
    % Perform Principal Component Analysis using Eigen decomposition or SVD
    % Args:
    %   method: Choose 1 for Eigen decomposition, 2 for SVD
    %   num_components: Number of principal components to retain
    %   input_data: Input data matrix (observations x features)
    %
    % Returns:
    %   transformed_data: Reduced data in the principal component space
    %   principal_components: Matrix of top principal components
    %   mean_vector: Mean of the features in the original data
    %   cumulative_variance: Cumulative variance explained by components
    
    % Transpose input data to align dimensions (features x observations)
    input_data = input_data';
    [num_features, num_observations] = size(input_data); % Dimensions of the dataset
    
    fprintf('Original data dimensions: %d features, %d observations\n', num_features, num_observations);
    
    % Step 1: Compute mean vector and subtract it from data
    mean_vector = mean(input_data, 2); % Compute mean of each feature
    centered_data = input_data - mean_vector; % Subtract mean to center the data
    
    % Choose the method: Eigen decomposition or SVD
    if method == 1
        fprintf('Using Eigen decomposition for PCA...\n');
        % Method 1: Eigen decomposition of covariance matrix
        covariance_matrix = (centered_data * centered_data') / (num_observations - 1);
        [eigen_vectors, eigen_values_matrix] = eig(covariance_matrix);
        
        % Extract and sort eigenvalues in descending order
        eigen_values = diag(eigen_values_matrix);
        [sorted_eigen_values, indices] = sort(eigen_values, 'descend');
        principal_components = eigen_vectors(:, indices); % Sort eigenvectors by eigenvalue
        
        % Select the top 'num_components' principal components
        principal_components = principal_components(:, 1:num_components);
        transformed_data = principal_components' * centered_data; % Project data
            
        % Calculate cumulative variance
        total_variance = sum(sorted_eigen_values); % Total variance (sum of eigenvalues)
        cumulative_variance = cumsum(sorted_eigen_values) / total_variance; % Cumulative variance explained by the components
    
    elseif method == 2
        fprintf('Using SVD for PCA...\n');
        % Method 2: Singular Value Decomposition (SVD)
        [U, S, ~] = svd(centered_data / (num_observations - 1), 'econ'); % Perform SVD
        principal_components = U(:, 1:num_components); % Retain top components
        transformed_data = principal_components' * centered_data; % Project data
        
        % Calculate cumulative variance for SVD
        singular_values = diag(S); % Singular values from SVD
        explained_variance = singular_values.^2; % Explained variance is the square of the singular values
        total_variance = sum(explained_variance); % Total variance
        cumulative_variance = cumsum(explained_variance) / total_variance; % Cumulative variance explained by the components
    else
        error('Invalid method selected. Choose 1 for Eigen decomposition or 2 for SVD.');
    end

    % Display results
    fprintf('Reduced dimensions: %d principal components retained\n', num_components);
    fprintf('Cumulative variance explained with %d components: %.4f%%\n', num_components, cumulative_variance(num_components) * 100);
    fprintf('Total variance captured: %.2f%%\n', cumulative_variance(end) * 100);
end
