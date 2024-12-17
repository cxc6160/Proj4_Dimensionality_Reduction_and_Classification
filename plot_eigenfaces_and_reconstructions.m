function plot_eigenfaces_and_reconstructions()
function plot_eigenfaces_and_reconstructions()
    % Load the dataset and perform PCA
    [labels, train_data] = load_dataset_pca('att_faces/');
    
    % Perform PCA with 400 principal components
    [transformed_data, principal_components, mean_vector, cumulative_variance] = Principal_Component_Analysis(2, 400, train_data);

    % Visualize the eigenfaces
    figure;
    subplot(2,3,1), imshow(reshape(principal_components(:,1), [112,92]), []);
    title('1st Eigenface');
    
    subplot(2,3,2), imshow(reshape(principal_components(:,2), [112,92]), []);
    title('2nd Eigenface');
    
    subplot(2,3,3), imshow(reshape(principal_components(:,3), [112,92]), []);
    title('3rd Eigenface');
    
    subplot(2,3,4), imshow(reshape(principal_components(:,4), [112,92]), []);
    title('4th Eigenface');

    subplot(2,3,5), imshow(reshape(principal_components(:,5), [112,92]), []);
    title('5th Eigenface');

    subplot(2,3,6), imshow(reshape(principal_components(:,6), [112,92]), []);
    title('6th Eigenface');
    
    % Reconstruct a sample image using different numbers of principal components
    figure;
    num_components_list = [400, 350, 300, 200, 100, 50, 10];

    % Initialize an array to store the reconstruction loss
    reconstruction_losses = zeros(1, length(num_components_list));
    
    % Display the original image in the first position
    subplot(2,4,1);
    imshow(reshape(train_data(1,:), [112,92]), []);
    title('Original Image','FontSize', 12);
    
    for i = 1:length(num_components_list)
        [transformed_data, principal_components, mean_vector] = Principal_Component_Analysis(2, num_components_list(i), train_data);
        reconstructed_data = principal_components * transformed_data + mean_vector;

        % Calculate the reconstruction loss (mean squared error)
        original_image = reshape(train_data(1,:), [112, 92]);  % Original image (first image in the dataset)
        reconstructed_image = reshape(reconstructed_data(:,1), [112, 92]);  % Reconstructed image
        
        % Compute the mean squared error (MSE)
        mse = mean((original_image - reconstructed_image).^2, 'all');
        reconstruction_losses(i) = mse;

        subplot(2,4,i+1);
        imshow(reshape(reconstructed_data(:,1), [112,92]), []);
        title(['Reconstructed: ', num2str(num_components_list(i)), ' PCs'], 'FontSize', 12);
    end

    % Plot the reconstruction loss for different numbers of principal components
    figure;
    plot(num_components_list, reconstruction_losses, '-o', 'LineWidth', 2, 'MarkerSize', 8);
    xlabel('Number of Principal Components');
    ylabel('Reconstruction Loss (MSE)');
    title('Reconstruction Loss for Different Numbers of Principal Components');
    grid on; 

    % Plot the total variance against the number of eigenfaces
    figure;
    plot(cumulative_variance(1:400), 'LineWidth', 2);
    xlabel('Number of Principal Components');
    ylabel('Cumulative Variance Explained');
    title('Cumulative Variance Explained by Principal Components');
    grid on; 

    % Save the reconstruction losses to a CSV file
    results = [num_components_list', reconstruction_losses'];
    csv_filename = 'Artifacts/reconstruction_losses.csv';
    writematrix(results, csv_filename);
    disp(['Reconstruction losses saved to ', csv_filename]);
end