function plot_eigenfaces_and_reconstructions()
    % Load the dataset and perform PCA
    [labels, train_data] = load_datasets('att_faces/', true);
    
    % Perform PCA with 200 principal components
    [transformed_data, principal_components, mean_vector] = Principal_Component_Analysis(2, 200, train_data);

    % Visualize the eigenfaces
    figure;
    subplot(3,3,1), imshow(reshape(principal_components(:,1), [112,92]), []);
    title('1st Eigenface');

    subplot(3,3,2), imshow(reshape(principal_components(:,2), [112,92]), []);
    title('2nd Eigenface');

    subplot(3,3,3), imshow(reshape(principal_components(:,3), [112,92]), []);
    title('3rd Eigenface');

    subplot(3,3,4), imshow(reshape(principal_components(:,50), [112,92]), []);
    title('50th Eigenface');

    subplot(3,3,5), imshow(reshape(principal_components(:,100), [112,92]), []);
    title('100th Eigenface');

    subplot(3,3,6), imshow(reshape(principal_components(:,150), [112,92]), []);
    title('150th Eigenface');

    subplot(3,3,7), imshow(reshape(principal_components(:,200), [112,92]), []);
    title('200th Eigenface');
    
    % Reconstruct a sample image using different numbers of principal components
    figure;
    num_components_list = [250, 200, 150, 100, 50, 10];
    
    % Adjust subplot grid to accommodate 6 images
    num_rows = 2; % Number of rows in the subplot grid
    num_cols = 3; % Number of columns in the subplot grid
    
    for i = 1:length(num_components_list)
        % Recalculate PCA for the specified number of components
        [transformed_data, principal_components, mean_vector] = Principal_Component_Analysis(2, num_components_list(i), train_data);
        
        % Reconstruct the first image
        reconstructed_data = principal_components * transformed_data + mean_vector;
        
        % Display the reconstructed image
        subplot(num_rows, num_cols, i); % Update grid dimensions
        imshow(reshape(reconstructed_data(:,1), [112,92]), []);
        title(['Reconstructed with ', num2str(num_components_list(i)), ' PCs']);
    end

end



