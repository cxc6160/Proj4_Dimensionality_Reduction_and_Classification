function plot_eigenfaces_and_reconstructions()
    % Load the dataset and perform PCA
    [labels, train_data] = load_datasets('att_faces/', true);
    
    % Perform PCA with 200 principal components
    [transformed_data, principal_components, mean_vector] = Principal_Component_Analysis(2, 200, train_data);

    % Visualize the first four eigenfaces
    figure;
    subplot(2,2,1), imshow(reshape(principal_components(:,1), [112,92]), []);
    title('1st Eigenface');
    
    subplot(2,2,2), imshow(reshape(principal_components(:,2), [112,92]), []);
    title('2nd Eigenface');
    
    subplot(2,2,3), imshow(reshape(principal_components(:,3), [112,92]), []);
    title('3rd Eigenface');
    
    subplot(2,2,4), imshow(reshape(principal_components(:,200), [112,92]), []);
    title('200th Eigenface');
    
    % Reconstruct a sample image using different numbers of principal components
    figure;
    num_components_list = [200, 150, 100, 50];
    
    for i = 1:length(num_components_list)
        % Recalculate PCA for the specified number of components
        [transformed_data, principal_components, mean_vector] = Principal_Component_Analysis(2, num_components_list(i), train_data);
        
        % Reconstruct the first image
        reconstructed_data = principal_components * transformed_data + mean_vector;
        
        % Display the reconstructed image
        subplot(2,2,i);
        imshow(reshape(reconstructed_data(:,1), [112,92]), []);
        title(['Reconstructed with ', num2str(num_components_list(i)), ' PCs']);
    end
end




