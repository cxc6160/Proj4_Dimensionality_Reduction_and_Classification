function [labels, dataset] = load_dataset_pca(dataset_path)
    % Load all face datasets
    %
    % Args:
    %   dataset_path: Root path to the dataset folders
    %
    % Returns:
    %   labels: Array of subject labels
    %   dataset: Matrix where each row is a flattened image
    
    % Subject labels
    subject_labels = ["s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10", ...
                      "s11", "s12", "s13", "s14", "s15", "s16", "s17", "s18", "s19", ...
                      "s20", "s21", "s22", "s23", "s24", "s25", "s26", "s27", "s28", ...
                      "s29", "s30", "s31", "s32", "s33", "s34", "s35", "s36", "s37", ...
                      "s38", "s39", "s40", 'Other_Images'];
    
    % Configuration
    image_size = [112, 92]; % Image dimensions (rows x columns)
    pixels_per_image = prod(image_size); % Total number of pixels per image
    
    % Initialize the dataset matrix
    dataset = [];
    
    % Load all images from all subjects
    for subject_idx = 1:length(subject_labels)
        current_path = fullfile(dataset_path, subject_labels(subject_idx));
        image_files = dir(current_path);
        
        % Loop through all the images for the current subject
        for img_idx = 3:length(image_files) % Skip '.' and '..'
            image_path = fullfile(current_path, image_files(img_idx).name);
            img = imread(image_path);
            img = double(img);
            dataset = [dataset; reshape(img, 1, pixels_per_image)];
        end
    end
    
    % Assign the labels (same for all images, based on the order of the subjects)
    labels = subject_labels;
end
