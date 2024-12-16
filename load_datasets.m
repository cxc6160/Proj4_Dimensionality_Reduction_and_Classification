function [labels, dataset] = load_datasets(dataset_path, is_training)
    % Load face datasets for training or testing
    %
    % Args:
    %   dataset_path: Root path to the dataset folders
    %   is_training: Boolean indicating whether to load training data (true) or testing data (false)
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
    images_per_subject_training = 8; % Number of images per subject for training
    num_training_subjects = 35; % Number of subjects for training
    total_subjects = 40; % Total number of subjects in the dataset
    image_size = [112, 92]; % Image dimensions (rows x columns)
    pixels_per_image = prod(image_size); % Total number of pixels per image
    
    % Initialize the dataset matrix
    if is_training
        num_training_images = num_training_subjects * images_per_subject_training;
        dataset = zeros(num_training_images, pixels_per_image);
        images_to_skip = 10 - images_per_subject_training;
    else
        num_testing_images = (total_subjects - num_training_subjects) * 10 ...
                             + num_training_subjects * 2 + 30; % Includes 'Other_Images'
        dataset = zeros(num_testing_images, pixels_per_image);
    end
    
    % Load training or testing data
    data_index = 1;
    for subject_idx = 1:length(subject_labels)
        current_path = fullfile(dataset_path, subject_labels(subject_idx));
        image_files = dir(current_path);
        
        if is_training && subject_idx <= num_training_subjects
            % Load training images
            for img_idx = 3:(length(image_files) - images_to_skip)
                image_path = fullfile(current_path, image_files(img_idx).name);
                dataset(data_index, :) = reshape(imread(image_path), 1, []);
                data_index = data_index + 1;
            end
        elseif ~is_training
            if subject_idx <= num_training_subjects
                % Load testing images from the training subjects (remaining images)
                for img_idx = (3 + images_per_subject_training):length(image_files)
                    image_path = fullfile(current_path, image_files(img_idx).name);
                    dataset(data_index, :) = reshape(imread(image_path), 1, []);
                    data_index = data_index + 1;
                end
            elseif subject_idx <= total_subjects
                % Load all images from the testing subjects
                for img_idx = 3:length(image_files)
                    image_path = fullfile(current_path, image_files(img_idx).name);
                    dataset(data_index, :) = reshape(imread(image_path), 1, []);
                    data_index = data_index + 1;
                end
            else
                % Load additional images from 'Other_Images'
                for img_idx = 3:length(image_files)
                    image_path = fullfile(current_path, image_files(img_idx).name);
                    dataset(data_index, :) = reshape(imread(image_path), 1, []);
                    data_index = data_index + 1;
                end
            end
        end
    end
    
    % Assign the labels
    labels = subject_labels;
end



