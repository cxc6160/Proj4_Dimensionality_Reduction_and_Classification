function expanded_features = basis_expansion(input_data, degree, add_bias)
    % Expands input features to the specified polynomial degree.
    % Optionally adds a bias term (constant feature).
    %
    % Args:
    %   input_data: Matrix of size (num_features x num_samples)
    %   degree: Degree of polynomial expansion
    %   add_bias: Boolean indicating whether to add a bias term (1 for yes, 0 for no)
    %
    % Returns:
    %   expanded_features: Expanded feature matrix of size 
    %                      (num_features * degree + add_bias) x num_samples

    [num_features, num_samples] = size(input_data); % Input dimensions

    % Initialize expanded feature matrix
    expanded_features = zeros(num_features * degree + add_bias, num_samples);

    % Copy original features
    expanded_features(1:num_features, :) = input_data;

    % Add polynomial terms
    for d = 2:degree
        start_row = num_features * (d - 1) + 1;
        end_row = num_features * d;
        expanded_features(start_row:end_row, :) = input_data .^ d;
    end

    % Add bias term if required
    if add_bias == 1
        expanded_features(end, :) = 1; % Last row becomes the bias term
    end
end





