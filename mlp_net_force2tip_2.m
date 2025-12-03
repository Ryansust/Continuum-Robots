% Add this as the first or second line of your entire script
rng('default'); % Resets the random number generator to a predictable state
clc;
clear;
close all;

%% 1. Load and Prepare the Data
% =========================================================================
disp('Step 1: Loading and Preparing Data...');

% Load the data from your Excel file
FileName = '/Users/ryan/Desktop/continuum robot/force_data/after_processing_data_0816.xlsx';
dataTable = readtable(FileName);

% --- Define your Inputs (X - Features) ---
% Let's use the 6 tendon forces as our input features
% Columns 5 through 10 in your Excel file
input_features = dataTable{3:end, 23:28}; % Using {:} to get a matrix directly

% --- Define your Outputs (Y - Targets) ---
% We will predict the XYZ position of the last marker (marker #9)
position_text_array = dataTable{3:end, 29};
num_experiments = size(position_text_array, 1);
output_targets = zeros(num_experiments, 3); % N experiments, 3 targets (X, Y, Z)

for i = 1:num_experiments
   % Use your existing function to parse the marker data
   all_markers = get_RealOffset_1S3CT(position_text_array{i});
   % Extract the coordinates of the LAST marker
   output_targets(i, :) = all_markers(:, end)'; % Get last column, transpose to a row
end

% MATLAB's toolbox expects data in columns, not rows
% So we need to transpose our matrices
inputs = input_features';  % Should be size [6 x num_experiments]
targets = output_targets'; % Should be size [3 x num_experiments]
% Normalize inputs to the range [-1, 1]
[inputs_normalized, ps_inputs] = mapminmax(inputs);

% Normalize targets to the range [-1, 1] (also good practice)
[targets_normalized, ps_targets] = mapminmax(targets);

%% 2. Define Hyperparameters to Test
% =========================================================================
disp('Step 2: Defining Hyperparameters for Cross-Validation...');

% We will systematically test different numbers of neurons
%formal test shows that 25 is the best
neuron_counts = [5, 8, 12, 16, 20, 25, 30, 40]; 
K = 5; % Number of folds for cross-validation

% Array to store the average performance for each neuron count
avg_rmse_scores = zeros(length(neuron_counts), 1);
std_rmse_scores = zeros(length(neuron_counts), 1);

%% 3. Perform K-Fold Cross-Validation for Each Hyperparameter
% =========================================================================
disp('Step 3: Starting Cross-Validation Loop...');

% Outer loop: Iterate through each neuron count we want to test
for i = 1:length(neuron_counts)
    
    current_neurons = neuron_counts(i);
    fprintf('\n--- Testing with %d neurons ---\n', current_neurons);
    
    % Array to store the RMSE for each of the K folds
    fold_rmse_scores = zeros(K, 1);
    
    % Create the partition object that defines the K folds
    c = cvpartition(num_experiments, 'KFold', K);
    
    % Inner loop: Iterate through each of the K folds
    for k = 1:K
        
        fprintf('  Fold %d/%d...\n', k, K);
        
        % --- Get training and testing indices for this fold ---
        train_idx = training(c, k);
        test_idx = test(c, k);
        
        % --- Prepare the data for this specific fold ---
        train_inputs = inputs_normalized(:, train_idx);
        train_targets = targets_normalized(:, train_idx);
        
        test_inputs_raw = inputs(:, test_idx); % Un-normalized for final test
        test_targets_raw = targets(:, test_idx);
        
        % --- Create and configure a NEW network for this fold ---
        % It's crucial to re-initialize the network for each fold
        net = feedforwardnet(current_neurons);
        
        % We are not using the internal validation set, so set division to none
        net.divideFcn = 'dividetrain'; 
        
        % To hide the training window for each fold, uncomment the line below
        % net.trainParam.showWindow = false;
        
        % --- Train the network on this fold's training data ---
        [net, ~] = train(net, train_inputs, train_targets);
        
        % --- Test the network and calculate performance ---
        test_inputs_norm = mapminmax('apply', test_inputs_raw, ps_inputs);
        predictions_norm = net(test_inputs_norm);
        predictions_meters = mapminmax('reverse', predictions_norm, ps_targets);
        
        % Calculate and store the RMSE for this fold
        fold_rmse_scores(k) = sqrt(perform(net, test_targets_raw, predictions_meters));
    end
    
    % --- Calculate and store the average performance for this neuron count ---
    avg_rmse_scores(i) = mean(fold_rmse_scores);
    std_rmse_scores(i) = std(fold_rmse_scores);
    
    fprintf('  Average RMSE for %d neurons: %.4f m (%.2f cm) +/- %.4f\n', ...
        current_neurons, avg_rmse_scores(i), avg_rmse_scores(i)*100, std_rmse_scores(i));
end

%% 4. Analyze Results and Choose the Best Model
% =========================================================================
disp('Step 4: Analyzing Cross-Validation Results...');

% Find the number of neurons that gave the lowest average RMSE
[min_rmse, best_idx] = min(avg_rmse_scores);
best_num_neurons = neuron_counts(best_idx);

fprintf('\n--- Optimization Complete ---\n');
fprintf('Lowest Average RMSE: %.4f m (%.2f cm) was achieved with %d neurons.\n', ...
    min_rmse, min_rmse*100, best_num_neurons);

% Plot the results to visualize the performance
figure;
errorbar(neuron_counts, avg_rmse_scores * 100, std_rmse_scores * 100, 'o-', 'LineWidth', 2);
title('Hyperparameter Optimization via 5-Fold Cross-Validation');
xlabel('Number of Hidden Neurons');
ylabel('Average RMSE (cm)');
grid on;
xticks(neuron_counts);

%% 5. Train the Final Model
% =========================================================================
disp('Step 5: Training the final model on ALL data...');

% Now that we know the best number of neurons, we train one last model
% using ALL available data to make it as robust as possible.
final_net = feedforwardnet(best_num_neurons);
final_net.divideFcn = 'dividetrain'; % Train on everything

[final_net, ~] = train(final_net, inputs_normalized, targets_normalized);

disp('Final model is trained and ready. You can now save it.');
% save('final_robot_model_cv.mat', 'final_net', 'ps_inputs', 'ps_targets');
%% 5. How to Predict on New Data
% =========================================================================
disp('Step 5: Example of making a new prediction...');
% Once the network is trained, you can use it to predict the outcome for
% any new set of tendon forces.
% testInputs = [100; 0; 0; 0; 0; 1.0]; % Example 6x1 column vector
% [testInputs_normalized, ps_targets] = mapminmax(testInputs);
% predictions_normalized = net(testInputs_normalized); % Assuming you normalize testInputs too
% predictions = mapminmax('reverse', predictions_normalized, ps_targets);
disp('Predicted position for new tendon forces:');
% disp(predicted_position');