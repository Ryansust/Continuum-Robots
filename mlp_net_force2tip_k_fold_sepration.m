% Add this as the first or second line of your entire script
%rng('default'); % Resets the random number generator to a predictable state
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
input_features = dataTable{3:end, 23:28}; % Using corrected columns 5-10

% --- Define your Outputs (Y - Targets) ---
position_text_array = dataTable{3:end, 29};
num_experiments = size(position_text_array, 1);
output_targets = zeros(num_experiments, 3);

for i = 1:num_experiments
   all_markers = get_RealOffset_1S3CT(position_text_array{i});
   output_targets(i, :) = all_markers(:, end)';
end

% Transpose for MATLAB's toolbox
inputs = input_features';
targets = output_targets';

% Normalize ALL data at once
[inputs_normalized, ps_inputs] = mapminmax(inputs);
[targets_normalized, ps_targets] = mapminmax(targets);

%% 2. Split Data Using a K-Fold Partition
% =========================================================================
disp('Step 2: Splitting data into Training and Final Test sets...');

% Use cvpartition for a robust, single split.
% We will hold out 20% of the data for a final, unbiased test.
cv = cvpartition(num_experiments, 'HoldOut', 0.20);

% Get the indices for the training set (80%) and test set (20%)
train_idx = cv.training;
test_idx = cv.test;

% Create the training set (this will be given to the network)
train_inputs_norm = inputs_normalized(:, train_idx);
train_targets_norm = targets_normalized(:, train_idx);

% Create the final test set (this will be used only for evaluation)
test_inputs_norm = inputs_normalized(:, test_idx);
test_targets_raw = targets(:, test_idx); % Keep raw targets for final RMSE

%% 3. Define the Neural Network Architecture
% =========================================================================
disp('Step 3: Defining the Network Architecture...');

hiddenLayerSize = 16; % Use your optimized number here
net = feedforwardnet(hiddenLayerSize);

% --- Configure the network's INTERNAL data division ---
% The network will now take the 80% of data we give it and split it
% further for training and validation (for early stopping).
% 85% of 80% is ~70% of total for training. 15% of 80% is ~12% for validation.
net.divideFcn = 'dividerand'; % Use the network's default random divider
net.divideParam.trainRatio = 0.85;
net.divideParam.valRatio   = 0.15;
net.divideParam.testRatio  = 0.0; % We are using our own final test set

%% 4. Train the Network
% =========================================================================
disp('Step 4: Training the Network...');

% Train the network ONLY on the larger training partition.
[net, tr] = train(net, train_inputs_norm, train_targets_norm);

%% 5. Evaluate the Network's Performance on the Hold-Out Test Set
% =========================================================================
disp('Step 5: Evaluating the Network on the final hold-out test set...');

% Make predictions using the final test inputs
predictions_normalized = net(test_inputs_norm);

% Un-normalize the predictions to get them back into meters
predictions_meters = mapminmax('reverse', predictions_normalized, ps_targets);

% --- Now we can properly calculate performance metrics in real units ---
mse_performance = perform(net, test_targets_raw, predictions_meters);
fprintf('Mean Squared Error (MSE) on Final Test Data: %f\n', mse_performance);

rmse_performance = sqrt(mse_performance);
fprintf('Root Mean Squared Error (RMSE) on Final Test Data: %f meters\n', rmse_performance);
fprintf('This means the average prediction error is about %.2f cm.\n', rmse_performance * 100);

% --- Visualize the Results ---
figure;
plotregression(test_targets_raw, predictions_meters);
title('Regression Plot on Final Test Data');

figure;
hold on;
plot3(test_targets_raw(1,:), test_targets_raw(2,:), test_targets_raw(3,:), 'bo', 'MarkerFaceColor', 'b', 'DisplayName', 'Actual Positions');
plot3(predictions_meters(1,:), predictions_meters(2,:), predictions_meters(3,:), 'r+', 'MarkerSize', 10, 'LineWidth', 2, 'DisplayName', 'Predicted Positions');
title('3D View: Actual vs. Predicted on Final Test Data');
xlabel('X (m)'); ylabel('Y (m)'); zlabel('Z (m)');
axis equal; grid on; view(30, 10);
legend;
hold off;
figure;
plot3(test_targets_raw(1,:), test_targets_raw(2,:), test_targets_raw(3,:), 'bo', 'MarkerFaceColor', 'b', 'DisplayName', 'Actual Positions');
axis equal; grid on; view(30, 10);legend;
figure;
plot3(predictions_meters(1,:), predictions_meters(2,:), predictions_meters(3,:), 'r+', 'MarkerSize', 10, 'LineWidth', 2, 'DisplayName', 'Predicted Positions');
axis equal; grid on; view(30, 10);legend;
%% 6. Example of making a new prediction...
% This section would remain the same, using the trained 'net' object.