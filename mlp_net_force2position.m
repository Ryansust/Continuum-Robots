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

%% 2. Define the Neural Network Architecture
% =========================================================================
disp('Step 2: Defining the Network Architecture...');

% A simple feed-forward network is perfect for this.
% We will create a network with one hidden layer of 15 neurons.
% 'feedforwardnet' is great for regression problems (predicting continuous values).
hiddenLayerSize = 9;
net = feedforwardnet(hiddenLayerSize);
%net.trainFcn = 'trainbr';

% For classification (like predicting the collision point), you would use:
% net = patternnet(hiddenLayerSize);

% Configure the network for our data
net = configure(net, inputs_normalized, targets_normalized);

%% 3. Split Data and Train the Network
% =========================================================================
disp('Step 3: Training the Network...');

% Set up the division of data for training, validation, testing
net.divideParam.trainRatio = 0.70; % 70% for training
net.divideParam.valRatio   = 0.15; % 15% for validation
net.divideParam.testRatio  = 0.15; % 15% for testing

% Train the network. This will open a training window showing progress.
% 'tr' contains the training record (e.g., performance over epochs).
[net, tr] = train(net, inputs_normalized, targets_normalized);

%% 4. Evaluate the Network's Performance (CORRECTED VERSION)
% =========================================================================
disp('Step 4: Evaluating the Network...');

% --- Get the indices for the test set ---
testIndices = tr.testInd;

% --- STEP 1: Get the original, un-normalized test data ---
% We need these for the final comparison.
testInputs_raw = inputs(:, testIndices);
testTargets_raw = targets(:, testIndices); % These are the ground truth in meters.

% --- STEP 2: Apply the SAME normalization to the test inputs ---
% The network MUST see the data in the same format it was trained on.
% We use 'mapminmax('apply', ...)' with the original settings 'ps_inputs'.
testInputs_normalized = mapminmax('apply', testInputs_raw, ps_inputs);

% --- STEP 3: Make predictions using the normalized inputs ---
predictions_normalized = net(testInputs_normalized);

% --- STEP 4: Un-normalize the predictions to get them back into meters ---
% We must reverse the normalization using the settings 'ps_targets'.
predictions_meters = mapminmax('reverse', predictions_normalized, ps_targets);

% --- Now we can properly calculate performance metrics in real units ---
mse_performance = perform(net, testTargets_raw, predictions_meters);
fprintf('Mean Squared Error (MSE) on Test Data: %f\n', mse_performance);

rmse_performance = sqrt(mse_performance);
fprintf('Root Mean Squared Error (RMSE) on Test Data: %f meters\n', rmse_performance);
fprintf('This means the average prediction error is about %.2f cm.\n', rmse_performance * 100);

% --- Visualize the Results ---
% The regression plot will now compare real targets vs. real predictions.
figure;
plotregression(testTargets_raw, predictions_meters);
title('Regression Plot: Actual vs. Predicted End-Effector Positions');

% The 3D scatter plot will now be meaningful.
figure;
hold on;
plot3(testTargets_raw(1,:), testTargets_raw(2,:), testTargets_raw(3,:), 'bo', 'MarkerFaceColor', 'b', 'DisplayName', 'Actual Positions');
plot3(predictions_meters(1,:), predictions_meters(2,:), predictions_meters(3,:), 'r+', 'MarkerSize', 10, 'LineWidth', 2, 'DisplayName', 'Predicted Positions');
title('3D View: Actual vs. Predicted Positions on Test Data');
xlabel('X (m)'); ylabel('Y (m)'); zlabel('Z (m)');
axis equal; grid on; view(30, 10);
legend;
hold off;
figure;
plot3(testTargets_raw(1,:), testTargets_raw(2,:), testTargets_raw(3,:), 'bo', 'MarkerFaceColor', 'b', 'DisplayName', 'Actual Positions');
axis equal; grid on; view(30, 10);legend;
figure;
plot3(predictions_meters(1,:), predictions_meters(2,:), predictions_meters(3,:), 'r+', 'MarkerSize', 10, 'LineWidth', 2, 'DisplayName', 'Predicted Positions');
axis equal; grid on; view(30, 10);legend;
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