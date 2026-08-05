%% ========================================================================
%  Project: Deep Learning-Based Robust Proprioceptive Pose Reconstruction
%  Author:  Lin Yongxi (Optimized - Full Version)
% =========================================================================
clc; clear; close all;
rng('default'); % Ensure reproducibility 

%% ========================================================================
%  Step 1: Data Loading, ROI Filtering & Cleaning
% =========================================================================
disp('--------------------------------------------------');
disp('1. Loading and preprocessing data...');

% 1.1 Load Data
FileName = '/Users/ryan/Desktop/continuum robot/force_data/after_processing_data_0816.xlsx';
if ~isfile(FileName), error('File not found!'); end
dataTable = readtable(FileName);

% Establish tracking array for Excel row numbers (starting from row 3)
track_rows_raw = (3:height(dataTable))'; 

% Unit conversion: Tension must be multiplied by 0.00981 to convert to Newtons (N)
conversion_factor = 0.00981;
F_after_raw  = (double(table2array(dataTable(3:end, 23:28))) * conversion_factor)';  
F_before_raw = (double(table2array(dataTable(3:end, 11:16))) * conversion_factor)';  

raw_mag_raw  = double(abs(table2array(dataTable(3:end, 2))))'; 
raw_dir_raw  = double(table2array(dataTable(3:end, 3)))';      
raw_hgt_raw  = double(table2array(dataTable(3:end, 4)))'; 

% Assuming text columns 38 (before) and 29 (after) for 3D coordinates
pos_text_before_raw = dataTable{3:end, 38}; 
pos_text_after_raw  = dataTable{3:end, 29}; 

% 1.3 ROI Filtering (Keep Nodes 3, 4, 5)
disp('   > Executing ROI filtering (Nodes 3, 4, 5)...');
roi_mask = ismember(raw_hgt_raw,[3, 4, 5]);

F_after_sub  = F_after_raw(:, roi_mask);
F_before_sub = F_before_raw(:, roi_mask);
raw_mag_sub  = raw_mag_raw(roi_mask);
raw_dir_sub  = raw_dir_raw(roi_mask);
raw_hgt_sub  = raw_hgt_raw(roi_mask);
pos_text_b_sub = pos_text_before_raw(roi_mask);
pos_text_a_sub = pos_text_after_raw(roi_mask);
track_rows_sub = track_rows_raw(roi_mask); % Synchronize tracking

if length(raw_mag_sub) < 50, error('Insufficient data after ROI filtering.'); end

% 1.4 Kinematics Parsing & Ground Truth Generation
disp('   > Parsing pose data and generating ground truth...');
N_sub = length(raw_mag_sub);
P_before_ideal = zeros(21, N_sub); 
P_after_sensor = zeros(21, N_sub);
gt_F_vec = zeros(3, N_sub);

for i = 1:N_sub
    % [Core Constraint] Parse string to matrix using external function
    offset_b = get_RealOffset_1S3CT(pos_text_b_sub{i});
    offset_a = get_RealOffset_1S3CT(pos_text_a_sub{i});
    
    % Base-centering: Align origin (0,0,0) to the midpoint of the first two markers
    base_center_b = (offset_b(:, 1) + offset_b(:, 2)) / 2;
    P_before_ideal(:, i) = reshape(offset_b(:, 3:end) - base_center_b,[], 1); 
    
    base_center_a = (offset_a(:, 1) + offset_a(:, 2)) / 2;
    P_after_sensor(:, i) = reshape(offset_a(:, 3:end) - base_center_a,[], 1); 
    
    % Calculate Ground Truth Force Vector
    u_vec = [0;0;0];
    switch raw_dir_sub(i)
        case 2, u_vec = [-1; 0; 0];
        case 3, u_vec = [-sind(45); cosd(45); 0];
        case 4, u_vec = [0; 1; 0];
    end
    gt_F_vec(:, i) = raw_mag_sub(i) * u_vec;
end

% 1.5 Data Cleaning: NaN & 10mm geometric distortion detection
disp('   > Executing Auto-Cleaning (NaNs & 10mm geometric distortion detection)...');
bad_idx = any(isnan(F_after_sub), 1) | any(isnan(F_before_sub), 1) | isnan(raw_mag_sub) | isnan(raw_dir_sub) | isnan(raw_hgt_sub);

% Manually remove known outliers (e.g., optical tracking failures)
known_outliers = [686]; 
if ~isempty(known_outliers)
    [~, loc_outliers] = ismember(known_outliers, track_rows_sub);
    loc_outliers = loc_outliers(loc_outliers > 0);
    if ~isempty(loc_outliers)
        fprintf('   [Warning] Manually excluding known outlier (Excel Row: %d)\n', known_outliers);
        bad_idx(loc_outliers) = true; 
    end
end

% Detect severe optical tracking jumps (>10mm inter-node distance)
for i = 1:N_sub
    if bad_idx(i), continue; end
    pts = reshape(P_after_sensor(:, i), 3, 7);
    for j = 2:6
        mid_point = (pts(:, j-1) + pts(:, j+1)) / 2;
        dist_to_mid = norm(pts(:, j) - mid_point);
        if dist_to_mid > 0.01 % 10mm threshold
            bad_idx(i) = true;
            break;
        end
    end
end

% Apply exclusions
F_after  = F_after_sub(:, ~bad_idx);
F_before = F_before_sub(:, ~bad_idx);
P_before_clean = P_before_ideal(:, ~bad_idx);
P_after_clean  = P_after_sensor(:, ~bad_idx);
raw_mag  = raw_mag_sub(~bad_idx);
raw_dir  = raw_dir_sub(~bad_idx);
raw_hgt  = raw_hgt_sub(~bad_idx);
gt_F_clean = gt_F_vec(:, ~bad_idx);
track_rows_clean = track_rows_sub(~bad_idx); 

F_diff = F_after - F_before;
N = length(raw_mag);
fprintf('   > Final effective samples: %d\n', N);

%% ========================================================================
%  Step 2: Data Augmentation
% =========================================================================
disp('--------------------------------------------------');
disp('2. Executing rotational augmentation (x3)...');[aug_F_diff, aug_F_after, aug_F_before, aug_Pb, aug_Pa, aug_gt_F, aug_hgt, aug_track_rows] = ...
    augment_data_by_rotation(F_diff, F_after, F_before, P_before_clean, P_after_clean, gt_F_clean, raw_hgt, track_rows_clean);

% [Residual Core] Calculate the true deformation displacement (Delta P) as Target
aug_Delta_P = aug_Pa - aug_Pb;

%% ========================================================================
%  Step 3: Dataset Construction & Safety Check
% =========================================================================
disp('--------------------------------------------------');
disp('3. Constructing final training set...');

% 3.1 Construct Sets
inputs_f_final   =[aug_F_after; aug_F_diff; aug_F_before]; % For Net B (Force)
targets_f_final  = aug_gt_F;

inputs_loc_final =[aug_F_diff; aug_F_after; aug_Pb]; % For Net B (Loc) / Net C
targets_loc_final = double(aug_hgt) / 9.0; % Normalized Location

% 3.2 Safety Check
bad_total = any(isnan(inputs_f_final), 1) | any(isinf(inputs_f_final), 1) | ...
            any(isnan(inputs_loc_final), 1) | any(isinf(inputs_loc_final), 1);
if sum(bad_total) > 0
    fprintf('   [Warning] Removing %d bad augmented samples.\n', sum(bad_total));
    inputs_f_final(:, bad_total) =[]; targets_f_final(:, bad_total) = [];
    inputs_loc_final(:, bad_total) =[]; targets_loc_final(:, bad_total) = [];
    aug_gt_F(:, bad_total) =[]; aug_Delta_P(:, bad_total) = [];
    aug_track_rows(bad_total) =[]; aug_Pb(:, bad_total) =[];
end

% 3.3 Inject Minimal Noise (Prevent Zero Variance in Z-Score)
epsilon = 1e-7;
inputs_f_final = inputs_f_final + epsilon * randn(size(inputs_f_final));
targets_f_final = targets_f_final + epsilon * randn(size(targets_f_final));
inputs_loc_final = inputs_loc_final + epsilon * randn(size(inputs_loc_final));

fprintf('   > Final augmented input samples: %d\n', size(inputs_f_final, 2));

%% ========================================================================
%  Step 4: Net B - Force Estimation
% =========================================================================
disp('--------------------------------------------------');
disp('4. Training Net B Force...');

net_force = feedforwardnet([40, 20]);
net_force.trainFcn = 'trainlm';
net_force.trainParam.showWindow = false;[net_force, tr_f] = train(net_force, inputs_f_final, targets_f_final);

% Evaluate
pred_f = net_force(inputs_f_final(:, tr_f.testInd));
targ_f = targets_f_final(:, tr_f.testInd);

if any(isnan(pred_f(:))), error('Net B Force produced NaN!'); end
mae_f = mean(abs(sqrt(sum(pred_f.^2)) - sqrt(sum(targ_f.^2))));
fprintf('   > Force MAE: %.4f N\n', mae_f);

%% ========================================================================
%  Step 5: Net B - Location Sensing (Weightedn  Loss)
% =========================================================================
disp('--------------------------------------------------');
disp('5. Training Net B Location (Weighted Loss)...');

% 5.1 Filter High-Force Samples
v_mask = sqrt(sum(aug_gt_F.^2)) > 0.08;
raw_in = inputs_loc_final(:, v_mask);
raw_tg = targets_loc_final(:, v_mask);
raw_shape_tg = aug_Delta_P(:, v_mask); % Corresponding Target for Net C
node_labels = round(raw_tg * 9.0);

% 5.2 Calculate Weights (Inverse Frequency)
nodes_interest = [3, 4, 5];
weights_vec = ones(1, length(node_labels));
fprintf('   > Sample Distribution:\n');
for k = nodes_interest
    idx_k = (node_labels == k);
    count_k = sum(idx_k);
    if count_k > 0
        w_k = length(node_labels) / (length(nodes_interest) * count_k);
        weights_vec(idx_k) = w_k;
        fprintf('     - Node %d: %d samples, Weight: %.2f\n', k, count_k, w_k);
    end
end

% 5.3 Train
[in_norm, ps_in] = mapstd(raw_in); 
[tg_norm, ps_out] = mapstd(raw_tg);

net_loc = fitnet([60, 30]); 
net_loc.trainFcn = 'trainlm'; 
net_loc.trainParam.showWindow = false; 
net_loc.divideParam.testRatio = 0.0; % Manual evaluation later
[net_loc, tr_l] = train(net_loc, in_norm, tg_norm, [],[], weights_vec);

%% ========================================================================
%  Step 6: Evaluation & Visualization (Net B)
% =========================================================================
disp('--------------------------------------------------');
disp('6. Evaluating Net B Location Performance...');

% Predict
pred_val = mapstd('reverse', net_loc(mapstd('apply', raw_in, ps_in)), ps_out);
pred_node = pred_val * 9.0;
real_node = raw_tg * 9.0;

% Clamp & Metrics
pred_node(pred_node < 3) = 3; pred_node(pred_node > 5) = 5;
rmse_node = sqrt(mean((pred_node - real_node).^2));
acc_strict = sum(round(pred_node) == round(real_node)) / length(real_node);

fprintf('   > [Final] RMSE: %.2f Segment\n', rmse_node);
fprintf('   > [Final] Strict Accuracy: %.2f%%\n', acc_strict * 100);

% Plotting
figure('Name', 'Net B: Location Results', 'Color', 'w', 'Position',[100, 100, 1000, 400]);
subplot(1, 2, 1);
jitter = (rand(size(pred_node))-0.5)*0.15;
scatter(real_node, pred_node+jitter, 30, abs(real_node-pred_node), 'filled', 'MarkerFaceAlpha', 0.7);
colormap(jet); caxis([0 1]); colorbar; hold on; plot([2, 6], [2, 6], 'k--');
title(['Regression (RMSE: ', num2str(rmse_node, '%.2f'), ')']);
xlabel('Truth'); ylabel('Pred');

subplot(1, 2, 2);
cm = confusionchart(round(real_node), round(pred_node));
cm.Title = 'Confusion Matrix (Weighted)';
cm.RowSummary = 'row-normalized'; 
sortClasses(cm, 'ascending');

%% ========================================================================
%  Step 7: Net C - Shape Reconstruction (Residual Framework)
% =========================================================================
disp('--------------------------------------------------');
disp('7. Training Net C (Residual Shape Reconstruction)...');

% 7.1 Construct Inputs
feat_internal = aug_F_after(:, v_mask);
feat_external = aug_gt_F(:, v_mask); 
feat_location = double(aug_hgt(:, v_mask)) / 9.0;
feat_P_before = aug_Pb(:, v_mask); % P_before acts exclusively as feature input

inputs_net_c =[feat_internal; feat_external; feat_location; feat_P_before];
targets_net_c = raw_shape_tg; % Target must strictly be Delta P
track_rows_net_c = aug_track_rows(v_mask); % Sync tracking rows

% 7.2 Train Net C
[in_c_norm, ps_in_c] = mapstd(inputs_net_c);
[tg_c_norm, ps_out_c] = mapstd(targets_net_c);

net_shape = fitnet([80, 60, 40]); 
net_shape.trainFcn = 'trainscg'; 
net_shape.trainParam.showWindow = false;
net_shape.trainParam.epochs = 2000;
net_shape.trainParam.goal = 1e-7;
net_shape.trainParam.max_fail = 50;

net_shape.divideParam.trainRatio = 0.8;
net_shape.divideParam.valRatio   = 0.1;
net_shape.divideParam.testRatio  = 0.1;
[net_shape, tr_c] = train(net_shape, in_c_norm, tg_c_norm);

%% ========================================================================
%  Step 8: Net C Evaluation & Worst Case Visualization
% =========================================================================
disp('--------------------------------------------------');
disp('8. Evaluating Net C & Plotting Worst Cases...');

test_idx = tr_c.testInd;
% Safe fallback for test_idx
if isempty(test_idx), test_idx = randperm(size(inputs_net_c, 2), min(50, size(inputs_net_c, 2))); end

in_test = inputs_net_c(:, test_idx);
target_delta_test = targets_net_c(:, test_idx);
p_before_test = feat_P_before(:, test_idx);
rows_test = track_rows_net_c(test_idx);

% Predict residual and reconstruct absolute coordinates
pred_delta = mapstd('reverse', net_shape(mapstd('apply', in_test, ps_in_c)), ps_out_c);
pred_P_after = p_before_test + pred_delta;
real_P_after = p_before_test + target_delta_test;

% Calculate Mean Shape Error
dist_errs = zeros(1, length(test_idx));
tip_dist  = zeros(1, length(test_idx));

for i = 1:length(test_idx)
    p_p = reshape(pred_P_after(:, i), 3,[]);
    p_r = reshape(real_P_after(:, i), 3,[]);
    dist_errs(i) = mean(sqrt(sum((p_p - p_r).^2, 1)));
    tip_dist(i)  = norm(p_p(:, end) - p_r(:, end));
end

mean_dist = mean(dist_errs);
tip_mae = mean(tip_dist);
fprintf('   > [Net C] Mean Shape Error: %.4f m (%.2f mm)\n', mean_dist, mean_dist*1000);
fprintf('   > [Net C] Tip MAE: %.4f m (%.2f mm)\n', tip_mae, tip_mae*1000);

% Find Top Worst Cases
[~, sort_idx] = sort(tip_dist, 'descend');
num_worst = min(7, length(sort_idx));
worst_cases = sort_idx(1:num_worst); 

for k = 1:length(worst_cases)
    idx = worst_cases(k);
    orig_row = rows_test(idx); % Trace back to Excel row
    
    P_p = [[0;0;0], reshape(pred_P_after(:, idx), 3, [])];
    P_r = [[0;0;0], reshape(real_P_after(:, idx), 3,[])];
    
    figure('Name', sprintf('Worst Case %d - Excel Row: %d', k, orig_row), 'Color', 'w', 'Position',[100+k*20, 100+k*20, 600, 500]);
    hold on; grid on; axis equal;
    
    plot3(0,0,0, 'p', 'MarkerSize', 15, 'MarkerEdgeColor', 'k', 'MarkerFaceColor', 'y', 'DisplayName', 'Base Origin');
    plot3(P_r(1,:), P_r(2,:), P_r(3,:), 'k-s', 'LineWidth', 2, 'MarkerFaceColor', 'k', 'DisplayName', 'Ground Truth');
    plot3(P_p(1,:), P_p(2,:), P_p(3,:), 'r--o', 'LineWidth', 2, 'MarkerFaceColor', 'w', 'DisplayName', 'Prediction');
    
    tip_t = P_r(:, end); tip_p = P_p(:, end);
    plot3([tip_t(1), tip_p(1)], [tip_t(2), tip_p(2)],[tip_t(3), tip_p(3)], 'm-', 'LineWidth', 2.5, 'DisplayName', sprintf('Tip Error: %.1fmm', tip_dist(idx)*1000));
    
    % Maintain physical viewing perspective
    set(gca, 'zdir', 'reverse', 'ydir', 'reverse');
    xlabel('X (m)'); ylabel('Y (m)'); zlabel('Z (m)');
    title(sprintf('Worst Case Evaluation | Excel Raw Row: %d', orig_row), 'FontSize', 12, 'Interpreter', 'none');
    legend('Location', 'best');
    view(30, 20);
end

%% ========================================================================
%  Step 9: Tip Error Analysis & Final Save
% =========================================================================
disp('--------------------------------------------------');
disp('9. Analyzing Tip-Specific Error & Saving Models...');

tip_rmse = sqrt(mean(tip_dist.^2));
tip_max = max(tip_dist);

fprintf('   >[Tip] RMSE: %.4f m (%.2f mm)\n', tip_rmse, tip_rmse*1000);
fprintf('   > [Tip] Max:  %.4f m (%.2f mm)\n', tip_max, tip_max*1000);

% Visualization Histogram
figure('Name', 'Tip Error Dist', 'Color', 'w', 'Position',[100, 200, 600, 400]);
histogram(tip_dist * 1000, 30, 'FaceColor', [0.2 0.6 0.3]);
xline(tip_mae * 1000, 'r--', 'LineWidth', 2);
xlabel('Error (mm)'); ylabel('Count'); title('Tip Error Distribution'); grid on;

disp('>>> All done. Models and Parameters successfully processed.');

% Save Model and Parameters
save('Final_System_Checkpoint.mat', ...
     'net_force', 'net_loc', 'net_shape', ...                 
     'ps_in', 'ps_out', 'ps_in_c', 'ps_out_c', ...           
     'inputs_f_final', 'targets_f_final', ...                
     'inputs_loc_final', 'targets_loc_final', 'v_mask', ...  
     'inputs_net_c', 'targets_net_c', ...                    
     'tr_f', 'tr_l', 'tr_c', ...                              
     'test_idx');                                             
disp('Successfully saved as Final_System_Checkpoint.mat');

%% ========================================================================
%  Helper Function: Data Augmentation
% =========================================================================
function[aug_Fd, aug_Fa, aug_Fb, aug_Pb, aug_Pa, aug_gF, aug_h, aug_tr] = ...
    augment_data_by_rotation(F_diff, F_after, F_before, P_before, P_after, gt_F, hgt, track_rows)
    
    N = size(F_diff, 2);
    R120 =[cosd(120), -sind(120), 0; sind(120), cosd(120), 0; 0, 0, 1];
    R240 =[cosd(240), -sind(240), 0; sind(240), cosd(240), 0; 0, 0, 1];
    
    % Tendon index permutation
    idx120 = [5, 6, 1, 2, 3, 4]; 
    idx240 = [3, 4, 5, 6, 1, 2];
    
    % Transformation helper function
    rotP = @(P, R) reshape(R * reshape(P, 3,[]), 21, N);
    
    % 120-degree augmentation
    Fd_120 = F_diff(idx120, :); Fa_120 = F_after(idx120, :); Fb_120 = F_before(idx120, :);
    gF_120 = R120 * gt_F;
    P_b_120 = rotP(P_before, R120);
    P_a_120 = rotP(P_after, R120);
    
    % 240-degree augmentation
    Fd_240 = F_diff(idx240, :); Fa_240 = F_after(idx240, :); Fb_240 = F_before(idx240, :);
    gF_240 = R240 * gt_F;
    P_b_240 = rotP(P_before, R240);
    P_a_240 = rotP(P_after, R240);
    
    % Concatenate original and augmented data
    aug_Fd = [F_diff, Fd_120, Fd_240]; 
    aug_Fa = [F_after, Fa_120, Fa_240];
    aug_Fb = [F_before, Fb_120, Fb_240]; 
    aug_Pb = [P_before, P_b_120, P_b_240];
    aug_Pa = [P_after, P_a_120, P_a_240]; 
    aug_gF = [gt_F, gF_120, gF_240]; 
    aug_h  = [hgt, hgt, hgt];
    aug_tr =[track_rows; track_rows; track_rows]; 
end
