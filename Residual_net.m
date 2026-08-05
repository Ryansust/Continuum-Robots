%% ========================================================================
%  Project: Deep Learning-Based Robust Proprioceptive Pose Reconstruction
%  Author:  Lin Yongxi (Student ID: 12313007)
%  Date:    2026-01-28
%  Description: 
%     This script implements a physics-aware residual framework for 
%     continuum robot shape sensing. It consists of three stages:
%     1. Data Preprocessing & Augmentation (4D Orthogonal Strategy)
%     2. Net B: Interaction Sensing (Force Regression + Location Classification)
%     3. Net C: Shape Reconstruction (Feature Fusion)
% =========================================================================

clc; clear; close all;
rng('default'); % Ensure reproducibility

%% ========================================================================
%  Step 1: Data Loading & Preprocessing
% =========================================================================
disp('[Step 1] Loading and preprocessing raw data...');

% 1.1 Load Dataset
DataPath = '/Users/ryan/Desktop/continuum robot/force_data/after_processing_data_0816.xlsx';
if ~isfile(DataPath)
    error('Error: Dataset file not found. Please check the path.');
end
dataTable = readtable(DataPath);

% 1.2 Extract Raw Signals
% Tendon Force (Post-disturbance)
F_after_raw  = double(table2array(dataTable(3:end, 23:28)))';  
% Tendon Force (Pre-disturbance / Baseline)
F_before_raw = double(table2array(dataTable(3:end, 11:16)))';  
% Ground Truth Labels
raw_mag_raw  = double(abs(table2array(dataTable(3:end, 2))))'; 
raw_dir_raw  = double(table2array(dataTable(3:end, 3)))';      
raw_hgt_raw  = double(table2array(dataTable(3:end, 4)))'; 
pos_text_raw = dataTable{3:end, 38}; % Raw strings for Nokov 3D positions

% 1.3 ROI Filtering (Region of Interest: Nodes 3, 4, 5)
% Filter out base and tip nodes to focus on the high-sensitivity middle section
target_nodes = [3, 4, 5];
roi_mask = ismember(raw_hgt_raw, target_nodes);

F_after_raw  = F_after_raw(:, roi_mask);
F_before_raw = F_before_raw(:, roi_mask);
raw_mag_raw  = raw_mag_raw(roi_mask);
raw_dir_raw  = raw_dir_raw(roi_mask);
raw_hgt_raw  = raw_hgt_raw(roi_mask);
pos_text_raw = pos_text_raw(roi_mask);

% 1.4 Data Cleaning (Remove NaN/Inf)
bad_idx = any(isnan(F_after_raw), 1) | any(isnan(F_before_raw), 1) | ...
          isnan(raw_mag_raw) | isnan(raw_dir_raw) | isnan(raw_hgt_raw);
          
F_after  = F_after_raw(:, ~bad_idx);
F_before = F_before_raw(:, ~bad_idx);
raw_mag  = raw_mag_raw(~bad_idx);
raw_dir  = raw_dir_raw(~bad_idx);
raw_hgt  = raw_hgt_raw(~bad_idx);
pos_text = pos_text_raw(~bad_idx); 

F_diff = F_after - F_before; % Differential Tension Features
N = length(raw_mag);

fprintf('   > Effective Samples (ROI & Cleaned): %d\n', N);

%% ========================================================================
%  Step 2: Ground Truth Generation & Kinematics Parsing
% =========================================================================
disp('[Step 2] Parsing 3D Kinematics from Nokov data...');

% 2.1 Parse 3D Coordinates (P_shape)
% Output: 21-dim vector (7 markers * 3 coords)
P_shape = zeros(21, N); 
for i = 1:N
    % External function to parse string format from Nokov
    real_offset = get_RealOffset_1S3CT(pos_text{i});
    body_markers = real_offset(:, 3:end); % Extract backbone markers
    P_shape(:, i) = reshape(body_markers, [], 1); 
end

% 2.2 Construct External Force Vectors (F_ext)
gt_F_vec = zeros(3, N);
for i = 1:N
    u_vec = [0;0;0];
    switch raw_dir(i)
        case 2, u_vec = [-1; 0; 0];               % Normal
        case 3, u_vec = [-sind(45); cosd(45); 0]; % Oblique
        case 4, u_vec = [0; 1; 0];                % Tangential
    end
    gt_F_vec(:, i) = raw_mag(i) * u_vec;
end

%% ========================================================================
%  Step 3: Data Augmentation (Rotational Symmetry)
% =========================================================================
disp('[Step 3] Applying Data Augmentation...');

% Apply 120-degree rotational symmetry to expand dataset size (x3)
[aug_F_diff, aug_F_after, aug_F_before, aug_P_shape, aug_gt_F, aug_hgt] = ...
    augment_data_by_rotation(F_diff, F_after, F_before, P_shape, gt_F_vec, raw_hgt);

% Final Dataset Construction
inputs_f_final   = [aug_F_after; aug_F_diff; aug_F_before]; % For Net B (Force)
targets_f_final  = aug_gt_F;

inputs_loc_final = [aug_F_diff; aug_F_after; aug_P_shape];  % For Net B (Loc) & Net C
targets_loc_final = double(aug_hgt) / 9.0; % Normalized location (Node Index)

% Add minimal noise for numerical stability in Z-Score normalization
epsilon = 1e-7;
inputs_f_final = inputs_f_final + epsilon * randn(size(inputs_f_final));
targets_f_final = targets_f_final + epsilon * randn(size(targets_f_final));
inputs_loc_final = inputs_loc_final + epsilon * randn(size(inputs_loc_final));

fprintf('   > Augmented Dataset Size: %d samples\n', size(inputs_f_final, 2));

%% ========================================================================
%  Step 4: Net B - Force Estimation (Regression)
% =========================================================================
disp('[Step 4] Training Net B (Force Regression)...');

net_force = feedforwardnet([40, 20]);
net_force.trainFcn = 'trainlm';
net_force.trainParam.showWindow = false;

[net_force, tr_f] = train(net_force, inputs_f_final, targets_f_final);

% Validation
pred_f_test = net_force(inputs_f_final(:, tr_f.testInd));
targ_f_test = targets_f_final(:, tr_f.testInd);
mae_f = mean(abs(sqrt(sum(pred_f_test.^2)) - sqrt(sum(targ_f_test.^2))));

% if any(isnan(pred_f_test(:)))
%     warning('Net_B_Force 输出了 NaN。尝试降低学习率或更换算法。');
%     mae_f = NaN;
% else
%     mae_f = mean(abs(sqrt(sum(pred_f_test.^2)) - sqrt(sum(targ_f_test.^2))));
%     fprintf('   > Force MAE: %.4f N\n', mae_f);
% end

fprintf('   > Force Estimation MAE: %.4f N\n', mae_f);


    if k==1, legend('Truth', 'Reconstruction'); end
    view(30, 20); 
end


%% === 9. Tip Error Analysis (Restored from Fix) ===
disp('--------------------------------------------------');
disp('9. Calculating Tip Independent Error...');

% Extract Tip Data (Indices 19, 20, 21)
tip_indices = [19, 20, 21];

tip_pred = pred_test(tip_indices, :);   
tip_real = target_test(tip_indices, :); 

% Calculate Euclidean Distance
tip_err_vec = tip_pred - tip_real;
tip_err_dist = sqrt(sum(tip_err_vec.^2, 1)); 

% Metrics
tip_mae = mean(tip_err_dist);
tip_rmse = sqrt(mean(tip_err_dist.^2));
tip_max = max(tip_err_dist);

fprintf('   > [Tip Specific] MAE:  %.4f m (%.2f mm)\n', tip_mae, tip_mae*1000);
fprintf('   > [Tip Specific] RMSE: %.4f m (%.2f mm)\n', tip_rmse, tip_rmse*1000);
fprintf('   > [Tip Specific] Max:  %.4f m (%.2f mm)\n', tip_max, tip_max*1000);

% Visualization: Tip Tracking
figure('Name', 'Tip Tracking Performance', 'Color', 'w', 'Position', [100, 200, 1000, 400]);

% Subplot 1: 3D Tracking
subplot(1, 2, 1);
num_show = min(50, length(tip_err_dist));
idx_show = randperm(length(tip_err_dist), num_show);

hold on; grid on; axis equal;
h1 = plot3(NaN,NaN,NaN, 'bo', 'MarkerSize', 5, 'MarkerFaceColor', 'b'); 
h2 = plot3(NaN,NaN,NaN, 'r.', 'MarkerSize', 10); 

for k = idx_show
    p_r = tip_real(:, k);
    p_p = tip_pred(:, k);
    plot3([p_r(1), p_p(1)], [p_r(2), p_p(2)], [p_r(3), p_p(3)], 'Color', [0.7 0.7 0.7], 'LineWidth', 1);
    plot3(p_r(1), p_r(2), p_r(3), 'bo', 'MarkerSize', 5, 'MarkerFaceColor', 'b'); 
    plot3(p_p(1), p_p(2), p_p(3), 'r.', 'MarkerSize', 10); 
end
xlabel('X (m)'); ylabel('Y (m)'); zlabel('Z (m)');
title('Tip Position Tracking (Blue=True, Red=Pred)');
legend([h1, h2], {'Ground Truth', 'Prediction'}, 'Location', 'best');
view(45, 30);

% Subplot 2: Histogram
subplot(1, 2, 2);
histogram(tip_err_dist * 1000, 30, 'FaceColor', [0.2 0.6 0.3]);
xline(tip_mae * 1000, 'r--', 'LineWidth', 2, 'Label', sprintf('Mean: %.2f mm', tip_mae*1000));
xlabel('Tip Position Error (mm)');
ylabel('Sample Count');
title('Tip Error Distribution');
grid on;

disp('>>> All processes completed.');


%% === Rotation Helper Function ===
function [aug_Fd, aug_Fa, aug_Fb, aug_Pb, aug_gF, aug_h] = ...
    augment_data_by_rotation(F_diff, F_after, F_before, P_before, gt_F, hgt)
    
    N = size(F_diff, 2);
    R120 = [cosd(120), -sind(120), 0; sind(120), cosd(120), 0; 0, 0, 1];
    R240 = [cosd(240), -sind(240), 0; sind(240), cosd(240), 0; 0, 0, 1];
    idx120 = [5, 6, 1, 2, 3, 4];
    idx240 = [3, 4, 5, 6, 1, 2];
    
    Fd_120 = F_diff(idx120, :); Fa_120 = F_after(idx120, :); Fb_120 = F_before(idx120, :);
    gF_120 = R120 * gt_F;
    P_tmp = reshape(P_before, 3, []);
    P_120 = reshape(R120 * P_tmp, 21, N);
    
    Fd_240 = F_diff(idx240, :); Fa_240 = F_after(idx240, :); Fb_240 = F_before(idx240, :);
    gF_240 = R240 * gt_F;
    P_240 = reshape(R240 * P_tmp, 21, N);
    
    aug_Fd = [F_diff, Fd_120, Fd_240];
    aug_Fa = [F_after, Fa_120, Fa_240];
    aug_Fb = [F_before, Fb_120, Fb_240];
    aug_Pb = [P_before, P_120, P_240];
    aug_gF = [gt_F, gF_120, gF_240];
    aug_h  = [hgt, hgt, hgt];
end