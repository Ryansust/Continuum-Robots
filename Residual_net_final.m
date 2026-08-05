%% ========================================================================
%  Project: Deep Learning-Based Robust Proprioceptive Pose Reconstruction
%  Author:  Lin Yongxi
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

% 1.2 Extract Signals
F_after_raw  = double(table2array(dataTable(3:end, 23:28)))';  
F_before_raw = double(table2array(dataTable(3:end, 11:16)))';  
raw_mag_raw  = double(abs(table2array(dataTable(3:end, 2))))'; 
raw_dir_raw  = double(table2array(dataTable(3:end, 3)))';      
raw_hgt_raw  = double(table2array(dataTable(3:end, 4)))'; 
pos_text_raw = dataTable{3:end, 29}; 

% 1.3 ROI Filtering (Keep Nodes 3, 4, 5)
disp('   > Executing ROI filtering (Nodes 3, 4, 5)...');
roi_mask = ismember(raw_hgt_raw, [3, 4, 5]);

% Apply Filter
F_after_sub  = F_after_raw(:, roi_mask);
F_before_sub = F_before_raw(:, roi_mask);
raw_mag_sub  = raw_mag_raw(roi_mask);
raw_dir_sub  = raw_dir_raw(roi_mask);
raw_hgt_sub  = raw_hgt_raw(roi_mask);
pos_text_sub = pos_text_raw(roi_mask);

if length(raw_mag_sub) < 50, error('Insufficient data after ROI filtering.'); end

% 1.4 Data Cleaning (Remove NaN/Inf)
disp('   > Removing invalid samples...');

bad_idx = any(isnan(F_after_sub), 1) | any(isnan(F_before_sub), 1) | ...
          isnan(raw_mag_sub) | isnan(raw_dir_sub) | isnan(raw_hgt_sub);
known_outliers = [686]; 

if ~isempty(known_outliers)
    fprintf('   ⚠手动剔除已知异常样本 ID: %s\n', num2str(known_outliers));
    bad_idx(known_outliers) = true; % 强制标记为坏样本
end
F_after  = F_after_sub(:, ~bad_idx);
F_before = F_before_sub(:, ~bad_idx);
raw_mag  = raw_mag_sub(~bad_idx);
raw_dir  = raw_dir_sub(~bad_idx);
raw_hgt  = raw_hgt_sub(~bad_idx);
pos_text = pos_text_sub(~bad_idx); 

F_diff = F_after - F_before;
N = length(raw_mag);
fprintf('   > Final effective samples: %d\n', N);

% 1.5 Kinematics Parsing & Ground Truth Generation
disp('   > Parsing pose data and generating ground truth...');
P_before = zeros(21, N); 
gt_F_vec = zeros(3, N);

for i = 1:N
    % Parse 3D coords
    real_offset = get_RealOffset_1S3CT(pos_text{i});
    P_before(:, i) = reshape(real_offset(:, 3:end), [], 1); 
    
    % Calc Force Vector
    u_vec = [0;0;0];
    switch raw_dir(i)
        case 2, u_vec = [-1; 0; 0];
        case 3, u_vec = [-sind(45); cosd(45); 0];
        case 4, u_vec = [0; 1; 0];
    end
    gt_F_vec(:, i) = raw_mag(i) * u_vec;
end

%% ========================================================================
%  Step 2: Data Augmentation
% =========================================================================
disp('--------------------------------------------------');
disp('2. Executing rotational augmentation...');
[aug_F_diff, aug_F_after, aug_F_before, aug_P_before, aug_gt_F, aug_hgt] = ...
    augment_data_by_rotation(F_diff, F_after, F_before, P_before, gt_F_vec, raw_hgt);

%% ========================================================================
%  Step 3: Dataset Construction & Safety Check
% =========================================================================
disp('--------------------------------------------------');
disp('3. Constructing final training set...');

% 3.1 Construct Sets
inputs_f_final   = [aug_F_after; aug_F_diff; aug_F_before]; % For Net B Force
targets_f_final  = aug_gt_F;

inputs_loc_final = [aug_F_diff; aug_F_after; aug_P_before]; % For Net B Loc / Net C
targets_loc_final = double(aug_hgt) / 9.0; % Normalized Location

% 3.2 Safety Check
bad_total = any(isnan(inputs_f_final), 1) | any(isinf(inputs_f_final), 1) | ...
            any(isnan(inputs_loc_final), 1) | any(isinf(inputs_loc_final), 1);
if sum(bad_total) > 0
    fprintf('   [Warning] Removing %d bad augmented samples.\n', sum(bad_total));
    inputs_f_final(:, bad_total) = []; targets_f_final(:, bad_total) = [];
    inputs_loc_final(:, bad_total) = []; targets_loc_final(:, bad_total) = [];
    aug_gt_F(:, bad_total) = [];
end

% 3.3 Inject Minimal Noise (Prevent Zero Variance in Z-Score)
epsilon = 1e-7;
inputs_f_final = inputs_f_final + epsilon * randn(size(inputs_f_final));
targets_f_final = targets_f_final + epsilon * randn(size(targets_f_final));
inputs_loc_final = inputs_loc_final + epsilon * randn(size(inputs_loc_final));

fprintf('   > Final input samples: %d\n', size(inputs_f_final, 2));

%% ========================================================================
%  Step 4: Net B - Force Estimation
% =========================================================================
disp('--------------------------------------------------');
disp('4. Training Net B Force...');

net_force = feedforwardnet([40, 20]);
net_force.trainFcn = 'trainlm';
net_force.trainParam.showWindow = false;

[net_force, tr_f] = train(net_force, inputs_f_final, targets_f_final);

% Evaluate
pred_f = net_force(inputs_f_final(:, tr_f.testInd));
targ_f = targets_f_final(:, tr_f.testInd);

if any(isnan(pred_f(:))), error('Net B Force produced NaN!'); end
mae_f = mean(abs(sqrt(sum(pred_f.^2)) - sqrt(sum(targ_f.^2))));
fprintf('   > Force MAE: %.4f N\n', mae_f);

%% ========================================================================
%  Step 5: Net B - Location Sensing (Weighted Loss)
% =========================================================================
disp('--------------------------------------------------');
disp('5. Training Net B Location (Weighted Loss)...');

% 5.1 Filter High-Force Samples
v_mask = sqrt(sum(aug_gt_F.^2)) > 0.08;
raw_in = inputs_loc_final(:, v_mask);
raw_tg = targets_loc_final(:, v_mask);
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

net_loc = fitnet([60, 40, 20]);
net_loc.trainFcn = 'trainlm'; 
net_loc.trainParam.showWindow = true; 
net_loc.trainParam.epochs = 1500;
net_loc.trainParam.max_fail = 20; 
net_loc.trainParam.goal = 1e-6;


net_loc.divideParam.trainRatio = 0.80;
net_loc.divideParam.valRatio   = 0.20;
net_loc.divideParam.testRatio  = 0.0;  % Manual eval later

[net_loc, tr_l] = train(net_loc, in_norm, tg_norm, [], [], weights_vec);

%% ========================================================================
%  Step 6: Evaluation & Visualization (Net B)
% =========================================================================
disp('--------------------------------------------------');
disp('6. Evaluating Net B performance...');

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
figure('Name', 'Net B: Location Results', 'Color', 'w', 'Position', [100, 100, 1000, 400]);
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
%  Step 7: Net C - Shape Reconstruction
% =========================================================================
disp('--------------------------------------------------');
disp('7. Training Net C (Shape Reconstruction)...');

% 7.1 Construct Inputs
data_pool = inputs_loc_final(:, v_mask);
feat_internal = data_pool(7:12, :);      % F_after
feat_external = targets_f_final(:, v_mask); % F_ext
feat_location = targets_loc_final(:, v_mask); % Location

inputs_net_c = [feat_internal; feat_external; feat_location];
targets_net_c = data_pool(13:33, :); % P_before (Shape GT)

% 7.2 Train (Using trainscg for speed)
[in_c_norm, ps_in_c] = mapstd(inputs_net_c);
[tg_c_norm, ps_out_c] = mapstd(targets_net_c);

net_shape = fitnet([80, 60, 40]); 
net_shape.trainFcn = 'trainscg'; % Fast algo
net_shape.trainParam.showWindow = true;
net_shape.trainParam.epochs = 2000;
net_shape.trainParam.goal = 1e-7;
net_shape.trainParam.max_fail = 50;

% Split 80/10/10
net_shape.divideParam.trainRatio = 0.8;
net_shape.divideParam.valRatio   = 0.1;
net_shape.divideParam.testRatio  = 0.1;

[net_shape, tr_c] = train(net_shape, in_c_norm, tg_c_norm);

%% ========================================================================
%  Step 8: Net C Evaluation
% =========================================================================
disp('--------------------------------------------------');
disp('8. Evaluating Net C Performance...');

test_idx = tr_c.testInd;
if isempty(test_idx), test_idx = randperm(size(inputs_net_c,2), 50); end

in_test = inputs_net_c(:, test_idx);
target_test = targets_net_c(:, test_idx);

pred_test = mapstd('reverse', net_shape(mapstd('apply', in_test, ps_in_c)), ps_out_c);

% Mean Shape Error
dist_errs = zeros(1, length(test_idx));
for i = 1:length(test_idx)
    p_p = reshape(pred_test(:, i), 3, []);
    p_r = reshape(target_test(:, i), 3, []);
    dist_errs(i) = mean(sqrt(sum((p_p - p_r).^2, 1)));
end
mean_dist = mean(dist_errs);
fprintf('   > [Net C] Mean Shape Error: %.4f m (%.2f mm)\n', mean_dist, mean_dist*1000);

% 3D Visualization
figure('Name', '3D Shape Reconstruction', 'Color', 'w', 'Position', [100, 100, 1200, 500]);
num_plot = 4;
plot_ids = test_idx(randperm(length(test_idx), num_plot));

for k = 1:num_plot
    idx = plot_ids(k);
    P_p = [[0;0;0], reshape(pred_test(:, find(test_idx==idx,1)), 3, [])];
    P_r = [[0;0;0], reshape(target_test(:, find(test_idx==idx,1)), 3, [])];
    
    subplot(1, num_plot, k);
    plot3(P_r(1,:), P_r(2,:), P_r(3,:), 'k-o', 'LineWidth', 2); hold on;
    plot3(P_p(1,:), P_p(2,:), P_p(3,:), 'r--.', 'LineWidth', 1.5);
    grid on; axis equal; xlabel('X'); zlabel('Z'); 
    title(['Sample ', num2str(k)]);
    if k==1, legend('Truth', 'Pred'); end
    view(30, 20);
end

%% ========================================================================
%  Step 9: Tip Error Analysis (Corrected Variable Names)
% =========================================================================
disp('--------------------------------------------------');
disp('9. Analyzing Tip-Specific Error...');

tip_idx = [19, 20, 21];
tip_pred = pred_test(tip_idx, :);
tip_real = target_test(tip_idx, :);

tip_vec = tip_pred - tip_real;
tip_dist = sqrt(sum(tip_vec.^2, 1)); 

tip_mae = mean(tip_dist);
tip_rmse = sqrt(mean(tip_dist.^2));
tip_max = max(tip_dist);

fprintf('   > [Tip] MAE:  %.4f m (%.2f mm)\n', tip_mae, tip_mae*1000);
fprintf('   > [Tip] RMSE: %.4f m (%.2f mm)\n', tip_rmse, tip_rmse*1000);
fprintf('   > [Tip] Max:  %.4f m (%.2f mm)\n', tip_max, tip_max*1000);

% Visualization
figure('Name', 'Tip Error Analysis', 'Color', 'w', 'Position', [100, 200, 1000, 400]);

subplot(1, 2, 1);
num_show = min(50, length(tip_dist));
idx_show = randperm(length(tip_dist), num_show);
hold on; grid on; axis equal;
h1 = plot3(NaN,NaN,NaN, 'bo'); h2 = plot3(NaN,NaN,NaN, 'r.');

for k = idx_show
    p_r = tip_real(:, k); p_p = tip_pred(:, k);
    plot3([p_r(1), p_p(1)], [p_r(2), p_p(2)], [p_r(3), p_p(3)], 'Color', [0.7 0.7 0.7]);
    plot3(p_r(1), p_r(2), p_r(3), 'bo', 'MarkerSize', 5, 'MarkerFaceColor', 'b');
    plot3(p_p(1), p_p(2), p_p(3), 'r.', 'MarkerSize', 10);
end
xlabel('X'); ylabel('Y'); zlabel('Z'); title('Tip Tracking'); legend([h1, h2], {'GT', 'Pred'}); view(45, 30);

subplot(1, 2, 2);
histogram(tip_dist * 1000, 30, 'FaceColor', [0.2 0.6 0.3]);
xline(tip_mae * 1000, 'r--', 'LineWidth', 2);
xlabel('Error (mm)'); ylabel('Count'); title('Tip Error Dist.'); grid on;

disp('>>> All done.');
%% === Save Model and Parameters ===
disp('9. Saving models and test indices for independent evaluation...');
% 保存所有必要的信息：
% 1. 网络模型 (net_force, net_loc, net_shape)
% 2. 归一化参数 (ps_in, ps_out 等)
% 3. 数据集全集 (inputs_*, targets_*)
% 4. 关键的测试集索引 (test_idx) -> 这就是“留出法”的证据
% 5. 辅助变量 (v_mask, target_nodes 等)
save('Final_System_Checkpoint.mat', ...
     'net_force', 'net_loc', 'net_shape', ...                 % Networks
     'ps_in', 'ps_out', 'ps_in_c', 'ps_out_c', ...           % Norm Params
     'inputs_f_final', 'targets_f_final', ...                % Force Data
     'inputs_loc_final', 'targets_loc_final', 'v_mask', ...  % Location Data
     'inputs_net_c', 'targets_net_c', ...                    % Shape Data
     'tr_f', 'tr_l', 'tr_c', ...                              % Training Records (contain indices)
     'test_idx');                                             % Net C Test Indices
disp('Successfully saved as Final_Models.mat');
%% ========================================================================
%  Helper Function: Data Augmentation
% =========================================================================
function [aug_Fd, aug_Fa, aug_Fb, aug_Pb, aug_gF, aug_h] = ...
    augment_data_by_rotation(F_diff, F_after, F_before, P_before, gt_F, hgt)
    
    N = size(F_diff, 2);
    R120 = [cosd(120), -sind(120), 0; sind(120), cosd(120), 0; 0, 0, 1];
    R240 = [cosd(240), -sind(240), 0; sind(240), cosd(240), 0; 0, 0, 1];
    idx120 = [5, 6, 1, 2, 3, 4]; idx240 = [3, 4, 5, 6, 1, 2];
    
    Fd_120 = F_diff(idx120, :); Fa_120 = F_after(idx120, :); Fb_120 = F_before(idx120, :);
    gF_120 = R120 * gt_F;
    P_tmp = reshape(P_before, 3, []); P_120 = reshape(R120 * P_tmp, 21, N);
    
    Fd_240 = F_diff(idx240, :); Fa_240 = F_after(idx240, :); Fb_240 = F_before(idx240, :);
    gF_240 = R240 * gt_F;
    P_240 = reshape(R240 * P_tmp, 21, N);
    
    aug_Fd = [F_diff, Fd_120, Fd_240]; aug_Fa = [F_after, Fa_120, Fa_240];
    aug_Fb = [F_before, Fb_120, Fb_240]; aug_Pb = [P_before, P_120, P_240];
    aug_gF = [gt_F, gF_120, gF_240]; aug_h  = [hgt, hgt, hgt];
end