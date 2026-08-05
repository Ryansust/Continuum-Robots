%% ========================================================================
%  Project: 基于物理特征增强的连续体机器人本体感知系统 (Net_B + Net_C)
%  Status:  逻辑严密修正版 + 绘图对齐 + 多点剔除支持
%  Author:  Lin Yongxi
% =========================================================================
clc; clear; close all;
rng('default'); 

%% ========================================================================
%  Step 1: 数据读取、ROI 筛选与清洗
% =========================================================================
disp('--------------------------------------------------');
disp('1. 正在读取原始数据并进行物理预处理...');

% 1.1 读取 Excel
FileName = '/Users/ryan/Desktop/continuum robot/force_data/after_processing_data_0816.xlsx';
if ~isfile(FileName), error('未找到 Excel 文件，请检查路径！'); end
dataTable = readtable(FileName);

% 1.2 提取信号
F_after_raw  = double(table2array(dataTable(3:end, 23:28)))'; 
F_before_raw = double(table2array(dataTable(3:end, 11:16)))'; 
raw_mag_raw  = double(abs(table2array(dataTable(3:end, 2))))'; 
raw_dir_raw  = double(table2array(dataTable(3:end, 3)))';     
raw_hgt_raw  = double(table2array(dataTable(3:end, 4)))';     
pos_text_raw = dataTable{3:end, 29}; % 碰撞发生后的位姿文本

% 1.3 ROI 筛选
disp('   > 执行 ROI 筛选 (Nodes 3, 4, 5)...');
roi_mask = ismember(raw_hgt_raw, [3, 4, 5]);

% 记录筛选后的行号追踪 (用于后续精准剔除和溯源)
track_rows = (3 : (length(raw_hgt_raw) + 2));
track_rows = track_rows(roi_mask);

F_after_sub  = F_after_raw(:, roi_mask);
F_before_sub = F_before_raw(:, roi_mask);
raw_mag_sub  = raw_mag_raw(roi_mask);
raw_dir_sub  = raw_dir_raw(roi_mask);
raw_hgt_sub  = raw_hgt_raw(roi_mask);
pos_text_sub = pos_text_raw(roi_mask);

% 1.4 数据清洗
disp('   > 正在剔除无效样本...');
bad_idx = any(isnan(F_after_sub), 1) | any(isnan(F_before_sub), 1) | ...
          isnan(raw_mag_sub) | isnan(raw_dir_sub) | isnan(raw_hgt_sub);
known_outliers = [686]; 
if ~isempty(known_outliers), bad_idx(known_outliers) = true; end
% 🌟 【根据 Excel 行号剔除多个异常点】
known_outliers_excel = [206,207,224,608,609,610,611,612,613,614,615,616,617,618,619,620,621]; % <--- 在这里直接填入你要删的 Excel 行号
bad_idx(ismember(track_rows, known_outliers_excel)) = true;

F_after  = F_after_sub(:, ~bad_idx);
F_before = F_before_sub(:, ~bad_idx);
raw_mag  = raw_mag_sub(~bad_idx);
raw_dir  = raw_dir_sub(~bad_idx);
raw_hgt  = raw_hgt_sub(~bad_idx);
pos_text = pos_text_sub(~bad_idx); 
track_rows_final = track_rows(~bad_idx);

N = length(raw_mag);
F_diff = F_after - F_before;

% 1.5 物理模型生成理想起始位姿 (Feature)
disp('   > 正在生成物理理想 $P_{before}$ (Feature) 与解析碰撞后真值 (Target)...');
tendon_p = 3; section_p = 2; D_p = 0.0006; E_p = 0.516e+12; L_ap = 0.0665; N_dp = 7;
H_listp = linspace(0.0025, 0.0025, 15); mu_p = 0.25; delta_alphap = 0; G_loadp = 4.0 * 0.00981;

P_before_ideal = zeros(21, N); 
P_after_actual = zeros(21, N); 
gt_F_vec = zeros(3, N);

for i = 1:N
    Fb_raw = F_before(:, i) * 0.00981; 
    Fb_sim = [Fb_raw(5); Fb_raw(6); Fb_raw(1); Fb_raw(2); Fb_raw(3); Fb_raw(4)];
    [P_Theo, ~, R_mat, ~, ~, ~] = solve_continuum_shape_nofig(tendon_p, section_p, D_p, E_p, L_ap, 0, N_dp, H_listp, mu_p, delta_alphap, G_loadp, [0;0;0], Fb_sim, 14);
    
    V_local = [0; -0.004; 0]; 
    P_m = zeros(3, size(P_Theo, 2));
    for pt = 1:size(P_Theo, 2), P_m(:, pt) = P_Theo(:, pt) + R_mat(:, :, pt) * V_local; end
    m_idx = round([2,4,6,8,10,12,14] * ((size(P_Theo,2)-1)/14)) + 1;
    P_before_ideal(:, i) = reshape(P_m(:, m_idx), 21, 1); 
    
    real_offset_after = get_RealOffset_1S3CT(pos_text{i});
    P_after_actual(:, i) = reshape(real_offset_after(:, 3:end), 21, 1); 
    
    u_vec = [0;0;0];
    switch raw_dir(i)
        case 2, u_vec = [-1; 0; 0]; case 3, u_vec = [-sind(45); cosd(45); 0]; case 4, u_vec = [0; 1; 0];
    end
    gt_F_vec(:, i) = raw_mag(i) * u_vec;
end

%% ========================================================================
%  Step 2: 数据增强 (旋转同步)
% =========================================================================
disp('--------------------------------------------------');
disp('2. 正在执行同步旋转增强...');
[aug_F_diff, aug_F_after, aug_F_before, aug_Pb_ideal, aug_Pa_actual, aug_gt_F, aug_hgt] = ...
    augment_data_consistent(F_diff, F_after, F_before, P_before_ideal, P_after_actual, gt_F_vec, raw_hgt);

%% ========================================================================
%  Step 3: 训练集构建与噪声注入
% =========================================================================
inputs_f_final   = [aug_F_after; aug_F_diff; aug_F_before]; 
targets_f_final  = aug_gt_F;
inputs_loc_final = [aug_F_diff; aug_F_after; aug_Pb_ideal]; 
targets_loc_final = double(aug_hgt) / 9.0; 
targets_shape_final = aug_Pa_actual; 

bad_total = any(isnan(inputs_f_final), 1) | any(isinf(inputs_f_final), 1) | ...
            any(isnan(inputs_loc_final), 1) | any(isnan(targets_shape_final), 1);
inputs_f_final(:, bad_total) = []; targets_f_final(:, bad_total) = [];
inputs_loc_final(:, bad_total) = []; targets_loc_final(:, bad_total) = [];
aug_gt_F(:, bad_total) = []; targets_shape_final(:, bad_total) = [];

epsilon = 1e-7;
inputs_f_final = inputs_f_final + epsilon * randn(size(inputs_f_final));
inputs_loc_final = inputs_loc_final + epsilon * randn(size(inputs_loc_final));

%% ========================================================================
%  Step 4, 5, 6: 训练 Net B (Force & Location)
% =========================================================================
disp('--------------------------------------------------');
disp('4 & 5. Training Net B...');
net_force = feedforwardnet([40, 20]); net_force.trainFcn = 'trainlm'; net_force.trainParam.showWindow = false;
[net_force, tr_f] = train(net_force, inputs_f_final, targets_f_final);

v_mask = sqrt(sum(aug_gt_F.^2)) > 0.08;
raw_in_l = inputs_loc_final(:, v_mask); raw_tg_l = targets_loc_final(:, v_mask);
raw_tg_s = targets_shape_final(:, v_mask); node_labels = round(raw_tg_l * 9.0);

nodes_i = [3, 4, 5]; w_vec = ones(1, length(node_labels));
for k = nodes_i
    idx_k = (node_labels == k); c_k = sum(idx_k);
    if c_k > 0, w_vec(idx_k) = length(node_labels)/(3*c_k); end
end
[in_l_norm, ps_in] = mapstd(raw_in_l); [tg_l_norm, ps_out] = mapstd(raw_tg_l);
net_loc = fitnet([60, 40, 20]); net_loc.trainFcn = 'trainlm';
[net_loc, tr_l] = train(net_loc, in_l_norm, tg_l_norm, [], [], w_vec);

%% ========================================================================
%  Step 7: 训练优化版 Net C (残差回归框架)
% =========================================================================
disp('--------------------------------------------------');
disp('7. 正在训练 Net C (Residual Shaping)...');
feat_Pb_ideal = raw_in_l(13:33, :);
inputs_net_c = [raw_in_l(7:12, :); targets_f_final(:, v_mask); raw_tg_l; feat_Pb_ideal];
targets_net_c = raw_tg_s - feat_Pb_ideal; % 目标是残差 Delta_P

[in_c_norm, ps_in_c] = mapstd(inputs_net_c); [tg_c_norm, ps_out_c] = mapstd(targets_net_c);
net_shape = fitnet([100, 80, 40]); net_shape.trainFcn = 'trainscg'; net_shape.trainParam.epochs = 2000;
[net_shape, tr_c] = train(net_shape, in_c_norm, tg_c_norm);

%% ========================================================================
%  Step 8, 9, 10: 性能分析与【全 3D 展示】
% =========================================================================
disp('--------------------------------------------------');
disp('8. 正在生成 3D 评估报表与误差分析图...');
test_idx = tr_c.testInd;
in_test = inputs_net_c(:, test_idx); 
Pb_test = feat_Pb_ideal(:, test_idx);
actual_after_test = raw_tg_s(:, test_idx);

pred_delta = mapstd('reverse', net_shape(mapstd('apply', in_test, ps_in_c)), ps_out_c);
pred_after_test = Pb_test + pred_delta;

% 误差计算
tip_dist = sqrt(sum((pred_after_test(19:21,:) - actual_after_test(19:21,:)).^2, 1));
mae_shape = mean(sqrt(sum((pred_after_test-actual_after_test).^2, 1))/7)*1000;
fprintf('   > [Net C 性能结果]\n');
fprintf('     平均形态误差 MAE: %.2f mm\n', mae_shape);
fprintf('     平均尖端误差 MAE: %.2f mm\n', mean(tip_dist)*1000);

% 🌟 3D 形态 GALLERY 展示 (4 样本)
figure('Name', '3D Shape Reconstruction Gallery', 'Color', 'w', 'Position', [100, 100, 1200, 500]);
p_ids = test_idx(randperm(length(test_idx), 4));
for k = 1:4
    idx_s = find(test_idx == p_ids(k));
    P_r = [[0;0;0], reshape(actual_after_test(:, idx_s), 3, [])];
    P_p = [[0;0;0], reshape(pred_after_test(:, idx_s), 3, [])];
    subplot(1, 4, k); hold on; grid on; axis equal;
    set(gca, 'zdir', 'reverse', 'ydir', 'reverse');
    plot3(P_r(1,:), P_r(2,:), P_r(3,:), 'k-s', 'LineWidth', 2, 'MarkerFaceColor','k');
    plot3(P_p(1,:), P_p(2,:), P_p(3,:), 'r--o', 'LineWidth', 1.5);
    xlabel('X (m)'); ylabel('Y (m)'); zlabel('Z (m)'); title(['Sample ', num2str(k)]);
    view(35, 25);
end

% 🌟 5 组最差 Tip Error 的 3D 详细诊断图 (独立窗口)
[sorted_err, s_idx] = sort(tip_dist, 'descend');
for k = 1:5
    l_idx = s_idx(k);
    P_rt = [[0;0;0], reshape(actual_after_test(:, l_idx), 3, [])];
    P_pt = [[0;0;0], reshape(pred_after_test(:, l_idx), 3, [])];
    
    figure('Name', sprintf('Worst Case #%d (%.2f mm)', k, sorted_err(k)*1000), 'Color', 'w', 'Position', [200, 200, 800, 700]);
    hold on; grid on; axis equal;
    set(gca, 'zdir', 'reverse', 'ydir', 'reverse');
    
    % 绘制真值骨架 (黑色)
    h1 = plot3(P_rt(1,:), P_rt(2,:), P_rt(3,:), 'k-s', 'LineWidth', 3, 'MarkerSize', 8, 'MarkerFaceColor','k');
    % 绘制预测骨架 (红色)
    h2 = plot3(P_pt(1,:), P_pt(2,:), P_pt(3,:), 'r--o', 'LineWidth', 2, 'MarkerSize', 8);
    % 绘制尖端误差连线 (紫红色)
    plot3([P_rt(1,end), P_pt(1,end)], [P_rt(2,end), P_pt(2,end)], [P_rt(3,end), P_pt(3,end)], 'm-', 'LineWidth', 4);
    
    % 绘制基准三轴 (RGB Quiver)
    quiver3(0,0,0, 0.05,0,0, 'r', 'LineWidth', 3, 'MaxHeadSize', 0.5); % X
    quiver3(0,0,0, 0,0.05,0, 'g', 'LineWidth', 3, 'MaxHeadSize', 0.5); % Y
    quiver3(0,0,0, 0,0,0.05, 'b', 'LineWidth', 3, 'MaxHeadSize', 0.5); % Z
    
    title(sprintf('Worst Case Analysis: %.2f mm Tip Error', sorted_err(k)*1000));
    xlabel('X (m)'); ylabel('Y (m)'); zlabel('Z (m)');
    legend([h1, h2], {'Actual Sensor (After)', 'System Prediction'}, 'Location', 'southoutside');
    view(40, 20);
end

% 🌟 Tip 追踪与直方图
figure('Name', 'Tip Performance Analysis', 'Color', 'w', 'Position', [100, 200, 1000, 450]);
subplot(1, 2, 1); hold on; grid on; axis equal; set(gca, 'zdir', 'reverse', 'ydir', 'reverse');
for k = 1:min(50, length(tip_dist))
    p_r = actual_after_test(19:21, k); p_p = pred_after_test(19:21, k);
    plot3([p_r(1), p_p(1)], [p_r(2), p_p(2)], [p_r(3), p_p(3)], 'Color', [0.7 0.7 0.7]);
    plot3(p_r(1), p_r(2), p_r(3), 'bo', 'MarkerFaceColor','b');
    plot3(p_p(1), p_p(2), p_p(3), 'r.', 'MarkerSize', 12);
end
title('Tip Tracking 3D Space'); view(45, 30);

subplot(1, 2, 2);
histogram(tip_dist * 1000, 25, 'FaceColor', [0.8 0.2 0.2]);
xline(mean(tip_dist)*1000, 'k--', 'LineWidth', 2, 'Label', 'MAE');
grid on; xlabel('Tip Error (mm)'); title('Error Dist.');

save('Corrected_Physics_Residual_Model.mat', 'net_force', 'net_loc', 'net_shape');
disp('>>> 流程运行完毕，逻辑完全拨乱反正。');

%% ========================================================================
%  Helper Function (Synchronized Rotation)
% =========================================================================
function [aug_Fd, aug_Fa, aug_Fb, aug_Pb, aug_Pa, aug_gF, aug_h] = augment_data_consistent(Fd, Fa, Fb, Pb, Pa, gF, h)
    N = size(Fd, 2); R120 = [cosd(120),-sind(120),0; sind(120),cosd(120),0; 0,0,1];
    R240 = [cosd(240),-sind(240),0; sind(240),cosd(240),0; 0,0,1];
    idx1 = [5,6,1,2,3,4]; idx2 = [3,4,5,6,1,2];
    rot = @(P, R) reshape(R * reshape(P, 3, []), 21, N);
    aug_Fd = [Fd, Fd(idx1,:), Fd(idx2,:)]; aug_Fa = [Fa, Fa(idx1,:), Fa(idx2,:)]; aug_Fb = [Fb, Fb(idx1,:), Fb(idx2,:)];
    aug_Pb = [Pb, rot(Pb, R120), rot(Pb, R240)]; aug_Pa = [Pa, rot(Pa, R120), rot(Pa, R240)];
    aug_gF = [gF, R120*gF, R240*gF]; aug_h = [h, h, h];
end