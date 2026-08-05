%% ========================================================================
%  Project: Deep Learning-Based Robust Proprioceptive Pose Reconstruction
%  Optimization: Units(mm), Relative Time, Latency Test & Cache
% =========================================================================
clc; clear; close all;
rng('default');

%% [配置项]
CACHE_FILE = 'PhysicsPriorsCache.mat'; 
USE_CACHE = true; % 如果已经生成过一次，改为 true 即可秒开程序
TEST_RATIO = 0.3;  % 留最后 20% 的连续数据用于"在线推理测试"

%% ========================================================================
%  Step 1: Robust Data Loading (保持你之前的解析逻辑)
% =========================================================================
disp('1. Loading and parsing data...');
FileName = '/Users/ryan/Desktop/continuum robot/dy_data_1858.xlsx'; 
dataTable = readtable(FileName, 'VariableNamingRule', 'preserve');
extract_col = @(tab, colIdx) reshape(double(string(tab{3:end, colIdx})), [], 1);
extract_range = @(tab, colRange) cell2mat(arrayfun(@(c) extract_col(tab, c), colRange, 'UniformOutput', false));

time_raw = extract_col(dataTable, 1);
F_all_raw = extract_range(dataTable, 2:7);
Fz_val = extract_col(dataTable, 10);
F_ext_mapped = [-abs(Fz_val), zeros(size(Fz_val)), zeros(size(Fz_val))];
raw_markers_mat = extract_range(dataTable, 17:43);
F_all_raw = F_all_raw * 0.00981;

%% ========================================================================
%  Step 2: Sequence Filtering & Dataset Splitting
% =========================================================================
disp('2. Processing sequence and splitting Test Set...');
F_diff_threshold = 1e-4; 
F_b_all = []; F_a_all = []; time_all = []; F_ext_all = []; P_gt_all = [];

for i = 1 : size(F_all_raw, 1) - 1
    Fb = F_all_raw(i, :)'; Fa = F_all_raw(i+1, :)';
    if norm(Fa - Fb) > F_diff_threshold
        F_b_all = [F_b_all, Fb]; F_a_all = [F_a_all, Fa];
        time_all = [time_all, time_raw(i+1)];
        F_ext_all = [F_ext_all, F_ext_mapped(i+1, :)'];
        m = raw_markers_mat(i+1, :); base_pos = m(1:3);
        p_seq = [m(1:3); m(10:12); m(13:15); m(16:18); m(4:6); m(19:21); m(22:24); m(25:27); m(7:9)]';
        P_gt_all = [P_gt_all, reshape(p_seq - base_pos', [], 1)];
    end
end
% 在训练输入前添加：
noise_level = 0.0; % 2% 噪声
F_a_all = F_a_all + noise_level * mean(F_a_all(:)) * randn(size(F_a_all));

% --- 关键：划分训练集和测试集 (保留后20%做在线模拟) ---
N_total = size(F_b_all, 2);
N_train = floor(N_total * (1 - TEST_RATIO));
train_idx = 1:N_train;
test_idx = N_train+1 : N_total;


%% ========================================================================
%  Step 3: Physics Priors (With Parallel Computing & Cache)
% =========================================================================
if USE_CACHE && isfile(CACHE_FILE)
    disp('3. Loading Physics Priors from Cache...');
    load(CACHE_FILE, 'P_before_ideal');
else
    disp('3. Generating Physics Priors (This may take a while)...');
    P_before_ideal = zeros(27, N_total);
    % 如果你有并行工具箱，把 for 改为 parfor 速度提升 4-8 倍
    for i = 1:N_total
        Fb_sim = [F_b_all(5,i); F_b_all(6,i); F_b_all(1,i); F_b_all(2,i); F_b_all(3,i); F_b_all(4,i)];
        [P_Theo, ~, R_mat, ~, ~, ~] = solve_continuum_shape_nofig(3, 2, 0.0006, 0.916e+12, 0.0665, 0.00, 18, linspace(0.0025, 0.0025, 37), 0.25, 0, 4.000*0.00981, [0;0;0], Fb_sim, 14);
        V_local = [0; -0.004; 0]; P_m = zeros(3, size(P_Theo, 2));
        for pt = 1:size(P_Theo, 2), P_m(:, pt) = P_Theo(:, pt) + R_mat(:, :, pt) * V_local; end
        P_before_ideal(:, i) = reshape(P_m(:, [1, 3, 5, 8, 10, 12, 14, 17, 19]), 27, 1);
    end
    save(CACHE_FILE, 'P_before_ideal');
end

%% ========================================================================
%  Step 4: Training & Online Inference Latency Test
% ========================================================================
disp('4. Training on Training Set & Measuring Latency...');
% 仅用训练集训练
inputs_B_train = [F_a_all(:,train_idx); F_a_all(:,train_idx)-F_b_all(:,train_idx); F_b_all(:,train_idx)];
targets_B_train = F_ext_all(1, train_idx);
net_force = train(feedforwardnet([10, 5]), inputs_B_train, targets_B_train);
net_force.trainFcn = 'traingd'; 
net_force.trainParam.lr = 0.01;
net.trainParam.epochs = 50;

inputs_C_train = [F_a_all(:,train_idx); net_force(inputs_B_train); repmat(5/19,1,N_train); P_before_ideal(:,train_idx)];
targets_C_train = P_gt_all(:,train_idx) - P_before_ideal(:,train_idx);
net_shape = train(fitnet([10,5]), inputs_C_train, targets_C_train);
net_shape.trainFcn = 'traingd'; 
net_shape.trainParam.lr = 0.01;
net.trainParam.epochs = 50;

% --- 在测试集上进行“模拟在线推理”并计时 ---
tic;
test_inputs_B = [F_a_all(:,test_idx); F_a_all(:,test_idx)-F_b_all(:,test_idx); F_b_all(:,test_idx)];
test_pred_F = net_force(test_inputs_B);
test_inputs_C = [F_a_all(:,test_idx); test_pred_F; repmat(5/19,1,length(test_idx)); P_before_ideal(:,test_idx)];
test_pred_delta = net_shape(test_inputs_C);
test_P_recon = P_before_ideal(:,test_idx) + test_pred_delta;
total_inference_time = toc;

fprintf('   > Average Online Inference Time: %.4f ms / sample\n', (total_inference_time/length(test_idx))*1000);
%% ========================================================================
%  Step 5: Visualizing Results (Scientific Plotting - Fixed Indexing)
% =========================================================================
disp('5. Visualizing Test Set Tracking...');

% --- 核心修正：直接使用推理结果的全部列，不需要再用 test_idx 索引 ---
% 因为 test_P_recon 本身就是针对测试集生成的，它的列 1 就对应测试集的第 1 个样本

% 1. 处理时间轴：让测试集的时间从 0 开始
time_test_raw = time_all(test_idx);
time_test_relative = time_test_raw - time_test_raw(1); 

% 2. 提取真值与预测值 (转为 mm)
% 真值需要从全量 P_gt_all 中截取测试集部分
P_gt_test_mm = P_gt_all(25:27, test_idx); 

% 预测值 test_P_recon 已经是测试集大小，直接取最后三行 (Tip)
P_pred_test_mm = test_P_recon(25:27, :); 

% 3. 绘图
figure('Name', 'Real-time Test Set Tracking', 'Color', 'w', 'Position', [100, 100, 1000, 800]);
dim_names = {'X', 'Y', 'Z'};
line_colors = {[0 0.447 0.741], [0.85 0.325 0.098], [0.929 0.694 0.125]}; % 后面备用

for i = 1:3
    subplot(3, 1, i);
    % 画真值
    plot(time_test_relative, P_gt_test_mm(i, :), 'k-', 'LineWidth', 1.8, 'DisplayName', 'Ground Truth');
    hold on;
    % 画预测值
    plot(time_test_relative, P_pred_test_mm(i, :), 'r--', 'LineWidth', 1.5, 'DisplayName', 'PGR Prediction');
    
    grid on; box on;
    set(gca, 'FontSize', 11, 'LineWidth', 1);
    ylabel(['$', dim_names{i}, '$ (mm)'], 'Interpreter', 'latex', 'FontSize', 13);
    
    if i == 1
        title('\textbf{Dynamic Pose Reconstruction on Unseen Test Set}', 'Interpreter', 'latex', 'FontSize', 15);
    end
    if i == 3
        xlabel('Relative Time (s)', 'Interpreter', 'latex', 'FontSize', 13);
    end
    legend('Location', 'best', 'FontSize', 10);
end

% 4. 统计误差
test_errors = sqrt(sum((P_gt_test_mm - P_pred_test_mm).^2, 1)); % 每个样本的欧氏距离误差
final_test_mae = mean(test_errors);
final_test_std = std(test_errors);

fprintf('--------------------------------------------------\n');
fprintf('Analysis Complete (Test Set Performance):\n');
fprintf('   > Mean Tip Error: %.2f mm\n', final_test_mae);
fprintf('   > Error Std Dev:  %.2f mm\n', final_test_std);
fprintf('   > Max Tip Error:  %.2f mm\n', max(test_errors));
fprintf('   > Average Inference Latency: %.4f ms\n', (total_inference_time/length(test_idx))*1000);
%% ========================================================================
%  Step 6: 综合实验分析图 - 动力学演化与位姿还原追踪
% =========================================================================
disp('--------------------------------------------------');
disp('6. Generating Comprehensive Tracking Visualization...');

% --- 1. 数据准备 ---
t_rel = time_test_relative;           % 相对时间
F_tendon = F_a_all(:, test_idx);      % 6根腱绳的实时张力
F_ext_gt = F_ext_all(1, test_idx);    % 外力真值 (X方向)
F_ext_pd = test_pred_F;               % Net B 预测的外力

P_gt = P_gt_test_mm;                  % 位姿真值 XYZ
P_pd = P_pred_test_mm;                % 位姿预测 XYZ
err_instant = sqrt(sum((P_gt - P_pd).^2, 1)); % 瞬时欧氏距离误差

% --- 2. 绘图设置 ---
figure('Name', 'System Performance Analysis', 'Color', 'w', 'Position', [50, 50, 1000, 900]);

% --- Subplot 1: 腱绳张力输入 (显示驱动力波动) ---
subplot(4, 1, 1);
plot(t_rel, F_tendon', 'LineWidth', 1);
grid on; ylabel('Tensions (N)');
title('\textbf{Stage 1: Input Tendon Tension Dynamics}', 'Interpreter', 'latex');
legend({'T1','T2','T3','T4','T5','T6'}, 'Location', 'eastoutside', 'FontSize', 7);

% --- Subplot 2: 外力感知 (显示外力从无到有，GT vs Pred) ---
subplot(4, 1, 2);
plot(t_rel, F_ext_gt, 'k-', 'LineWidth', 2, 'DisplayName', 'Force GT'); hold on;
plot(t_rel, F_ext_pd, 'r--', 'LineWidth', 1.5, 'DisplayName', 'Net B Pred');
grid on; ylabel('Ext. Force (N)');
title('\textbf{Stage 2: External Force Perception (Net B)}', 'Interpreter', 'latex');
legend('Location', 'best');

% --- Subplot 3: 位姿还原追踪 (XYZ 随动图) ---
subplot(4, 1, 3);
plot(t_rel, P_gt(1,:), 'k-', 'LineWidth', 1.5); hold on; % 只取X轴作为演示，或画总位移
plot(t_rel, P_pd(1,:), 'r--', 'LineWidth', 1.5);
grid on; ylabel('Tip-X (mm)');
title('\textbf{Stage 3: Pose Reconstruction (Net C)}', 'Interpreter', 'latex');
legend({'Truth', 'PGR Pred.'}, 'Location', 'best');

% --- Subplot 4: 实时重构误差 (显示精度波动) ---
subplot(4, 1, 4);
area(t_rel, err_instant, 'FaceColor', [0.8 0.2 0.2], 'FaceAlpha', 0.3, 'EdgeColor', 'r');
grid on; ylabel('Error (mm)'); xlabel('Time (s)');
ylim([0, max(err_instant)*1.5]);
title('\textbf{Stage 4: Real-time Tip Euclidean Error}', 'Interpreter', 'latex');

% 自动优化排版
sgtitle('Continuum Robot Dynamic Proprioception Analysis', 'FontSize', 16, 'FontWeight', 'bold');
%% ========================================================================
%  Step 6: 高级系统综合分析图 - 包含基准态与全轴随动追踪
% =========================================================================
disp('--------------------------------------------------');
disp('6. Generating Multi-Axis System Evolution Plot...');

% --- 1. 构建 0 时刻前的基准态 (Base State) ---
N_base = 0; % 基准态点数
dt = mean(diff(time_test_relative));
t_base = linspace(-dt*N_base, -dt, N_base); % -2s 到 0s 左右

% 模拟基准态数据
F_tendon_base = 3.0 + 0.02 * randn(6, N_base); % 约 3N 的预紧力
F_ext_base = zeros(1, N_base);                 % 外力为 0
P_xyz_base = repmat(P_gt_test_mm(:, 1), 1, N_base); % 初始位姿
err_base = zeros(1, N_base);                   % 基准态误差为 0

% --- 2. 拼接数据 (Base + Test) ---
t_total = [t_base, time_test_relative];
F_tendon_total = [F_tendon_base, F_a_all(:, test_idx)];
F_ext_gt_total = [F_ext_base, F_ext_all(1, test_idx)];
F_ext_pd_total = [F_ext_base, test_pred_F];

P_gt_total = [P_xyz_base, P_gt_test_mm];
P_pd_total = [P_xyz_base, P_pred_test_mm];
err_instant = sqrt(sum((P_gt_total - P_pd_total).^2, 1));

% --- 3. 统计数据输出 (仅输出到终端) ---
fprintf('\n--- [Final Performance Statistics] ---\n');
% 位姿误差统计
fprintf('Tip Pose Reconstruction (XYZ):\n');
fprintf('   > Mean Euclidean Error: %.4f mm\n', mean(err_instant(N_base+1:end)));
fprintf('   > Max Euclidean Error:  %.4f mm\n', max(err_instant(N_base+1:end)));
fprintf('   > Error Std Dev:       %.4f mm\n', std(err_instant(N_base+1:end)));
% 外力感知统计
force_err = abs(F_ext_gt_total(N_base+1:end) - F_ext_pd_total(N_base+1:end));
fprintf('External Force Perception (X-axis):\n');
fprintf('   > Mean Force Error:     %.4f N\n', mean(force_err));
fprintf('   > Max Force Error:      %.4f N\n', max(force_err));
fprintf('--------------------------------------\n');

% --- 4. 绘图 ---
figure('Name', 'Comprehensive Dynamic Analysis', 'Color', 'w', 'Position', [50, 50, 1000, 1000]);

% A. 腱绳张力 (驱动层)
subplot(6, 1, 1);
plot(t_total, F_tendon_total', 'LineWidth', 1); hold on;
xline(0, 'k--', 'LineWidth', 1.5, 'HandleVisibility','off');
grid on; ylabel('Tensions (N)');
title('\textbf{Tendon Tension Input (Pre-load $\approx$ 3N)}', 'Interpreter', 'latex');

% B. 外力感知 (感知层)
subplot(6, 1, 2);
plot(t_total, F_ext_gt_total, 'k-', 'LineWidth', 1.5, 'DisplayName', 'GT'); hold on;
plot(t_total, F_ext_pd_total, 'r--', 'LineWidth', 1.2, 'DisplayName', 'Pred');
xline(0, 'k--', 'LineWidth', 1.5, 'HandleVisibility','off');
grid on; ylabel('Ext. Force (N)');
title('\textbf{External Force Perception (X-axis)}', 'Interpreter', 'latex');
legend('Location', 'northeast');

% C. Tip X 还原
subplot(6, 1, 3);
plot(t_total, P_gt_total(1,:), 'k-', 'LineWidth', 1.5); hold on;
plot(t_total, P_pd_total(1,:), 'r--', 'LineWidth', 1.2);
xline(0, 'k--', 'LineWidth', 1.5);
grid on; ylabel('X (mm)');
title('\textbf{Tip Pose Tracking: X-axis}', 'Interpreter', 'latex');

% D. Tip Y 还原
subplot(6, 1, 4);
plot(t_total, P_gt_total(2,:), 'k-', 'LineWidth', 1.5); hold on;
plot(t_total, P_pd_total(2,:), 'r--', 'LineWidth', 1.2);
xline(0, 'k--', 'LineWidth', 1.5);
grid on; ylabel('Y (mm)');
title('\textbf{Tip Pose Tracking: Y-axis}', 'Interpreter', 'latex');

% E. Tip Z 还原
subplot(6, 1, 5);
plot(t_total, P_gt_total(3,:), 'k-', 'LineWidth', 1.5); hold on;
plot(t_total, P_pd_total(3,:), 'r--', 'LineWidth', 1.2);
xline(0, 'k--', 'LineWidth', 1.5);
grid on; ylabel('Z (mm)');
title('\textbf{Tip Pose Tracking: Z-axis}', 'Interpreter', 'latex');

% F. 欧氏距离误差 (评价层)
subplot(6, 1, 6);
area(t_total, err_instant, 'FaceColor', [0.8 0.2 0.2], 'FaceAlpha', 0.3, 'EdgeColor', 'r');
hold on; xline(0, 'k--', 'LineWidth', 1.5);
grid on; ylabel('Error (mm)'); xlabel('Time (s)');
title('\textbf{Real-time Euclidean Tip Error}', 'Interpreter', 'latex');

% 调整布局
sgtitle(['Dynamic Proprioception: Baseline (t<0) to Active Loading (t>0)'], 'FontSize', 14, 'FontWeight', 'bold');
%% ========================================================================
%  Step 6: IEEE Half-Column Dynamic Results Figure
%
%  Figure content:
%  (a) External contact-force tracking
%  (b) Dynamic tip-position tracking along the dominant X direction
%  (c) Instantaneous Euclidean tip reconstruction error
%
%  Notes:
%  - This section only changes visualization.
%  - No data are regenerated or modified.
%  - The plotted samples may be downsampled for visual clarity only.
% =========================================================================
disp('--------------------------------------------------');
disp('6. Generating IEEE half-column dynamic results figure...');

%% ------------------------------------------------------------------------
%  USER CONTROLS
% -------------------------------------------------------------------------
show_tick_labels = false;      % Show numerical tick labels
show_legend      = false;      % Show legends
show_axis_labels = false;      % Show x/y-axis labels
show_panel_titles = false;     % Show compact panel titles
show_key_markers = false;      % Show representative-state vertical markers

% Plotting density only; this does NOT change the data or statistics.
max_plot_points = 220;

% Number of representative states used for later photo composition.
num_key_states = 4;

% Export control.
export_figure = false;
export_file_name = 'dynamic_results_half_column.pdf';
export_dpi = 600;

% IEEE half-column target size.
figure_width_cm  = 8.6;
figure_height_cm = 10.2;

%% ------------------------------------------------------------------------
%  1. DATA PREPARATION
% -------------------------------------------------------------------------

% Display time window for the final figure.
display_time = [0, 30];   % [start time, end time] in seconds

% -------------------------------------------------------------------------
% First obtain ALL full-length test data.
% -------------------------------------------------------------------------
t_all_plot = time_test_relative(:)';

F_ext_gt_all_plot = reshape(F_ext_all(1, test_idx), 1, []);
F_ext_pd_all_plot = reshape(test_pred_F, 1, []);

tip_gt_x_all_plot = reshape(P_gt_test_mm(1, :), 1, []);
tip_pd_x_all_plot = reshape(P_pred_test_mm(1, :), 1, []);

tip_error_all_plot = reshape( ...
    sqrt(sum((P_gt_test_mm - P_pred_test_mm).^2, 1)), ...
    1, []);

% -------------------------------------------------------------------------
% Check the original data lengths BEFORE cropping.
% -------------------------------------------------------------------------
N_original = numel(t_all_plot);

assert(numel(F_ext_gt_all_plot) == N_original, ...
    'Original force ground-truth length does not match the time vector.');

assert(numel(F_ext_pd_all_plot) == N_original, ...
    'Original predicted-force length does not match the time vector.');

assert(numel(tip_gt_x_all_plot) == N_original, ...
    'Original Tip-X ground-truth length does not match the time vector.');

assert(numel(tip_pd_x_all_plot) == N_original, ...
    'Original predicted Tip-X length does not match the time vector.');

assert(numel(tip_error_all_plot) == N_original, ...
    'Original tip-error length does not match the time vector.');

% -------------------------------------------------------------------------
% Use ONE common mask to crop every plotted variable.
% -------------------------------------------------------------------------
display_mask = ...
    t_all_plot >= display_time(1) & ...
    t_all_plot <= display_time(2);

assert(any(display_mask), ...
    'No samples were found inside the requested display time window.');

t_plot_full = t_all_plot(display_mask);

F_ext_gt_full = F_ext_gt_all_plot(display_mask);
F_ext_pd_full = F_ext_pd_all_plot(display_mask);

tip_gt_x_full = tip_gt_x_all_plot(display_mask);
tip_pd_x_full = tip_pd_x_all_plot(display_mask);

tip_error_full = tip_error_all_plot(display_mask);

% Final cropped-data consistency check.
N_plot = numel(t_plot_full);

assert(numel(F_ext_gt_full) == N_plot, ...
    'Force ground-truth length does not match the cropped time vector.');

assert(numel(F_ext_pd_full) == N_plot, ...
    'Predicted-force length does not match the cropped time vector.');

assert(numel(tip_gt_x_full) == N_plot, ...
    'Tip-X ground-truth length does not match the cropped time vector.');

assert(numel(tip_pd_x_full) == N_plot, ...
    'Predicted Tip-X length does not match the cropped time vector.');

assert(numel(tip_error_full) == N_plot, ...
    'Tip-error length does not match the cropped time vector.');

%% ------------------------------------------------------------------------
%  3. REPRESENTATIVE DYNAMIC STATES
%
%  The states are selected from the measured sequence:
%  State 1: beginning of the test sequence
%  State 2: strongest external loading
%  State 3: representative recovery/transition state
%  State 4: end of the test sequence
%
%  These markers can guide later photo composition. Do not claim exact
%  synchronization unless the photographs were acquired synchronously.
% -------------------------------------------------------------------------
if num_key_states ~= 4
    warning(['The current automatic selector is designed for four states. ', ...
        'num_key_states is reset to 4.']);
    num_key_states = 4;
end

idx_state_1 = 1;

% Strongest measured contact-force magnitude.
[~, idx_state_2] = max(abs(F_ext_gt_full));

% Select a transition state after the strongest loading event.
search_start = min(idx_state_2 + 1, N_plot);

if search_start < N_plot
    force_after_peak = abs(F_ext_gt_full(search_start:end));

    % Prefer a relatively recovered state after the peak.
    [~, local_transition_idx] = min(force_after_peak);
    idx_state_3 = search_start + local_transition_idx - 1;
else
    idx_state_3 = max(1, round(0.75 * N_plot));
end

idx_state_4 = N_plot;

key_idx = unique([ ...
    idx_state_1, ...
    idx_state_2, ...
    idx_state_3, ...
    idx_state_4], ...
    'stable');

% If two automatically selected states coincide, use temporal quartiles.
if numel(key_idx) < 4
    key_idx = unique(round(linspace(1, N_plot, 4)), 'stable');
end

key_time = t_plot_full(key_idx);

%% ------------------------------------------------------------------------
%  4. PERFORMANCE STATISTICS
% -------------------------------------------------------------------------
mean_tip_error = mean(tip_error_full);
std_tip_error  = std(tip_error_full);
max_tip_error  = max(tip_error_full);

force_abs_error = abs(F_ext_gt_full - F_ext_pd_full);
mean_force_error = mean(force_abs_error);

fprintf('\n--- Dynamic Test Visualization Statistics ---\n');
fprintf('Mean tip error:      %.3f mm\n', mean_tip_error);
fprintf('Tip error std.:      %.3f mm\n', std_tip_error);
fprintf('Maximum tip error:   %.3f mm\n', max_tip_error);
fprintf('Mean force error:    %.4f N\n', mean_force_error);
fprintf('Displayed samples:   %d / %d\n', numel(plot_idx), N_plot);
fprintf('---------------------------------------------\n');

%% ------------------------------------------------------------------------
%  5. FIGURE STYLE
% -------------------------------------------------------------------------
font_name = 'Times New Roman';

axis_font_size  = 8.0;
label_font_size = 8.5;
title_font_size = 8.5;
legend_font_size = 7.5;

axis_line_width = 0.85;
curve_line_width_gt = 1.35;
curve_line_width_pd = 1.20;
error_line_width = 1.15;

% Keep the original visual identity:
% GT: black solid
% Prediction: red dashed
color_gt = [0, 0, 0];
color_pred = [1, 0, 0];

% Light reference-line color.
color_reference = [0.35, 0.35, 0.35];

% Representative-state marker color.
color_key = [0.20, 0.40, 0.70];

%% ------------------------------------------------------------------------
%  6. CREATE HALF-COLUMN FIGURE
% -------------------------------------------------------------------------
fig_dynamic = figure( ...
    'Name', 'IEEE Half-Column Dynamic Results', ...
    'Color', 'w', ...
    'Units', 'centimeters', ...
    'Position', [3, 3, figure_width_cm, figure_height_cm], ...
    'PaperPositionMode', 'auto', ...
    'Renderer', 'painters');

layout = tiledlayout(fig_dynamic, 3, 1, ...
    'TileSpacing', 'compact', ...
    'Padding', 'compact');



%% ------------------------------------------------------------------------
%  PANEL (b): TIP-X TRACKING
% -------------------------------------------------------------------------
ax_tip = nexttile(layout, 1);

plot(ax_tip, ...
    t_plot, ...
    tip_gt_x_plot, ...
    '-', ...
    'Color', color_gt, ...
    'LineWidth', curve_line_width_gt, ...
    'DisplayName', 'Ground truth');

hold(ax_tip, 'on');

plot(ax_tip, ...
    t_plot, ...
    tip_pd_x_plot, ...
    '--', ...
    'Color', color_pred, ...
    'LineWidth', curve_line_width_pd, ...
    'DisplayName', 'Reconstruction');

if show_key_markers
    add_key_state_markers( ...
        ax_tip, ...
        key_time, ...
        color_key, ...
        false, ...
        font_name);
end

if show_panel_titles
    title(ax_tip, ...
        '(b) Dynamic tip-position tracking', ...
        'FontName', font_name, ...
        'FontSize', title_font_size, ...
        'FontWeight', 'normal');
end

if show_axis_labels
    ylabel(ax_tip, ...
        'Tip-X (mm)', ...
        'FontName', font_name, ...
        'FontSize', label_font_size);
end

if show_legend
    legend(ax_tip, ...
        'Location', 'best', ...
        'FontName', font_name, ...
        'FontSize', legend_font_size, ...
        'Box', 'off');
end

%% ------------------------------------------------------------------------
%  PANEL (c): TIP RECONSTRUCTION ERROR
% -------------------------------------------------------------------------
ax_error = nexttile(layout, 2);
area( ...
    ax_error, ...
    t_plot, ...
    tip_error_plot, ...
    'FaceColor', [0.8, 0.2, 0.2], ...
    'FaceAlpha', 0.30, ...
    'EdgeColor', 'r', ...
    'LineWidth', 0.9, ...
    'DisplayName', 'Tip error');

hold(ax_error, 'on');

% Mean-error reference line.
% yline(ax_error, ...
%     mean_tip_error, ...
%     '--', ...
%     sprintf('Mean = %.2f mm', mean_tip_error), ...
%     'Color', color_reference, ...
%     'LineWidth', 0.8, ...
%     'FontName', font_name, ...
%     'FontSize', 7.0, ...
%     'LabelHorizontalAlignment', 'right', ...
%     'LabelVerticalAlignment', 'bottom', ...
%     'HandleVisibility', 'off');

if show_key_markers
    add_key_state_markers( ...
        ax_error, ...
        key_time, ...
        color_key, ...
        false, ...
        font_name);
end

if show_panel_titles
    title(ax_error, ...
        '(c) Instantaneous tip reconstruction error', ...
        'FontName', font_name, ...
        'FontSize', title_font_size, ...
        'FontWeight', 'normal');
end

if show_axis_labels
    ylabel(ax_error, ...
        'Error (mm)', ...
        'FontName', font_name, ...
        'FontSize', label_font_size);

    xlabel(ax_error, ...
        'Time (s)', ...
        'FontName', font_name, ...
        'FontSize', label_font_size);
end

if show_legend
    legend(ax_error, ...
        'Location', 'best', ...
        'FontName', font_name, ...
        'FontSize', legend_font_size, ...
        'Box', 'off');
end
%% ------------------------------------------------------------------------
%  PANEL (a): EXTERNAL CONTACT FORCE
% -------------------------------------------------------------------------
ax_force = nexttile(layout, 3);

plot(ax_force, ...
    t_plot, ...
    F_ext_gt_plot, ...
    '-', ...
    'Color', color_gt, ...
    'LineWidth', curve_line_width_gt, ...
    'DisplayName', 'Ground truth');

hold(ax_force, 'on');

plot(ax_force, ...
    t_plot, ...
    F_ext_pd_plot, ...
    '--', ...
    'Color', color_pred, ...
    'LineWidth', curve_line_width_pd, ...
    'DisplayName', 'Prediction');

yline(ax_force, ...
    0, ...
    ':', ...
    'Color', color_reference, ...
    'LineWidth', 0.7, ...
    'HandleVisibility', 'off');

if show_key_markers
    add_key_state_markers( ...
        ax_force, ...
        key_time, ...
        color_key, ...
        true, ...
        font_name);
end

if show_panel_titles
    title(ax_force, ...
        '(a) External contact force', ...
        'FontName', font_name, ...
        'FontSize', title_font_size, ...
        'FontWeight', 'normal');
end

if show_axis_labels
    ylabel(ax_force, ...
        'Force (N)', ...
        'FontName', font_name, ...
        'FontSize', label_font_size);
end

if show_legend
    legend(ax_force, ...
        'Location', 'best', ...
        'FontName', font_name, ...
        'FontSize', legend_font_size, ...
        'Box', 'off');
end
%% ------------------------------------------------------------------------
%  7. UNIFIED AXIS FORMATTING
% -------------------------------------------------------------------------
all_axes = [ax_force, ax_tip, ax_error];

for ax = all_axes
    set(ax, ...
        'FontName', font_name, ...
        'FontSize', axis_font_size, ...
        'LineWidth', axis_line_width, ...
        'Box', 'on', ...
        'TickDir', 'in', ...
        'TickLength', [0.018, 0.018], ...
        'Layer', 'top', ...
        'XMinorTick', 'off', ...
        'YMinorTick', 'off');

    grid(ax, 'off');
    ax.GridAlpha = 0.13;
    ax.GridLineStyle = '-';

    % xlim(ax, [t_plot_full(1), t_plot_full(end)]);
    xlim(ax, display_time);
end

% The upper panels share the same time axis but do not need repeated labels.
if show_tick_labels
    ax_force.XTickLabel = [];
    ax_tip.XTickLabel = [];
else
    for ax = all_axes
        ax.XTickLabel = [];
        ax.YTickLabel = [];
    end
end

% Preserve tick marks while hiding axis-label text.
if ~show_axis_labels
    ylabel(ax_force, '');
    ylabel(ax_tip, '');
    ylabel(ax_error, '');
    xlabel(ax_error, '');
end

% Link the horizontal axes.
linkaxes(all_axes, 'x');

%% ------------------------------------------------------------------------
%  8. OPTIONAL EXPORT
% -------------------------------------------------------------------------
if export_figure
    [~, ~, export_extension] = fileparts(export_file_name);

    if strcmpi(export_extension, '.pdf')
        exportgraphics( ...
            fig_dynamic, ...
            export_file_name, ...
            'ContentType', 'vector', ...
            'BackgroundColor', 'white');
    else
        exportgraphics( ...
            fig_dynamic, ...
            export_file_name, ...
            'Resolution', export_dpi, ...
            'BackgroundColor', 'white');
    end

    fprintf('Dynamic figure exported to:\n%s\n', export_file_name);
end

%% ========================================================================
%  LOCAL FUNCTION
% =========================================================================
function add_key_state_markers( ...
    ax, ...
    key_time, ...
    marker_color, ...
    show_numbers, ...
    font_name)

    for k = 1:numel(key_time)

        xline(ax, ...
            key_time(k), ...
            '--', ...
            'Color', marker_color, ...
            'LineWidth', 0.75, ...
            'HandleVisibility', 'off');

        if show_numbers
            y_limits = ylim(ax);

            y_text = y_limits(2) ...
                - 0.07 * (y_limits(2) - y_limits(1));

            text(ax, ...
                key_time(k), ...
                y_text, ...
                sprintf('%d', k), ...
                'HorizontalAlignment', 'center', ...
                'VerticalAlignment', 'top', ...
                'FontName', font_name, ...
                'FontSize', 7.5, ...
                'FontWeight', 'bold', ...
                'Color', marker_color, ...
                'BackgroundColor', 'white', ...
                'Margin', 0.8, ...
                'Clipping', 'on');
        end
    end
end