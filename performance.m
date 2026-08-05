%% ========================================================================
%  Script: Model vs. Ground Truth Comparison & Error Analysis
%  Goal:   Load checkpoint, predict on Test Set, visualize & generate report
% =========================================================================
clc; clear; close all;

%% 1. 加载训练好的模型和测试集索引
disp('1. Loading System Checkpoint...');
if ~isfile('Final_System_Checkpoint.mat')
    error('错误：找不到 Final_System_Checkpoint.mat，请检查路径！');
end
load('Final_System_Checkpoint.mat');
% 包含变量: net_shape (Net C), ps_in_c, ps_out_c, inputs_net_c, targets_net_c, test_idx 等

disp('   > 模型加载成功。');

%% 2. 准备测试数据与执行预测
disp('2. Running Inference on Test Set...');

% 提取测试集的输入和真值
% inputs_net_c 结构: [Internal_Force(6); External_Force(3); Location(1)]
% targets_net_c 结构: [Shape_Coords(21)]
X_Test = inputs_net_c(:, test_idx);
Y_True = targets_net_c(:, test_idx); 

% 归一化输入
X_Test_Norm = mapstd('apply', X_Test, ps_in_c);

% 网络预测 (Net C)
Y_Pred_Norm = net_shape(X_Test_Norm);

% 反归一化输出 (还原为真实物理坐标 m)
Y_Pred = mapstd('reverse', Y_Pred_Norm, ps_out_c);

fprintf('   > 已对 %d 组测试样本完成预测。\n', length(test_idx));

%% 3. 提取物理属性用于分析 (用于生成表格)
% 我们需要从输入数据 X_Test 中把物理含义解析出来

% Row 7-9: External Force Vector (F_ext_x, F_ext_y, F_ext_z)
F_ext_vec = X_Test(7:9, :);
Force_Mag = sqrt(sum(F_ext_vec.^2, 1)); % 外力大小 (N)

% Row 10: Normalized Location (0.xx)
% 还原为 Node Index (3, 4, 5)
Loc_Norm = X_Test(10, :);
Node_Index = round(Loc_Norm * 9.0); 

% 计算 Tip 误差 (最后3个坐标: x,y,z)
Tip_True = Y_True(19:21, :);
Tip_Pred = Y_Pred(19:21, :);
Tip_Error_Vec = Tip_Pred - Tip_True;
Tip_Error_mm = sqrt(sum(Tip_Error_Vec.^2, 1)) * 1000; % 转换为 mm

%% 4. 可视化对比 (复刻你的参考风格)
disp('3. Visualizing Comparison...');

% --- 随机挑选 15 组样本展示，避免图太乱 ---
num_samples = size(X_Test, 2);
num_plots = 10; 
plot_indices = randperm(num_samples, num_plots);

figure('Name', 'Model Prediction vs Ground Truth', 'Color', 'w', 'Position', [100, 100, 1000, 700]);
hold on; grid on; axis equal;

% --- 配色方案 (双色系) ---
% 预测值 (Model): 蓝色系
model_color = [0.0, 0.45, 0.74]; 
% 真值 (Ground Truth): 橙红色系
real_color  = [0.85, 0.33, 0.10]; 

for k = 1:num_plots
    idx = plot_indices(k);
    
    % 提取单条骨架数据 (3x7)
    % 并在头部加 (0,0,0) 基座点
    shape_true = [[0;0;0], reshape(Y_True(:, idx), 3, [])];
    shape_pred = [[0;0;0], reshape(Y_Pred(:, idx), 3, [])];
    
    % 1. 绘制真值 (实线 + 实心点)
    plot3(shape_true(1,:), shape_true(2,:), shape_true(3,:), '-', ...
        'Color', [real_color, 0.6], 'LineWidth', 2, ...
        'Marker', 's', 'MarkerSize', 4, 'MarkerFaceColor', real_color);
    
    % 2. 绘制预测 (虚线 + 空心圆)
    plot3(shape_pred(1,:), shape_pred(2,:), shape_pred(3,:), '--', ...
        'Color', [model_color, 0.8], 'LineWidth', 1.5, ...
        'Marker', 'o', 'MarkerSize', 4, 'MarkerFaceColor', 'w');

    % % 3. 画一条细灰线连接 Tip 点，直观展示误差
    % plot3([shape_true(1,end), shape_pred(1,end)], ...
    %       [shape_true(2,end), shape_pred(2,end)], ...
    %       [shape_true(3,end), shape_pred(3,end)], ...
    %       'Color', [0.5, 0.5, 0.5, 0.5], 'LineWidth', 0.5);
end

% --- 美化坐标轴 ---
% 绘制基坐标系箭头
quiver3(0, 0, 0, 0.04, 0, 0, 'r', 'LineWidth', 2); 
quiver3(0, 0, 0, 0, 0.04, 0, 'g', 'LineWidth', 2); 
quiver3(0, 0, 0, 0, 0, 0.04, 'b', 'LineWidth', 2); 

xlabel('X (m)'); ylabel('Y (m)'); zlabel('Z (m)');
title(sprintf('Model Evaluation (Test Set N=%d)', num_plots));

% 倒挂视角设置 (保留你的习惯)
set(gca, 'ZDir', 'reverse'); 
set(gca, 'YDir', 'reverse');
view(30, 20); 
set(gca, 'FontSize', 11, 'LineWidth', 1.2);

% 虚拟图例 (只显示一次)
h1 = plot3(NaN,NaN,NaN, '-s', 'Color', real_color, 'MarkerFaceColor', real_color, 'LineWidth', 2);
h2 = plot3(NaN,NaN,NaN, '--o', 'Color', model_color, 'MarkerFaceColor', 'w', 'LineWidth', 1.5);
legend([h1, h2], {'Ground Truth (Nokov)', 'Prediction (Net C)'}, 'Location', 'northeast');

hold off;

%% 5. 生成误差分析表 (Error Analysis Table)
disp('4. Generating Error Report...');

% 从 plot_indices 创建表格，只展示图上画出来的这几组，方便对照
% 或者你可以选择展示所有测试集的前20个

report_indices = plot_indices; % 对应图上的样本

% 提取数据列
tab_ID    = report_indices';
tab_Node  = Node_Index(report_indices)';
tab_Force = Force_Mag(report_indices)';
tab_Err   = Tip_Error_mm(report_indices)';

% 创建表格
ResultsTable = table(tab_ID, tab_Node, tab_Force, tab_Err, ...
    'VariableNames', {'SampleID', 'Contact_Node', 'Force_Mag_N', 'Tip_Error_mm'});

% 排序 (按误差从小到大排序，或者按节点排序)
ResultsTable = sortrows(ResultsTable, 'Tip_Error_mm');

% 显示表格
disp(' ');
disp('============== [Evaluation Report] ==============');
disp(ResultsTable);
disp('=================================================');
fprintf('测试集平均 Tip 误差: %.2f mm\n', mean(Tip_Error_mm));
fprintf('测试集最大 Tip 误差: %.2f mm\n', max(Tip_Error_mm));
%% === 侦探模式：寻找 40mm 误差的罪魁祸首 ===
disp('--------------------------------------------------');
disp('🔍 正在定位最大误差样本...');

% 1. 找到最大误差的索引
[max_err_val, max_loc_idx] = max(Tip_Error_mm); % max_loc_idx 是在测试集中的位置
global_idx = test_idx(max_loc_idx);             % global_idx 是在原始全集中的位置

% 2. 提取该样本的详细信息
culprit_force_mag = Force_Mag(max_loc_idx);
culprit_node      = Node_Index(max_loc_idx);

% 3. 打印详细信息
fprintf('found!\n');
fprintf('   > 样本 ID (Global): %d\n', global_idx);
fprintf('   > 样本 ID (Test Set): %d\n', max_loc_idx);
fprintf('   > 接触位置 (Node):   %d\n', culprit_node);
fprintf('   > 外力大小 (Force):  %.4f N\n', culprit_force_mag);
fprintf('   > 产生的误差:        %.2f mm\n', max_err_val);

% 4. 可视化：看看它到底错哪了？
figure('Name', 'The Culprit Analysis', 'Color', 'w');
hold on; grid on; axis equal;

% 提取坐标
P_bad_pred = reshape(Y_Pred(:, max_loc_idx), 3, []);
P_bad_true = reshape(Y_True(:, max_loc_idx), 3, []);

% 加基座
P_bad_pred = [[0;0;0], P_bad_pred];
P_bad_true = [[0;0;0], P_bad_true];

% 绘图
plot3(P_bad_true(1,:), P_bad_true(2,:), P_bad_true(3,:), 'k-s', 'LineWidth', 2, 'DisplayName', 'Ground Truth (Nokov)');
plot3(P_bad_pred(1,:), P_bad_pred(2,:), P_bad_pred(3,:), 'r--o', 'LineWidth', 2, 'DisplayName', 'Prediction (Wrong!)');

% 画连接线展示误差
tip_true = P_bad_true(:, end);
tip_pred = P_bad_pred(:, end);
plot3([tip_true(1), tip_pred(1)], [tip_true(2), tip_pred(2)], [tip_true(3), tip_pred(3)], 'm-', 'LineWidth', 2, 'DisplayName', 'Error Vector');

legend;
xlabel('X'); ylabel('Y'); zlabel('Z');
title(sprintf('Worst Case Analysis (Err: %.1f mm)', max_err_val));
view(30, 20);
%% ========================================================================
%  Script: Robust Speed Test (Fixed Dimensions)
%  Goal:   Measure latency correctly without bsxfun errors
% =========================================================================


% 1. 确保环境里有数据（如果没有就加载）
if ~exist('net_loc', 'var') || ~exist('ps_in', 'var')
    disp('Loading checkpoint...');
    load('Final_System_Checkpoint.mat');
end

disp('--------------------------------------------------');
disp('🚀 正在进行速度测试 (Robust Mode)...');

% 2. 【关键修正】明确区分 Net B 和 Net C 的输入数据
% 不要混用 X_Test，直接从源头取
sample_B_raw = inputs_loc_final(:, 1);  % Net B 的输入 (33维)
sample_C_raw = inputs_net_c(:, 1);      % Net C 的输入 (10维)

% 3. 维度安全检查 (防止 bsxfun 报错)
if size(sample_B_raw, 1) ~= ps_in.xrows
    error('Net B 维度不匹配！模型需要 %d 维，数据是 %d 维。请检查 inputs_loc_final。', ...
          ps_in.xrows, size(sample_B_raw, 1));
end
if size(sample_C_raw, 1) ~= ps_in_c.xrows
    error('Net C 维度不匹配！模型需要 %d 维，数据是 %d 维。请检查 inputs_net_c。', ...
          ps_in_c.xrows, size(sample_C_raw, 1));
end

% 4. 预热 (Warm-up) - 激活 JIT 编译
try
    a = net_loc(mapstd('apply', sample_B_raw, ps_in));
    b = net_shape(mapstd('apply', sample_C_raw, ps_in_c));
catch
    disp('预热跳过...');
end

% 5. 开始循环计时
N_loops = 1000;
tic;
for k = 1:N_loops
    % --- Net B 推理 ---
    in_b = mapstd('apply', sample_B_raw, ps_in);
    out_b = net_loc(in_b);
    % (此处省略反归一化和数据传递的微小开销，仅测核心计算)
    
    % --- Net C 推理 ---
    in_c = mapstd('apply', sample_C_raw, ps_in_c);
    out_c = net_shape(in_c);
    final_shape = mapstd('reverse', out_c, ps_out_c);
end
total_time = toc;

% 6. 计算结果
latency_ms = (total_time / N_loops) * 1000; % 平均单次耗时 (ms)
fps = 1000 / latency_ms;                    % 帧率 (Hz)

fprintf('   > 循环次数: %d 次\n', N_loops);
fprintf('   > 总耗时:   %.4f 秒\n', total_time);
fprintf('   > 单帧延迟: %.3f ms\n', latency_ms);
fprintf('   > 实时帧率: %.0f FPS\n', fps);
disp('--------------------------------------------------');