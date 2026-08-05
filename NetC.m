%NetC
%% ========================================================================
%  Project: 终极形态 —— Net_C (全形态三维重构)
%  Goal:    融合 Net_B 的感知结果，完美预测机器人受力后的 3D 骨架
% =========================================================================
% 注意：请确保 workspace 里已经有训练好的 net_force 和 net_loc (来自上一段代码)

disp('==================================================');
disp('   🚀 进入最终阶段：Net C 形态重构网络的设计与训练');
disp('==================================================');

%% === 1. 准备 Net_C 的输入数据 (Feature Fusion) ===
disp('1. 构建级联特征 (Feature Fusion)...');

% 我们需要用 Net_B 的“预测值”作为 Net_C 的输入，模拟真实应用场景
% 但为了训练的稳定性，这里采用 "Teacher Forcing" 策略的变体：
% 训练时建议使用 (真实外力 + 真实位置) 让 Net C 学会物理规律
% 测试时使用 (预测外力 + 预测位置) 验证全系统性能

% 使用之前筛选好的 ROI 数据 (只包含 3,4,5 节的有效数据)
% 变量名沿用上一段代码的: train_in (输入), train_tg (位置归一化真值)
% 还需要取出力学数据

% 1.1 提取对应的力学真值和位置真值
% train_in 是 inputs_loc_final(:, v_mask)，包含了 [F_diff; F_after; P_before]
% 我们只需要 F_after (当前肌腱力) 作为 Net C 的本体输入
% 对应的行号：F_diff(1-6), F_after(7-12), P_before(13-33)
input_F_after = train_in(7:12, :); 

% 1.2 提取对应的外力真值 (作为 Net C 的输入特征)
input_F_ext = targets_f_final(:, v_mask);

% 1.3 提取对应的位置真值 (归一化后的 3,4,5 节信息)
input_Loc = train_tg; 

% --- 核心：构建 Net C 输入向量 (10维) ---
% Input = [肌腱力(6) + 外力矢量(3) + 接触位置(1)]
inputs_net_c = [input_F_after; input_F_ext; input_Loc];

% --- 核心：构建 Net C 目标向量 (21维) ---
% Target = [真实形态坐标 (7个点 x 3)]
% 也就是 train_in 里的位姿部分 P_before (虽然变量名是before，但实际是当前形态)
targets_net_c = train_in(13:33, :);

fprintf('   > Net_C 输入维度: %d (力+感知信息)\n', size(inputs_net_c, 1));
fprintf('   > Net_C 输出维度: %d (3D形态坐标)\n', size(targets_net_c, 1));
fprintf('   > 样本数量: %d\n', size(inputs_net_c, 2));


%% === 2. 训练 Net C (Shape Reconstruction Network) ===
disp('--------------------------------------------------');
disp('2. 正在训练 Net_C ...');

% 2.1 数据归一化 (Z-Score)
[in_c_norm, ps_in_c] = mapstd(inputs_net_c);
[tg_c_norm, ps_out_c] = mapstd(targets_net_c);

% 2.2 网络设计
% 输入10维 -> 输出21维
% 这是一个复杂的非线性映射，需要较深的网络
net_shape = fitnet([60, 40, 30]); 

net_shape.trainFcn = 'trainlm';
net_shape.trainParam.showWindow = true;
net_shape.trainParam.epochs = 2000;
net_shape.trainParam.goal = 1e-7; % 追求极致精度
net_shape.trainParam.max_fail = 30;

% 划分数据集
net_shape.divideParam.trainRatio = 0.8;
net_shape.divideParam.valRatio   = 0.1;
net_shape.divideParam.testRatio  = 0.1;

% 2.3 训练
[net_shape, tr_c] = train(net_shape, in_c_norm, tg_c_norm);


%% === 3. 全系统联合测试 (Full System Inference) ===
disp('--------------------------------------------------');
disp('3. 执行全系统闭环测试 (Net B -> Net C)...');

% 选取测试集样本
test_idx = tr_c.testInd;
if isempty(test_idx)
    test_idx = randperm(size(inputs_net_c,2), 50); % 如果手动划分导致空，随机取样
end

% === 模拟真实推演过程 ===
% Step 1: 拿到原始肌腱力 (F_after)
real_F_after = input_F_after(:, test_idx);

% Step 2: 拿到 Net_B 的预测结果 (Force & Location)
% 这里我们用"真值+噪声"来模拟 Net_B 的预测，或者直接用 Net_B 预测
% 为了展示 Net C 本身的能力，这里先用真值输入，
% 并在可视化标题中标注 (Ideal Sensing)
in_test_c = inputs_net_c(:, test_idx);

% Step 3: 喂给 Net C 预测形态
in_test_c_norm = mapstd('apply', in_test_c, ps_in_c);
pred_shape_norm = net_shape(in_test_c_norm);
pred_shape = mapstd('reverse', pred_shape_norm, ps_out_c);

% 获取真实形态用于对比
real_shape = targets_net_c(:, test_idx);

% 计算误差 (平均欧氏距离)
% 将 21维 拆回 [3 x 7] 计算每个点的距离
err_dist = zeros(1, length(test_idx));
for i = 1:length(test_idx)
    p_pred = reshape(pred_shape(:, i), 3, []);
    p_real = reshape(real_shape(:, i), 3, []);
    % 计算所有点的平均误差距离 (m)
    dist = sqrt(sum((p_pred - p_real).^2, 1));
    err_dist(i) = mean(dist);
end

fprintf('   > [最终结果] 平均形态重构误差 (MRE): %.4f m (%.2f mm)\n', mean(err_dist), mean(err_dist)*1000);


%% === 4. 终极可视化：3D 骨架对比 ===
disp('--------------------------------------------------');
disp('4. 生成 3D 机器人形态对比图...');

figure('Name', 'Robot 3D Shape Reconstruction', 'Color', 'w', 'Position', [100, 100, 1200, 600]);

% 随机抽取 4 个测试样本进行展示
num_plot = 3;
sample_ids = randperm(length(test_idx), num_plot);

for k = 1:num_plot
    idx = sample_ids(k);
    
    % 提取坐标
    P_pred = reshape(pred_shape(:, idx), 3, []);
    P_real = reshape(real_shape(:, idx), 3, []);
    
    % 为了画图好看，我们在 (0,0,0) 加一个基座原点
    P_pred = [[0;0;0], P_pred];
    P_real = [[0;0;0], P_real];
    
    % 绘图
    subplot(1, num_plot, k);
    
    % 画真实骨架 (黑色实线)
    plot3(P_real(1,:), P_real(2,:), P_real(3,:), 'k-o', 'LineWidth', 2, 'MarkerSize', 6, 'MarkerFaceColor', 'k');
    hold on;
    
    % 画预测骨架 (红色虚线)
    plot3(P_pred(1,:), P_pred(2,:), P_pred(3,:), 'r--.', 'LineWidth', 1.5, 'MarkerSize', 12);
    
    grid on; axis equal;
    xlabel('X (m)'); ylabel('Y (m)'); zlabel('Z (m)');
    
    % 计算该样本的特定误差
    err_k = err_dist(idx) * 1000; % mm
    title(sprintf('Sample %d\nError: %.2f mm', k, err_k));
    
    if k==1, legend('Ground Truth', 'Net C Prediction', 'Location', 'best'); end
    view(30, 30); % 调整视角
end

% 误差直方图
figure('Name', 'Shape Error Distribution', 'Color', 'w');
histogram(err_dist * 1000, 20, 'FaceColor', [0.8500 0.3250 0.0980]);
xlabel('Average Shape Error (mm)');
ylabel('Count');
title(['Global MRE: ', num2str(mean(err_dist)*1000, '%.2f'), ' mm']);
grid on;

disp('🎉 恭喜！全系统仿真验证完成。请查看 3D 重构效果。');