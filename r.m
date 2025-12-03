%% ========================================================================
%  PART B: Net_B_Height (基于旋转增强的分类器)
% =========================================================================
clc; clear; close all;
rng('default');

% === 1. 数据读取与基础预处理 ===
FileName = '/Users/ryan/Desktop/continuum robot/force_data/after_processing_data_0816.xlsx';
dataTable = readtable(FileName);

F_after = table2array(dataTable(3:end, 23:28))'; 
F_before = table2array(dataTable(3:end, 11:16))'; 
F_diff = F_after - F_before;

raw_mag = abs(table2array(dataTable(3:end, 2)))'; 
raw_hgt = table2array(dataTable(3:end, 4))'; 

% === 2. 筛选有效数据 (Step 1) ===
% 依然建议保留这个筛选，只用有力的数据训练
min_force_threshold = 0.1; 
valid_mask = raw_mag > min_force_threshold;

fprintf('数据筛选: 保留有效样本 %d 个 (原 %d 个)\n', sum(valid_mask), length(raw_mag));

% 提取基础特征
valid_F_after = F_after(:, valid_mask);
valid_F_diff  = F_diff(:, valid_mask);
valid_targets = raw_hgt(valid_mask);

% 计算归一化模式特征 (Pattern)
diff_norms = sqrt(sum(valid_F_diff.^2, 1));
valid_F_pattern = valid_F_diff ./ (diff_norms + 1e-8);

% 原始训练输入 (12维)
inputs_original = [valid_F_after; valid_F_pattern];
targets_original = valid_targets;

% === 3. 划分数据集 (Train/Test) ===
% 【重要】必须在增强前划分！确保测试集是真实的原始数据，没经过旋转伪造。
cv_h = cvpartition(length(targets_original), 'HoldOut', 0.2);

train_in_raw = inputs_original(:, cv_h.training);
train_tg_raw = targets_original(:, cv_h.training);

test_in  = inputs_original(:, cv_h.test); % 纯净测试集
test_tg  = targets_original(:, cv_h.test);

% === 4. 旋转数据增强 (Step 2 - 核心修改) ===
disp('>>> 正在执行 120度/240度 旋转增强...');

% --- 定义轮换索引 (Permutation Indices) ---
% 假设结构：[1,3,5] 互为120度; [2,4,6] 互为120度
% 旋转 120 度: 1->3,2->4,3->5
base_perm_120=[5,6,1,2,3,4];
idx_120 = [base_perm_120,base_perm_120+6];
% 旋转 240 度: (即在120的基础上再转一次)
base_perm_240 =[3,4,5,6,1,2];
idx_240 = [base_perm_240,base_perm_240+6];

% 1. 生成 120度 数据
train_in_120 = train_in_raw(idx_120, :); % 重新排列行顺序
% 2. 生成 240度 数据
train_in_240 = train_in_raw(idx_240, :);

% 3. 合并数据 (3倍数据量)
train_in_aug = [train_in_raw, train_in_120, train_in_240];
train_tg_aug = [train_tg_raw, train_tg_raw, train_tg_raw]; % 高度标签不变

fprintf('   原始训练集: %d\n', length(train_tg_raw));
fprintf('   增强后训练集: %d (3倍)\n', length(train_tg_aug));

% === 5. 网络训练 ===
% 转换 One-Hot
train_tg_onehot = full(ind2vec(train_tg_aug));

% 网络架构：因为数据全是真实的物理分布，网络可以稍微大一点点
net_height = patternnet([30, 20]); 

net_height.trainFcn = 'trainscg'; 
net_height.performFcn = 'crossentropy';
net_height.performParam.regularization = 0.05; % 保持正则化防止过拟合
net_height.trainParam.epochs = 1000;
net_height.trainParam.showWindow = false;
net_height.trainParam.max_fail = 20;

% 训练
[net_height, tr_h] = train(net_height, train_in_aug, train_tg_onehot);

% === 6. 评估 (Evaluation) ===
pred_scores = net_height(test_in);
[~, pred_classes] = max(pred_scores);

% 准确率
acc = sum(pred_classes == test_tg) / length(test_tg);
acc_tol = sum(abs(pred_classes - test_tg) <= 1) / length(test_tg);

fprintf('\n>>> [Net_B_Height 旋转增强版] 结果:\n');
fprintf('   严格准确率: %.2f%%\n', acc * 100);
fprintf('   容忍±1误差准确率: %.2f%%\n', acc_tol * 100);

% 混淆矩阵
figure('Name', 'Height Confusion Matrix (Rotational Aug)', 'Color', 'w');
confusionchart(test_tg, pred_classes);
title(sprintf('高度分类 (旋转增强, Acc: %.1f%%)', acc*100));
%% %% === 4. 混合数据增强 (Hybrid Augmentation: Rotation + Noise) ===
% 策略：先做物理旋转，再对旋转后的所有数据加噪声
disp('>>> 正在执行 [混合增强]：物理旋转(3x) + 高斯噪声(5x)...');

% --- A. 第一步：物理旋转增强 (3倍) ---
base_perm_120 = [3, 4, 5, 6, 1, 2];
base_perm_240 = [5, 6, 1, 2, 3, 4];
idx_120_full = [base_perm_120, base_perm_120 + 6];
idx_240_full = [base_perm_240, base_perm_240 + 6];

% 生成旋转数据
train_in_rot120 = train_in_raw(idx_120_full, :);
train_in_rot240 = train_in_raw(idx_240_full, :);

% 合并旋转数据 (此时是 3倍)
train_in_phys = [train_in_raw, train_in_rot120, train_in_rot240];
train_tg_phys = [train_tg_raw, train_tg_raw, train_tg_raw];

% --- B. 第二步：高斯噪声增强 (对上述3倍数据，再扩充N倍) ---
noise_times = 4;     % 每个物理样本生成 4 个噪声样本
noise_level = 0.015; % 1.5% 的噪声 (不要太大，否则会模糊特征)

% 初始化最终容器
train_in_final = train_in_phys;
train_tg_final = train_tg_phys;

for k = 1:noise_times
    % 生成噪声
    noise = noise_level * randn(size(train_in_phys));
    
    % 注入噪声
    train_in_noisy = train_in_phys + noise;
    
    % 拼接到最终集合
    train_in_final = [train_in_final, train_in_noisy];
    train_tg_final = [train_tg_final, train_tg_phys];
end

% 计算最终倍率
total_multiplier = 3 * (1 + noise_times);
fprintf('   原始训练集: %d\n', length(train_tg_raw));
fprintf('   物理旋转后: %d (3倍)\n', length(train_tg_phys));
fprintf('   混合增强后: %d (%d倍)\n', length(train_tg_final), total_multiplier);

%% === 5. 网络训练 (混合增强版) ===
train_tg_onehot = full(ind2vec(train_tg_final));

% 网络可以稍微加深一点点，因为数据量大了，不容易过拟合
net_height = patternnet([40, 25]); 

net_height.trainFcn = 'trainscg'; 
net_height.performFcn = 'crossentropy';
net_height.performParam.regularization = 0.02; % 数据量大了，正则化系数可以稍微减小
net_height.trainParam.epochs = 1500;
net_height.trainParam.showWindow = false;
net_height.trainParam.max_fail = 30; % 允许更多次尝试

% 训练
[net_height, tr_h] = train(net_height, train_in_final, train_tg_onehot);

%% === 6. 评估 (依然使用纯净的测试集) ===
pred_scores = net_height(test_in);
[~, pred_classes] = max(pred_scores);

acc = sum(pred_classes == test_tg) / length(test_tg);
acc_tol = sum(abs(pred_classes - test_tg) <= 1) / length(test_tg);

fprintf('\n>>> [Net_B_Height 混合增强结果]\n');
fprintf('   严格准确率: %.2f%%\n', acc * 100);
fprintf('   容忍±1误差准确率: %.2f%%\n', acc_tol * 100);

figure('Name', 'Height Confusion Matrix (Hybrid)', 'Color', 'w');
confusionchart(test_tg, pred_classes);
title(sprintf('混合增强结果 (Acc: %.1f%%)', acc*100));
%% %% === 尝试 SVM 分类器 ===
disp('>>> 正在训练 SVM 分类器...');

% SVM 需要输入是 [N x Features]，所以要转置
X_train = train_in_final'; 
Y_train = vec2ind(train_tg_onehot)'; % 转回 1-9 的标签

X_test = test_in';
Y_test = test_tg';

% 训练多分类 SVM (使用高斯核 KernelFunction: 'gaussian' 或 'rbf')
% OptimizeHyperparameters: 'auto' 会自动帮你找最好的参数，但比较慢
t_svm = templateSVM('KernelFunction', 'gaussian', 'Standardize', true);
model_svm = fitcecoc(X_train, Y_train, 'Learners', t_svm);

% 预测
pred_svm = predict(model_svm, X_test);

% 评估
acc_svm = sum(pred_svm' == Y_test') / length(Y_test);
fprintf('\n>>> [SVM 结果] 严格准确率: %.2f%%\n', acc_svm * 100);
figure;
confusionchart(Y_test, pred_svm);
title(sprintf('混合增强结果 (Acc: %.1f%%)', acc_svm*100));