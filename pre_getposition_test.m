clc; clear; close all;

%% 1. 数据录入 (根据图片手动提取)
% 格式：每一组是一个结构体，包含 F_old (1x6) 和 F_new (Nx6)
% F_old: 对应图片中的 'real' 行 (测量的力，即中间那组f1-f6)
% F_new: 对应图片中的 1, 2, 3, 4 行

data_groups = {};

% --- 第1组数据 (主驱动 f2) ---
group1.name = 'Group 1 (Main: f2)';
group1.F_old = [-7, -268, -6, -9, -10, -15];
group1.F_new = [
    -15, -269, -4, -13, -8, -21;
    -14, -268, -5, -12, -7, -21;
    -15, -268, -5, -12, -8, -21;
    -14, -267, -4, -12, -8, -21
];
data_groups{end+1} = group1;

% --- 第2组数据 (主驱动 f1) ---
group2.name = 'Group 2 (Main: f1)';
group2.F_old = [-207.29, -9, -8, -9, -10, -7];
group2.F_new = [
    -207.29, -15, -7, -9, -14, -7;
    -207.29, -15, -6, -9, -13, -7;
    -207.29, -15, -6, -10, -13, -5
];
data_groups{end+1} = group2;

% --- 第3组数据 (主驱动 f2, f5) ---
group3.name = 'Group 3 (Main: f2, f5)';
group3.F_old = [-11, -137, -6, -13, -509, -12];
group3.F_new = [
    -15, -136, -6, -12, -507, -20;
    -16, -138, -6, -13, -505, -20
];
data_groups{end+1} = group3;

% --- 第4组数据 (主驱动 f1, f2) ---
group4.name = 'Group 4 (Main: f1, f2)';
group4.F_old = [-99, -481, -6, -12, -7, -16];
group4.F_new = [
    -99, -479, -5, -16, -13, -27;
    -98, -486, -4, -17, -12, -26
];
data_groups{end+1} = group4;

%% 2. 批量计算指标
fprintf('================ 数据一致性分析报告 ================\n');

for k = 1:length(data_groups)
    g = data_groups{k};
    F_old = g.F_old;
    F_new_mean = mean(g.F_new, 1); % 取新数据的平均值来对比
    
    fprintf('\n%s:\n', g.name);
    
    % 判定主动力阈值 (绝对值大于50N视为主驱动力)
    active_threshold = 50; 
    active_idx = abs(F_old) > active_threshold;
    passive_idx = ~active_idx;
    
    % 获取最大驱动力 (作为漂移参考分母)
    max_active_force = max(abs(F_old));
    
    % --- 指标 1: 主动力复现误差 (Relative Error) ---
    % 公式: |F_new - F_old| / |F_old|
    if sum(active_idx) > 0
        active_diff = abs(F_new_mean(active_idx) - F_old(active_idx));
        active_base = abs(F_old(active_idx));
        active_error_pct = mean(active_diff ./ active_base) * 100;
        fprintf('  1. 主动力复现误差: %.2f%%\n', active_error_pct);
    else
        fprintf('  1. 主动力复现误差: N/A (无大力)\n');
    end
    
    % --- 指标 2: 被动力漂移占比 (Passive Drift Ratio) ---
    % 公式: |F_new_pas - F_old_pas| / Max_Active_Force
    if sum(passive_idx) > 0
        passive_diff = abs(F_new_mean(passive_idx) - F_old(passive_idx));
        drift_ratio_pct = mean(passive_diff) / max_active_force * 100;
        fprintf('  2. 被动力漂移占比: %.2f%%\n', drift_ratio_pct);
    else
        fprintf('  2. 被动力漂移占比: 0%%\n');
    end
    
    % --- 指标 3: 余弦相似度 (Cosine Similarity) ---
    % 公式: (A . B) / (|A| * |B|)
    v1 = F_old;
    v2 = F_new_mean;
    similarity = dot(v1, v2) / (norm(v1) * norm(v2));
    fprintf('  3. 整体余弦相似度: %.5f\n', similarity);
    
    % --- 简单评判 ---
    if active_error_pct < 5 && similarity > 0.99
        disp('  >> [结论]: 系统状态极佳 (Excellent)');
    elseif active_error_pct < 10
        disp('  >> [结论]: 状态良好，可接受 (Good)');
    else
        disp('  >> [结论]: 偏差较大，建议检查 (Check)');
    end
end
fprintf('\n====================================================\n');