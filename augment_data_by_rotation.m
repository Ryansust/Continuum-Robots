function [aug_F_diff, aug_F_after, aug_P_before, aug_gt_F, aug_targets] = ...
    augment_data_by_rotation(F_diff, F_after, P_before, gt_F_vec, targets)
% AUGMENT_DATA_BY_ROTATION 利用120度对称性增强数据
%
% 输入 (假设样本数为 N):
%   F_diff   : [6 x N] 肌腱力差值
%   F_after  : [6 x N] 当前肌腱力
%   P_before : [27 x N] 形态位姿 (9个点 x 3坐标)
%   gt_F_vec : [3 x N] 外力矢量 (Fx, Fy, Fz)
%   targets  : [1 x N] 标签 (如接触节数或归一化位置)，旋转不改变接触位置
%
% 输出:
%   返回原始数据 + 120度旋转 + 240度旋转后的合并数据 (样本数变为 3*N)

    fprintf('>> 正在执行旋转对称性增强 (x3 数据量)...\n');
    
    %% 1. 准备工作
    N = size(F_diff, 2);
    
    % 定义旋转矩阵 (绕 Z 轴)
    theta1 = 120;
    theta2 = 240;
    
    Rz_120 = [cosd(theta1), -sind(theta1), 0;
              sind(theta1),  cosd(theta1), 0;
              0,             0,            1];
          
    Rz_240 = [cosd(theta2), -sind(theta2), 0;
              sind(theta2),  cosd(theta2), 0;
              0,             0,            1];

    %% 2. 处理力数据 (Permutation / 轮换)
    % 假设肌腱布局为 120度对称，顺序通常为 [T1, T2, T3, T4, T5, T6]
    % 旋转 120 度：原来的 T5,T6 位置变成了现在的 T1,T2 位置
    % 参考你的代码逻辑：[5,6, 1,2, 3,4]
    
    idx_120 = [5; 6; 1; 2; 3; 4];
    idx_240 = [3; 4; 5; 6; 1; 2];
    
    % 生成增强的力数据
    F_diff_120  = F_diff(idx_120, :);
    F_after_120 = F_after(idx_120, :);
    
    F_diff_240  = F_diff(idx_240, :);
    F_after_240 = F_after(idx_240, :);
    
    %% 3. 处理位姿数据 (Geometry Rotation)
    % P_before 是 [27 x N]。我们需要先把它变成 [3 x (9*N)] 以便批量旋转
    
    % 120 度旋转
    P_reshaped = reshape(P_before, 3, []); % 变成 [3 x 9N]
    P_rot_120_mat = Rz_120 * P_reshaped;
    P_before_120 = reshape(P_rot_120_mat, 27, N); % 变回 [27 x N]
    
    % 240 度旋转
    P_rot_240_mat = Rz_240 * P_reshaped;
    P_before_240 = reshape(P_rot_240_mat, 27, N);
    
    %% 4. 处理外力矢量 (Vector Rotation)
    % 外力矢量 [3 x N] 也是空间向量，需要随坐标系旋转
    gt_F_120 = Rz_120 * gt_F_vec;
    gt_F_240 = Rz_240 * gt_F_vec;
    
    %% 5. 处理标签 (Invariant)
    % 旋转机器人不会改变"接触点是在第几节"，所以标签直接复制
    targets_aug = targets; 
    
    %% 6. 合并数据
    aug_F_diff   = [F_diff,   F_diff_120,   F_diff_240];
    aug_F_after  = [F_after,  F_after_120,  F_after_240];
    aug_P_before = [P_before, P_before_120, P_before_240];
    aug_gt_F     = [gt_F_vec, gt_F_120,     gt_F_240];
    aug_targets  = [targets,  targets_aug,  targets_aug];
    
    fprintf('   > 增强完成。原始: %d -> 增强后: %d\n', N, size(aug_F_diff, 2));
end