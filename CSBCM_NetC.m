%% ========================================================================
%  最小限度修正版：使用真实样本测试对比 物理模型 与 Net C
% ========================================================================

% 1. 直接从 Net C 的测试集里取第 1 个样本
k = 1;

% 【老老实实提取你指定的三个值】
% ① 碰撞后六轴拉力 (第 1-6 行，转置成 1x6 行向量以对齐 main.m 的 F)
F_after_real = in_test(1:6, k)'; 

% ② 受力向量 (第 7-9 行，3x1 列向量)
F_ex_real = in_test(7:9, k);     

% ③ 受力位置 (第 10 行，归一化值 * 9 还原成节数)
node_real = round(in_test(10, k) * 9.0); 

% 将节数映射到 main.m 中的 touch_id (假设 2段 * 每段7盘 = 14个受力盘)
touch_id = round((node_real / 9.0) * 14);
touch_id = max(1, min(14, touch_id));

fprintf('>>> 提取成功！外力=[%.2f, %.2f, %.2f], 位置=第%d节, 拉力=[%.2f, %.2f, %.2f, %.2f, %.2f, %.2f]\n', ...
        F_ex_real, node_real, F_after_real);

% 2. 把真实的拉力、外力、位置直接喂给你 main.m 的物理求解器
[Point_phys, ~, ~, ~, ~, ~] = solve_continuum_shape(...
    3, 2, 0.0006, 1.016e+12, 0.09, 0.00, 7, ...
    linspace(0.0025, 0.0025, 15), 0.25, 0, 4.000*0.00981, ...
    F_ex_real, F_after_real, touch_id);

% 从物理模型的轨迹中均匀提取 7 个点对齐你的传感器
idx_7pts = round(linspace(1, size(Point_phys, 2), 7));
Pts_Phys = Point_phys(:, idx_7pts);

% 3. 获取 Net C 的真值和预测值 (严格是 21x1 变 3x7，不再报错)
Pts_GT   = reshape(target_test(:, k), 3, 7);
Pts_NetC = reshape(pred_test(:, k), 3, 7);

% 4. 画图对比
figure('Name', 'Direct Compare: Physics Model vs Net C', 'Color', 'w');
% 加上基座原点 [0;0;0] 将线连起来
plot3([0, Pts_GT(1,:)],  [0, Pts_GT(2,:)],[0, Pts_GT(3,:)],  'k-o', 'LineWidth', 2, 'MarkerFaceColor','k'); hold on;
plot3([0, Pts_Phys(1,:)],[0, Pts_Phys(2,:)],[0, Pts_Phys(3,:)],'b--s', 'LineWidth', 1.5);
plot3([0, Pts_NetC(1,:)],[0, Pts_NetC(2,:)],[0, Pts_NetC(3,:)],'r--^', 'LineWidth', 1.5);

grid on; axis equal; view(30, 20);
xlabel('X (m)'); ylabel('Y (m)'); zlabel('Z (m)');
legend('Ground Truth', 'Physics Solver (main.m)', 'Net C', 'Location', 'best');
title(sprintf('Sample %d Comparison', k));