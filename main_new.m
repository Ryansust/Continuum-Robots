%% 连续体机器人形变求解主程序
% 功能：计算给定驱动力和外力下的机器人末端及主干形变
clc; 
clear; 
close all;

%% 1. 机器人基础参数设置
tendon = 3;         % 每段驱动线数量
section = 2;        % 段数
D = 0.0006;         % 主梁直径 (m)
E = 0.516e+12;      % 弹性模量
L_a = 0.09;         % 单段长度 (m)
L_b = 0.00;         % 末端刚性段长度 (m)
N_d = 7;           % 每段的基础支撑盘数量
H_list = linspace(0.0025, 0.0025, section*N_d+1); % 过线孔分布半径
mu = 0.25;          % 摩擦系数
delta_alpha = 0 * pi / 180; % 扭转角偏差

%% 2. 受力参数设置
G = 4.000 * 0.00981;        % 重力载荷
F = [1, 0, 0, 0, 0, 0];     % 驱动线拉力分布 (x轴方向为1，顺时针递增)

F_ex = [0.1; 0; 0];       % 外力大小及方向[Fx; Fy; Fz]
% =========================================================
% 定义受力点位置 (touch_id)
% 若 section=2, N_d=15，则机器人总共有 30 个有效支撑盘。
% touch_id = 30 代表受力在第二个弯曲段末端
% touch_id = 15 代表受力在第一个弯曲段末端
% touch_id = 31 代表受力在刚性段的最末端 (默认)
% =========================================================
touch_id = 10;              % 外力作用的圆盘编号

%% 3. 调用核心求解器计算位姿
fprintf('正在计算机器人形变，请稍候...\n');[Point, Q, R_mat, N_step, L_total, P_ex_idx] = solve_continuum_shape_nofig(tendon, section, D, E, L_a, L_b, N_d, H_list, mu, delta_alpha, G, F_ex, F, touch_id);
fprintf('计算完成！\n');

%% 3.5 计算 Marker 的偏置坐标 (新增部分)
offset_distance = 0.004; % 4mm = 0.004m
V_local =[0; -offset_distance; 0]; % 局部坐标系下沿 y 轴负方向

% 初始化 Marker 整体轨迹坐标矩阵
Point_marker = zeros(size(Point));

% 遍历模型计算出的每一个离散点，将局部偏置转换到全局坐标系
for i = 1 : size(Point, 2)
    Point_marker(:, i) = Point(:, i) + R_mat(:, :, i) * V_local;
end

% ========================================================
% 提取真实物理圆盘对应的 Marker 坐标用于跟 OptiTrack/NDI 真值比较
% 假设你实验中 marker 贴在第 5, 10, 15, 30 个圆盘上 (id_list)
% ========================================================
id_list =[2,4,6,8,10,12]; 
kappa = 4; % 求解器内部固定的离散分辨率倍数，必须为 4
marker_indices = id_list * kappa + 1; % 计算这些盘在 Point 数组中的对应列索引

% 这就是你最终拿去求 error 的模型预测 Marker 坐标！
Model_Marker_Coords = Point_marker(:, marker_indices); 

% 打印出来看看
disp('对应实验圆盘的模型 Marker 预测坐标 (x, y, z):');
disp(Model_Marker_Coords);

%% 4. 可视化绘图 - 最终状态精细图
color = zeros(3, section+1);
for i = 1 : section+1
    color(:, i) = ones(3,1) * (0.6/(section) * (i - 1) + 0.2);
end

figure('Name', 'Final State: Continuum Robot Shape', 'Color', 'w');
% 绘制主梁
for i = 1 : section
    idx_start = (i-1)*N_step + 1;
    idx_end = i*N_step + 1;
    plot3(Point(1,idx_start:idx_end), Point(2,idx_start:idx_end), Point(3,idx_start:idx_end), ...
          '-*', 'linewidth', 3, 'color', color(:, i));  
    hold on;
end
plot3(Point(1,end-1:end), Point(2,end-1:end), Point(3,end-1:end), '-*', 'linewidth', 3, 'color', color(:, end));

% 绘制拉线
for i = 1 : section * tendon
    num = mod(i, section); 
    if num == 0, num = section; end
    plot3(Q(1,1:num*N_step+1,i)*L_total, Q(2,1:num*N_step+1,i)*L_total, Q(3,1:num*N_step+1,i)*L_total, 'k-');
end

% 绘制支撑盘
for j = 1 : section
    for i = 1 : N_step
        num = (j-1)*N_step + i;
        for k = 1 : section * tendon 
            for t = k + 1 : section * tendon
                plot3([Q(1,num,k), Q(1,num,t)]*L_total,[Q(2,num,k), Q(2,num,t)]*L_total,[Q(3,num,k), Q(3,num,t)]*L_total, ...
                      '-', "Color", color(:, j), 'linewidth', 1); 
            end
        end
    end
end
for k = 1 : section * tendon 
    for t = k + 1 : section * tendon
        plot3([Q(1,end,k), Q(1,end,t)]*L_total,[Q(2,end,k), Q(2,end,t)]*L_total,[Q(3,end,k), Q(3,end,t)]*L_total, ...
              '-', "Color", color(:, end), 'linewidth', 1); 
    end
end

% ==================== 新增：绘制 Marker 及偏置连线 ====================
% 1. 画出 Marker 的整体轨迹虚线
plot3(Point_marker(1,:), Point_marker(2,:), Point_marker(3,:), 'm--', 'LineWidth', 1.5);

% 2. 用品红色实心圆点标出提取的实际物理圆盘上的 Marker
scatter3(Model_Marker_Coords(1,:), Model_Marker_Coords(2,:), Model_Marker_Coords(3,:), ...
    60, 'm', 'filled');

% 3. 画灰色的细实线，连接主干中心点和 Marker 偏置点，直观展示 4mm 的偏置方向
for idx = marker_indices
    plot3([Point(1,idx), Point_marker(1,idx)], ...
        [Point(2,idx), Point_marker(2,idx)], ...
        [Point(3,idx), Point_marker(3,idx)], '-', 'Color', [0.5 0.5 0.5], 'LineWidth', 1.5);
end
% ======================================================================

% ==== 绘制外力作用箭头 ====
if norm(F_ex) > 0
    P_force = Point(:, P_ex_idx);
    F_dir = F_ex / norm(F_ex) * 0.03; % 箭头显示长度(按比例缩放)
    quiver3(P_force(1), P_force(2), P_force(3), F_dir(1), F_dir(2), F_dir(3), ...
            0, 'm', 'LineWidth', 3, 'MaxHeadSize', 0.5);
end

% 绘制末端坐标系
quiver3(Point(1,end), Point(2,end), Point(3,end), R_mat(1,1,end), R_mat(2,1,end), R_mat(3,1,end), 0.02, 'r', 'LineWidth', 2); 
quiver3(Point(1,end), Point(2,end), Point(3,end), R_mat(1,2,end), R_mat(2,2,end), R_mat(3,2,end), 0.02, 'g', 'LineWidth', 2); 
quiver3(Point(1,end), Point(2,end), Point(3,end), R_mat(1,3,end), R_mat(2,3,end), R_mat(3,3,end), 0.02, 'b', 'LineWidth', 2); 

% 绘制基坐标系
quiver3(0, 0, 0, 1, 0, 0, 0.03, 'r', 'LineWidth', 2); 
quiver3(0, 0, 0, 0, 1, 0, 0.03, 'g', 'LineWidth', 2); 
quiver3(0, 0, 0, 0, 0, 1, 0.03, 'b', 'LineWidth', 2); 

% 视图设置
xlabel('X (m)'), ylabel('Y (m)'), zlabel('Z (m)');
title(sprintf('Final Shape (External Force applied at Disc %d)', touch_id));
axis equal, view(30,20), grid on;
set(gca,'zdir','reverse'); set(gca,'ydir','reverse');