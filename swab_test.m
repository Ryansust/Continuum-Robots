clear;clc;close all;
%% import data
SensorValue1=readmatrix('swab_test.xlsx');

time_s1=SensorValue1(:,11)-SensorValue1(1,11); 
time_s1=time_s1(~isnan(time_s1));
%Here we give an example of 3-point optoelectronic sensors values

s1=SensorValue1(:,12); s1=s1(~isnan(s1));
s2=SensorValue1(:,13); s2=s2(~isnan(s2));
s3=SensorValue1(:,14); s3=s3(~isnan(s3));
% s4=SensorValue1(:,15); s4=s4(~isnan(s4));
% s5=SensorValue1(:,16); s5=s5(~isnan(s5));
% L_1_ATI_all=length(time_s11);
L_1_Mega_all=length(time_s1);

Fx=SensorValue1(:,1);Fx=Fx(~isnan(Fx));
Fy=SensorValue1(:,2);Fy=Fy(~isnan(Fy));
Fz=SensorValue1(:,3);Fz=Fz(~isnan(Fz));

[s1, PS1] = mapminmax(s1');
s1=s1';
[s2, PS2] = mapminmax(s2');
s2=s2';
[s3, PS3] = mapminmax(s3');
s3=s3';
% [s4, PS4] = mapminmax(s4',-1,1);
% s4=s4';
% [s5, PS5] = mapminmax(s5',-1,1);
% s5=s5';

Sensor_all1=[s1 s2 s3];

[Fx, PSx] = mapminmax(Fx');
Fx = Fx';
[Fy, PSy] = mapminmax(Fy');
Fy = Fy';
[Fz, PSz] = mapminmax(Fz');
Fz = Fz';
F = [Fx Fy Fz];

train_number = floor(0.6665 * size(F, 1));
test_number = 1*size(F, 1) - train_number;
xTrain  = Sensor_all1(1:train_number,:);
yTrain_x = Fx(1:train_number,:);
yTrain_y = Fy(1:train_number,:);
yTrain_z = Fz(1:train_number,:);
xTest = Sensor_all1(train_number+1 : train_number+test_number,:);
yTest_x = Fx(train_number+1 : train_number+test_number, :);
yTest_y = Fy(train_number+1 : train_number+test_number, :);
yTest_z = Fz(train_number+1 : train_number+test_number, :);

%%
% Features and HiddenUnits
numFeatures = 3;
numHiddenUnits = 20;
numHiddenUnits1 = 100;
numHiddenUnits2 = 10;
numHiddenUnits3 = 50;

numResponses = 1;

layers = [sequenceInputLayer(numFeatures)
%           fullyConnectedLayer(numHiddenUnits1)
%           lstmLayer(numHiddenUnits)
          bilstmLayer(numHiddenUnits2)
%           gruLayer(numHiddenUnits2)
          fullyConnectedLayer(numResponses)
          regressionLayer];

miniBatchSize = 64;
options = trainingOptions('sgdm', ...
                          'ExecutionEnvironment', 'cpu', ...
                          'MaxEpochs', 100, ...
                          'MiniBatchSize', miniBatchSize, ...
                          'GradientThreshold', 1, ...
                          'InitialLearnRate', 0.01, ...
                          'LearnRateSchedule', 'piecewise', ...
                          'LearnRateDropPeriod', 250, ...
                          'LearnRateDropFactor', 0.1, ...
                          'Verbose', false, ...
                          'Plots', 'training-progress');

% Training
net_x = trainNetwork(xTrain', yTrain_x', layers, options);
net_y = trainNetwork(xTrain', yTrain_y', layers, options);
net_z = trainNetwork(xTrain', yTrain_z', layers, options);

% Performance
yTrain_pre_x = predict(net_x, xTrain', 'MiniBatchSize', miniBatchSize, 'SequenceLength', 'longest');
yTrain_pre_y = predict(net_y, xTrain', 'MiniBatchSize', miniBatchSize, 'SequenceLength', 'longest');
yTrain_pre_z = predict(net_z, xTrain', 'MiniBatchSize', miniBatchSize, 'SequenceLength', 'longest');

% Mapping
yTrain_pre_x = mapminmax('reverse', yTrain_pre_x, PSx);
yTrain_pre_y = mapminmax('reverse', yTrain_pre_y, PSy);
yTrain_pre_z = mapminmax('reverse', yTrain_pre_z, PSz);
yTrain_pre_x = yTrain_pre_x';
yTrain_pre_y = yTrain_pre_y';
yTrain_pre_z = yTrain_pre_z';
yTrain_x = mapminmax('reverse', yTrain_x, PSx);
yTrain_y = mapminmax('reverse', yTrain_y, PSy);
yTrain_z = mapminmax('reverse', yTrain_z, PSz);

err_train_x = yTrain_pre_x - yTrain_x;
err_train_y = yTrain_pre_y - yTrain_y;
err_train_z = yTrain_pre_z - yTrain_z;

% RMSE
rmse_train_x = rms(err_train_x);
rmse_train_y = rms(err_train_y);
rmse_train_z = rms(err_train_z);
line_acc = 0.002 * ones(1, train_number);

%Performance
tic
yTest_pre_x = predict(net_x, xTest', 'MiniBatchSize', miniBatchSize, 'SequenceLength', 'longest');
yTest_pre_y = predict(net_y, xTest', 'MiniBatchSize', miniBatchSize, 'SequenceLength', 'longest');
yTest_pre_z = predict(net_z, xTest', 'MiniBatchSize', miniBatchSize, 'SequenceLength', 'longest');
toc

% Mapping
yTest_pre_x = mapminmax('reverse', yTest_pre_x, PSx);
yTest_pre_y = mapminmax('reverse', yTest_pre_y, PSy);
yTest_pre_z = mapminmax('reverse', yTest_pre_z, PSz);
yTest_pre_x = yTest_pre_x';
yTest_pre_y = yTest_pre_y';
yTest_pre_z = yTest_pre_z';
yTest_x = mapminmax('reverse', yTest_x, PSx);
yTest_y = mapminmax('reverse', yTest_y, PSy);
yTest_z = mapminmax('reverse', yTest_z, PSz);

err_test_x = yTest_pre_x - yTest_x;
err_test_y = yTest_pre_y - yTest_y;
err_test_z = yTest_pre_z - yTest_z;

% RMSE in Test
rmse_test_x = rms(err_test_x);
rmse_test_y = rms(err_test_y);
rmse_test_z = rms(err_test_z);
% line_acc = 0.002 * ones(1, train_number);

%%
mae_x = mae(yTest_pre_x-yTest_x);
mae_y = mae(yTest_pre_y-yTest_y);
mae_z = mae(yTest_pre_z-yTest_z);

%% Plot
figure('Position', [10, 10, 900, 400]);
subplot(3, 1, 1);
plot(yTrain_x(:,1), 'k');
hold on
plot(yTrain_pre_x(:,1), 'r');
hold on 
%  plot(3,p_x, 'c*');
% plot(p_x(:,1), 'b');
hold off
xlabel('序号');
ylabel('下浮率');
title('LSTM训练集拟合效果');
xlim([1, train_number]);

subplot(3, 1, 2);
plot(yTrain_y(:,1), 'k');
hold on
plot(yTrain_pre_y(:,1), 'r');
hold on 
%  plot(3,p_y, 'c*');
% plot(p_y(:,1), 'b');
hold off
xlabel('序号');
ylabel('下浮率');
title('LSTM训练集拟合效果');
xlim([1, train_number]);

subplot(3, 1, 3);
plot(yTrain_z(:,1), 'k');
hold on
plot(yTrain_pre_z(:,1), 'r');
hold on 
%  plot(3,p_z, 'c*');
% plot(p_z(:,1), 'b');
hold off
xlabel('序号');
ylabel('下浮率');
title('LSTM训练集拟合效果');
xlim([1, train_number]);

% Test
figure('Position', [10, 10, 900, 400]);
subplot(3, 1, 1);
plot(yTest_x(:,1), 'k');
hold on
plot(yTest_pre_x(:,1), 'r');
% hold on 
%  plot(3,p_x, 'c*');
% plot(p_x(:,1), 'b');
hold off
xlabel('序号');
ylabel('fx');
title('LSTM测试集拟合效果');
xlim([1, test_number]);

subplot(3, 1, 2);
plot(yTest_y(:,1), 'k');
hold on
plot(yTest_pre_y(:,1), 'r');
%hold on 
%  plot(3,p_y, 'c*');
% plot(p_y(:,1), 'b');
hold off
xlabel('序号');
ylabel('fy');
title('LSTM测试集拟合效果');
xlim([1, test_number]);

subplot(3, 1, 3);
plot(yTest_z(:,1), 'k');
hold on
plot(yTest_pre_z(:,1), 'r');
% hold on 
%  plot(3,p_z, 'c*');
% plot(p_z(:,1), 'b');
hold off
xlabel('序号');
ylabel('fz');
title('LSTM测试集拟合效果');
xlim([1, test_number]);