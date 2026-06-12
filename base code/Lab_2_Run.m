clc; clear;
close all

%% Step 1: Data Preprocessing
% Load the Iris dataset
load fisheriris.mat;

% Feature matrix (X) and target vector (Y)
X = meas; % Features (sepal length, sepal width, petal length, petal width)
Y = grp2idx(species); % Convert species names to indices (1, 2, 3 for different species)

% Standardize the features (mean 0, variance 1)
X = zscore(X);

% Split data into training (80%) and test (20%) sets
cv = cvpartition(size(X, 1), 'HoldOut', 0.2);
X_train = X(training(cv), :);
Y_train = Y(training(cv));
X_test = X(test(cv), :);
Y_test = Y(test(cv));

Y_train = categorical(Y_train); 
Y_test = categorical(Y_test);   

Y_train_one_hot = onehotencode(Y_train,3); % One-hot encoding for Y_train
Y_train_one_hot = squeeze(Y_train_one_hot);
Y_test_one_hot = onehotencode(Y_test,3);   % One-hot encoding for Y_test
Y_test_one_hot = squeeze(Y_test_one_hot);

%% Step 2: Network Initialization
input_size = size(X, 2); % Number of input features (4)
hidden_size = 8; % Number of neurons in the hidden layer
output_size = 3; % Number of output classes (3 species)

% Random initialization of weights and biases
W1 = randn(input_size, hidden_size) * 0.01; % Weight matrix for first layer (4x8)
b1 = zeros(1, hidden_size); % Bias for first layer (1x8)

W2 = randn(hidden_size, output_size) * 0.01; % Weight matrix for second layer (8x3)
b2 = zeros(1, output_size); % Bias for second layer

%% Step 7: Training the Network and Plotting Loss

epochs = 2000; 
learning_rate = 0.1;
losses = zeros(epochs, 1); % Array to store the loss for each epoch

% Training loop
for epoch = 1:epochs
    % Perform gradient descent and update parameters
    [W1, b1, W2, b2, L] = gradient_descent(X_train, Y_train_one_hot, W1, b1, W2, b2, learning_rate);
    
    losses(epoch) = L;
    
    % display the loss every 100 epochs for monitoring
    if mod(epoch, 100) == 0
        fprintf('Epoch %d/%d, Loss: %.4f\n', epoch, epochs, L);
    end
end

figure;
plot(1:epochs, losses);
xlabel('Epoch');
ylabel('Loss');
title('Training Loss Curve');
grid on;

%% Step 8: Evaluation 

% --- EVALUATE TRAINING SET ---
[~, A2_train, ~,~] = forward(X_train, W1, b1, W2, b2);
[~, predIdx_train] = max(A2_train, [], 2);
[~, realIdx_train] = max(Y_train_one_hot, [], 2);

C_train = confusionmat(realIdx_train, predIdx_train);
accuracy_train = sum(diag(C_train)) / sum(C_train(:)) * 100;
fprintf('Final Training Set Accuracy: %.2f%%\n', accuracy_train);

figure;
confusionchart(C_train);
title(['Training Set Confusion Matrix (Accuracy: ', num2str(accuracy_train, '%.2f'), '%)']);

% --- EVALUATE TEST SET ---
[~, A2_test, ~,~] = forward(X_test, W1, b1, W2, b2);
[~, predIdx_test] = max(A2_test, [], 2);
[~, realIdx_test] = max(Y_test_one_hot, [], 2);

C_test = confusionmat(realIdx_test, predIdx_test);
accuracy_test = sum(diag(C_test)) / sum(C_test(:)) * 100;
fprintf('Final Test Set Accuracy: %.2f%%\n', accuracy_test);

figure;
confusionchart(C_test);
title(['Test Set Confusion Matrix (Accuracy: ', num2str(accuracy_test, '%.2f'), '%)']);


%% Step 9: Visualize 2-D Decision Boundaries 

feature_names = {'Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width'};
pairs = [1,2; 1,3; 1,4; 2,3; 2,4; 3,4]; 

figure('Name', '2-D Decision Boundaries', 'NumberTitle', 'off');

for p = 1:6
    f1 = pairs(p,1);
    f2 = pairs(p,2);
    
    subplot(2, 3, p);
    
    x1_range = min(X(:, f1))-0.5 : 0.05 : max(X(:, f1))+0.5;
    x2_range = min(X(:, f2))-0.5 : 0.05 : max(X(:, f2))+0.5;
    [X1_grid, X2_grid] = meshgrid(x1_range, x2_range);
    
    
    mock_X = zeros(numel(X1_grid), 4);
    mock_X(:, f1) = X1_grid(:);
    mock_X(:, f2) = X2_grid(:);
    
    [~, A2_grid, ~,~] = forward(mock_X, W1, b1, W2, b2);
    [~, grid_preds] = max(A2_grid, [], 2);
    GRID_PREDS = reshape(grid_preds, size(X1_grid));
    

    
    contourf(X1_grid, X2_grid, GRID_PREDS);
    hold on;
    
    % Overlay actual dataset observations onto the 2D plane
    gscatter(X(:, f1), X(:, f2), Y, 'rgb', 'osd', 6, 'off');
    
    xlabel(feature_names{f1});
    ylabel(feature_names{f2});
    title(sprintf('Pair %d: %s vs %s', p, feature_names{f1}, feature_names{f2}));
    axis tight;
    grid on;
end

lgd = legend({'Setosa', 'Versicolor', 'Virginica'}, ...
    'Orientation', 'horizontal');
lgd.Position(1) = 0.35;
lgd.Position(2) = 0.02;