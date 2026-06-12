%% Step 3: Forward Propagation Function
% This function computes the forward pass for the network
function [A1, A2, Z1, Z2] = forward(X, W1, b1, W2, b2)
    Z1 = X * W1 + b1; % Affine transformation for first layer % Z1 = X 120x4 W 4x1x3
    A1 = 1 ./ (1 + exp(-Z1)); % Sigmoid activation function
    Z2 = A1 * W2 + b2; % Affine transformation for second layer 
    A2 = exp(Z2) ./ sum(exp(Z2), 2); % Softmax activation function (output layer) A2 120x1x3
end