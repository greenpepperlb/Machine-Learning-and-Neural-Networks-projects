%% Step 5: Backpropagation Function
% This function computes the gradients for backpropagation
function [dW1, db1, dW2, db2] = backprop(X, Y, A1, A2, Z1, Z2, W2)
    m = size(X, 1); % Number of training examples
    
    % Gradient for second layer (output layer)
    dZ2 = A2 - Y; % Derivative of softmax loss
    dW2 = (A1' * dZ2) / m; % Gradient for weights of second layer
    db2 = sum(dZ2) / m; % Gradient for biases of second layer
    
    % Gradient for first layer (hidden layer)
    dZ1 = (dZ2 * W2') .* (A1 .* (1 - A1)); % Derivative of sigmoid activation
    dW1 = (X' * dZ1) / m; % Gradient for weights of first layer
    db1 = sum(dZ1) / m; % Gradient for biases of first layer
end