%% Step 6: Gradient Descent Function
% This function performs the gradient descent update for the weights and biases
function [W1, b1, W2, b2, L] = gradient_descent(X, Y, W1, b1, W2, b2, learning_rate)
    % Forward pass
    [A1, A2, Z1, Z2] = forward(X, W1, b1, W2, b2);
    
    % Compute the loss
    L = compute_loss(A2, Y);
    
    % Backpropagation
    [dW1, db1, dW2, db2] = backprop(X, Y, A1, A2, Z1, Z2, W2);
    
    % Update the parameters using gradient descent
    W1 = W1 - learning_rate * dW1;
    b1 = b1 - learning_rate * db1;
    W2 = W2 - learning_rate * dW2;
    b2 = b2 - learning_rate * db2;
    
     % Return the loss for monitoring
end