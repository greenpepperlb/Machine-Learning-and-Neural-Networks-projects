%% Step 4: Loss Function
% This function calculates the categorical cross-entropy loss
function L = compute_loss(A2, Y)
    m = size(Y, 1); % Number of training examples
    L = -sum(sum(Y .* log(A2))) / m; % Categorical cross-entropy loss
end