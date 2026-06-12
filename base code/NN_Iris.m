%% Notes
%Standardize the parameters to have a unitless parameter


%% Help

% dlarray/sigmoid - Apply sigmoid activation
%     The sigmoid activation operation applies the sigmoid function to the
%     input data.
% 
%     Syntax
%       Y = sigmoid(X)
% 
%     Input Arguments
%       X - Input data
%         dlarray
% 
%     Output Arguments
%       Y - Sigmoid activations
%         dlarray
% 
% 
% 
% softmax - Softmax transfer function
%     This MATLAB function takes a S-by-Q matrix of net input (column)
%     vectors, N, and returns the S-by-Q matrix, A, of the softmax competitive
%     function applied to each column of N.
% 
%     Syntax
%       A = softmax(N)
%       info = softmax(code)
% 
%     Input Arguments
%       N - Input matrix
%         matrix
%       code - Information option
%         'name' | 'output' | 'active' | 'fullderiv' | 'fpnames' |
%         'fpdefaults'
% 
%     Output Arguments
%       A - Output matrix
%         matrix
%       info - Information output
%         string | vector | scalar
% 
% 
% 
%  onehotdecode - Decode probability vectors into class labels
%     This MATLAB function decodes each probability vector in B to the most
%     probable class label from the labels specified by classes.
% 
%     Syntax
%       A = onehotdecode(B,classes,featureDim)
%       A = onehotdecode(B,classes,featureDim,typename)
% 
%     Input Arguments
%       B - Probability vectors
%         numeric array
%       classes - Classes
%         cell array | string vector | numeric vector | character array
%       featureDim - Dimension containing probability vectors
%         positive integer
%       typename - Data type of decoded labels
%         'categorical' (default) | character vector | string scalar
% 
%     Output Arguments
%       A - Decoded class labels
%         categorical array | string array | numeric array
% 
% 
% 
%      Syntax
%       perf = crossentropy(net,targets,outputs,perfWeights)
%       perf = crossentropy(___,Name,Value)
% 
%     Input Arguments
%       net - neural network
%         network object
%       targets - neural network target values
%         matrix or cell array of numeric values
%       outputs - neural network output values
%         matrix or cell array of numeric values
%       perfWeights - performance weights
%         {1} (default) | vector or cell array of numeric values
% 
%     Name-Value Arguments
%       regularization - proportion of performance attributed to weight/bias values
%         0 (default) | numeric value in the range (0,1)
%       normalization - Normalization mode for outputs, targets, and errors
%         'none' (default) | 'standard' | 'percent'
% 
%     Output Arguments
%       perf - network performance
%         double

load fisheriris.mat
y_train = [species(1:40), species(51:90), species(101:140)];
y_test = [species(41:50), species(91:100), species(141:150)];

y_train = categorical(y_train);
y_test = categorical(y_test);


%sigmoid

%eq 8
dZ2 = A2 - Y;
%eq 9
m = size(Y, 1);
dW2 = (A1') * dZ2 / m;
db2 = sum(dZ2, 1) / m;  % sum over rows (samples)
%eq 10 
sigmoid_derivative = A1 .* (1 - A1);  % Since A1 = sigmoid(Z1)
dZ1 = (dZ2 * W2') .* sigmoid_derivative;
%eq 11

dW1 = (X') * dZ1 / m;
db1 = sum(dZ1, 1) / m;
