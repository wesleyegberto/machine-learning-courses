function [theta, J_history] = gradientDescentLogistic(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %

    h_theta = sigmoid(X * theta);
    
    nabla = (1 / m) * X' * (h_theta - y);
    theta = theta - alpha * nabla; % don't need to transpose as nabla is already transposed

    % ============================================================

    % Save the cost J in every iteration    
    [cost, grad] = costFunction(theta, X, y);
    J_history(iter) = cost;

end

end
