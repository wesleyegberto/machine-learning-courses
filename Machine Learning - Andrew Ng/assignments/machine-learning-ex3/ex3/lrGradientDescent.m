function [theta, J_history] = lrGradientDescent(X, y, theta, alpha, lambda, num_iters)
% Performs gradient descent to learn theta
%   theta = lrGradientDescent(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

% ============================================================

for iter = 1:num_iters

    h_theta = sigmoid(X * theta);
    
    nabla = (1 / m) * X' * (h_theta - y);

    % Apply regularization
    nabla(2:end) = nabla(2:end) + (lambda / m) * theta(2:end);

    theta = theta - alpha * nabla; % don't need to transpose as nabla is already transposed


    % Save the cost J in every iteration    
    [cost, grad] = lrCostFunction(theta, X, y, lambda);
    J_history(iter) = cost;

end

% ============================================================
end
