function [all_theta, all_J_history] = oneVsAll_GD(X, y, num_labels, alpha, lambda, num_iters)
% ONEVSALL_GD trains multiple logistic regression classifiers using
% Gradient Descent and returns all
% the classifiers in a matrix all_theta, where the i-th row of all_theta 
% corresponds to the classifier for label i

% Some useful variables
m = size(X, 1);
n = size(X, 2);

% You need to return the following variables correctly 
all_theta = zeros(num_labels, n + 1);

all_J_history = zeros(num_labels, num_iters);

% Add ones to the X data matrix
X = [ones(m, 1) X];

% =========================================================================
% Custom just to compare GD with fmincg

for k = 1:num_labels
    fprintf('Training classifier for label %d\n', k);
    initial_theta = zeros(n + 1, 1);

    % This function will return theta and the cost history
    [theta, J_history] = lrGradientDescent(X, (y == k), initial_theta, alpha, lambda, num_iters);
    all_J_history(k, 1:end) = J_history;

    plot(1:numel(J_history), J_history, 'LineWidth', 2);
    
    all_theta(k, 1:end) = theta;
end

% =========================================================================


end
