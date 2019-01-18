function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%


% % Calculate the cost using iteration
% e = 0; % sum of errors
% for i = 1:m
%     h = sigmoid(X(i,:) * theta);
%     e = e + (-y(i) * log(h) - (1 - y(i)) * log(1 - h));
% endfor
% J = e / m;

% % Calculate the gradient for Theta_0 using iteration
% e = 0;
% for i = 1:m
%     e += (sigmoid(X(i,:) * theta) - y(i));
% endfor
% grad(1) = e / m;

% % Calculate the gradient for Theta_1 ... Theta_n using iteration
% for j = 2:size(grad)
%     e = 0;
%     for i = 1:m
%         e += (sigmoid(X(i,:) * theta) - y(i)) * X(i,j);
%     endfor
%     grad(j) = e / m;
% endfor


% Calculate using matrix form
h = sigmoid(X * theta); % x_0 = 1 and X is row-wise
J = (1 / m) * ( (-y)' * log(h) - (1 - y)' * log(1 - h) );
grad = (1 / m) * X' * (sigmoid(X * theta) - y);

% =============================================================

end
