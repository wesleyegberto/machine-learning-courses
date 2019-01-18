function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

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


% % Calculate the cost using iteration
% e = 0; % sum of errors
% for i = 1:m
%     h = sigmoid(X(i,:) * theta);
%     e = e + (-y(i) * log(h) - (1 - y(i)) * log(1 - h));
% endfor
% pC = 0;
% % sum the square of parameters cost
% for i = 2:size(grad)
%     pC += theta(i) ^ 2;
% endfor
% J = e / m + (lambda / (2 * m) * pC);

% % Calculate the gradient for Theta_0 using iteration
% e = 0;
% for i = 1:m
%     e += (sigmoid(X(i,:) * theta) - y(i));
% endfor
% grad(1) = e / m;

% % Calculate the gradient for Theta_1 ... Theta_n applying regularization
% for j = 2:size(grad)
%     e = 0;
%     for i = 1:m
%         e += (sigmoid(X(i,:) * theta) - y(i)) * X(i,j);
%     endfor
%     grad(j) = e / m + (lambda / m * theta(j));
% endfor


% Calculate using matrix form

h = sigmoid(X * theta); % x_0 = 1 and X is row-wise
n = size(grad);
J = (1 / m) * ( (-y)' * log(h) - (1 - y)' * log(1 - h) ) + (lambda / (2 * m) * (theta(2:n)' * theta(2:n)));
grad_0 = (1 / m) * X(:,1)' * (sigmoid(X * theta) - y);
grad = ((1 / m) * X' * (sigmoid(X * theta) - y)) + (lambda / m * theta);
grad(1) = grad_0;

% =============================================================

end
