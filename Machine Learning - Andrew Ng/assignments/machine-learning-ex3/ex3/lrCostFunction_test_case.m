theta = [-2; -1; 1; 2];
X = [ones(5,1) reshape(1:15,5,3)/10];
y = [1;0;1;0;1] >= 0.5;       % creates a logical array

% test the unregularized results
[J grad] = lrCostFunction(theta, X, y, 0)

% J =  0.73482
% grad =
%    0.146561
%    0.051442
%    0.124722
%    0.198003


% test the regularized results
lambda = 3;
[J grad] = lrCostFunction(theta, X, y, lambda)

% results
% J =  2.5348
% grad =
%    0.14656
%   -0.54856
%    0.72472
%    1.39800