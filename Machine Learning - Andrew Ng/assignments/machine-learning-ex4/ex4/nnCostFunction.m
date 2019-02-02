function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%



% ====================== Part 1

% Feed Forward - Loop way
% for i = 1:m
%     % === Activations from Input Layer (features) ===
%     X = [1; X(i,:)']; % add a1_0 = 1
%     
% 
%     % === Activations from Hidden Layer ===
%     k = size(Theta1, 1); % qty units in Hidden Layer
%     a2 = zeros(k, 1);
%     
%     % Loop through Hidden Layer's units
%     for j = 1:k
%         z2_j = Theta1(j,:) * X;
%         a2(j) = sigmoid(z2_j);
%     end
%     a2 = [1; a2]; % add a2_0 = 1
% 
% 
%     % === Activations from Output Layer ===
%     k = size(Theta2, 1); % qty units in Output Layer
%     a3 = zeros(k, 1);
% 
%     % Loop through Output Layer's units
%     for j = 1:k
%         z3_j = Theta2(j,:) * a2;
%         a3(j) = sigmoid(z3_j);
%     end
% 
% 
%     % === softmax from our output (the index is our classification class) ===
%     [_ p(i)] = max(a3', [], 2);
% end


% Feed Forward - Vectorized way

% === Activations from Input Layer (features) ===
X = [ones(m, 1) X]; % add a1_0 = 1

% === Activations from Hidden Layer ===
z2 = X * Theta1';
a2 = sigmoid(z2);
a2 = [ones(m, 1) a2]; % add a2_0 = 1

% === Activations from Output Layer ===
z2 = a2 * Theta2';
a3 = sigmoid(z2);



h = a3;
totalError = 0;
regParam = 0;

% % Compute the cost - Loop way (too slow)
% for i = 1:m
%     for k = 1:num_labels
%         % y(i) == k will only be 1 when k-th unit is y_i
%         totalError = totalError + ((y(i) == k) * log(h(i,k)) + (1 - (y(i) == k)) * log(1 - h(i,k)));
%     end
% end
% % regularization
% [J K] = size(Theta1);
% for j = 1:J
%     for k = 1:K
%         regParam = regParam + Theta1(j, k) ** 2;
%     end
% end
% 
% [J K] = size(Theta2);
% for j = 1:J
%     for k = 1:K
%         regParam = regParam + Theta2(j, k) ** 2;
%     end
% end


%% Compute the cost - Vector way
% for i = 1:m
%     % generate the y_i values at output layer - only the k-th unit respect to y_i will be 1
%     y_k = y(i) == linspace(1, num_labels, num_labels);
% 
%     totalError = totalError + sum(y_k .* log(h(i,:)) + (1 - y_k) .* log(1 - h(i,:)));
% end
% % regularization
% J = size(Theta1, 1);
% for j = 1:J
%     regParam = regParam + (Theta1(j, :) * Theta1(j, :)');
% end
% 
% J = size(Theta2, 1);
% for j = 1:J
%     regParam = regParam + (Theta2(j, :) * Theta2(j, :)');
% end


% Compute the cost - Matrix way
% generate the matrix y with the output from output layer - only the k-th unit respect to y_i will be 1
y_k = y == linspace(1, num_labels, num_labels);

outputs_error = y_k .* log(h) + (1 - y_k) .* log(1 - h); % result in matrix (m)x(num_labels)
totalError = sum(sum(outputs_error));
% % regularization
regParam = sum(sum(Theta1 .* Theta1)) + sum(sum(Theta2 .* Theta2));



J = ((-1 / m) * totalError) + (lambda / (2 * m) * regParam);


% ====================== Part 2



% ====================== Part 3




% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
