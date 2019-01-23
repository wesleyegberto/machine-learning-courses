function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

% Loop way
for i = 1:m
    % === Activations from Input Layer (features) ===
    a1 = [1; X(i,:)']; % add a1_0 = 1
    

    % === Activations from Hidden Layer ===
    k = size(Theta1, 1); % qty units in Hidden Layer
    a2 = zeros(k, 1);
    
    % Loop through Hidden Layer's units
    for j = 1:k
        z2_j = Theta1(j,:) * a1;
        a2(j) = sigmoid(z2_j);
    end
    a2 = [1; a2]; % add a2_0 = 1


    % === Activations from Output Layer ===
    k = size(Theta2, 1); % qty units in Output Layer
    a3 = zeros(k, 1);

    % Loop through Output Layer's units
    for j = 1:k
        z3_j = Theta2(j,:) * a2;
        a3(j) = sigmoid(z3_j);
    end


    % === softmax from our output (the index is our classification class) ===
    [_ p(i)] = max(a3', [], 2);
end



% Add 1's at input layer
% X = [ones(m, 1) X];


% =========================================================================


end
