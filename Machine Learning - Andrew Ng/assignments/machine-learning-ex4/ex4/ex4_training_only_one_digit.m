% This variation will be trained to recognize only digit 0
% We can see how the NN will look

%% Initialization
clear ; close all; clc

%% Setup the parameters you will use for this exercise
input_layer_size  = 400;  % 20x20 Input Images of Digits
hidden_layer_size = 25;   % 25 hidden units
num_labels = 10;          % 10 labels, from 1 to 10
                          % (note that we have mapped "0" to label 10)


fprintf('Loading and Visualizing Data ...\n')
load('ex4data1.mat');


% Digit to train
digit_train = 3;

% the training set has 500 inputs for each digit
digit_start = digit_train * 500 + 1;
digit_end = digit_start + 499;

% Extract only the training set for selected digit
X = X(digit_start:digit_end,:);
y = y(digit_start:digit_end);

m = size(X, 1);

% Randomly select 100 data points to display
sel = randperm(size(X, 1));
sel = sel(1:100);

displayData(X(sel, :));

fprintf('Program paused. Press enter to continue.\n');
pause;



fprintf('\nInitializing Neural Network Parameters ...\n');

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);

% Unroll parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];



fprintf('\nTraining Neural Network for Digit %d... \n', digit_train);
%  After you have completed the assignment, change the MaxIter to a larger
%  value to see how more training helps.
options = optimset('MaxIter', 400);

%  You should also try different values of lambda
lambda = 1;

% Create "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunction(p, input_layer_size, hidden_layer_size, num_labels, X, y, lambda);

% Now, costFunction is a function that takes in only one argument (the neural network parameters)
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

fprintf('Program paused. Press enter to continue.\n');
pause;


fprintf('\nVisualizing Neural Network Hidden Layer... \n');
displayData(Theta1(:, 2:end));
pause;
fprintf('\nProgram paused. Press enter to continue.\n');

fprintf('\nVisualizing Neural Network Output Layer... \n');
displayData(Theta2(:, 2:end));
fprintf('\nProgram paused. Press enter to continue.\n');
pause;


fprintf('\nTesting Neural Network... \n');
pred = predict(Theta1, Theta2, X);

fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);
