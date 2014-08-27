function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

% create list of parameters to look at
C_list = [.01 .03 .1 .3 1 3 10 30];
sigma_list = [.01 .03 .1 .3 1 3 10 30];
% initialize values for the for loop
m = length(C_list);
n = length(sigma_list);
% create empty list to keep track of error values
error_list = zeros(m, n);


for i = 1:m
	for j = 1:n
		
		% train the model
		temp_model = svmTrain(X, y, C_list(i), @(x1, x2) gaussianKernel(x1, x2, sigma_list(j)));
		% predict the model on cv data
		predictions = svmPredict(temp_model, Xval);
		% calculate the error
		error = mean(double(predictions ~= yval));
		% store the error in a matrix
		error_list(i, j) = error;
	end
end

% find the combo of C and sigma that gives the lowest error
[xval, yval] = find(error_list==min(min(error_list)));
C = C_list(xval);
sigma = sigma_list(yval);





% =========================================================================

end