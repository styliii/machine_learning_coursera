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


values = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
prediction_errors = zeros(8, 8);

for i = 1:8;
  C = values(i);

  for j = 1:8;
    sigma = values(j);

    model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
    predictions = svmPredict(model, Xval);
    prediction_error = mean(double(predictions ~= yval));
    prediction_errors(i, j) = prediction_error;

    j += 1;
  end

  i += 1;
end

[x, ix] = min(prediction_errors);
[y, iy] = min(x);

prediction_errors

min_ix = ix(iy);
min_iy = iy;

C = values(min_ix);
sigma = values(min_iy);



% =========================================================================

end
